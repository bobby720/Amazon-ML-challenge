"""LightGBM training pipeline with F_0.5 threshold optimization."""

import json
import logging
import gc
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from .config import CONFIG, AppConfig
from .preprocess import preprocess_polars_df
from .blocking import MultiPassBlocker
from .feature_engine import FeatureEngine, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F-beta score (default beta=0.5 for 2x precision weighting)."""
    if precision + recall == 0 or precision == 0 or recall == 0:
        return 0.0
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return float(f_beta)


def evaluate_predictions_f05(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    beta: float = 0.5,
) -> Dict[str, float]:
    """Calculate TP, FP, FN, Precision, Recall, and F_beta at a given probability threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_beta = compute_f_beta_score(precision, recall, beta=beta)

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_grid: np.ndarray,
    beta: float = 0.5,
) -> Tuple[float, float, Dict[str, Any]]:
    """Grid search over probability thresholds to maximize F_0.5 score."""
    best_thresh = 0.75
    best_f_score = -1.0
    best_metrics = {}

    for thresh in threshold_grid:
        metrics = evaluate_predictions_f05(y_true, y_probs, thresh, beta=beta)
        if metrics["f_beta"] > best_f_score:
            best_f_score = metrics["f_beta"]
            best_thresh = thresh
            best_metrics = metrics

    logger.info(
        f"Optimal Threshold Found: {best_thresh:.4f} | "
        f"F_0.5: {best_metrics.get('f_beta', 0.0):.4f} | "
        f"Precision: {best_metrics.get('precision', 0.0):.4f} | "
        f"Recall: {best_metrics.get('recall', 0.0):.4f}"
    )
    return best_thresh, best_f_score, best_metrics


def load_ground_truth_map(gt_path: Path) -> Dict[str, Set[str]]:
    """Parse ground truth TSV into a mapping: {source1_entity_id: set(matched_candidate_ids)}."""
    logger.info(f"Loading ground truth from: {gt_path}")
    df_gt = pl.read_csv(gt_path, separator="\t", has_header=True)
    cols = df_gt.columns
    s1_col = cols[0]
    matched_col = cols[1]

    gt_map: Dict[str, Set[str]] = {}
    for row in df_gt.iter_rows(named=True):
        s1_id = str(row[s1_col]).strip()
        matched_str = str(row[matched_col]) if row[matched_col] is not None else ""
        if matched_str and matched_str != "None" and matched_str != "null":
            # Split on comma or space
            matches = set(
                m.strip() for m in matched_str.replace(" ", ",").split(",") if m.strip()
            )
            gt_map[s1_id] = matches
        else:
            gt_map[s1_id] = set()

    return gt_map


class EntityResolutionTrainer:
    """Trainer class orchestrating dataset construction, training, and threshold optimization."""

    def __init__(self, config: AppConfig = CONFIG):
        self.config = config
        self.blocker = MultiPassBlocker(self.config.blocking)

    def train_pipeline(self) -> Dict[str, Any]:
        """Execute end-to-end training and evaluation pipeline."""
        paths = self.config.paths
        paths.ensure_directories()

        logger.info("Step 1: Reading and preprocessing training source files...")
        df_s1 = pl.read_csv(paths.TRAIN_SOURCE1_PATH, separator="\t", has_header=True)
        df_s2 = pl.read_csv(paths.TRAIN_SOURCE2_PATH, separator="\t", has_header=True)
        df_s3 = pl.read_csv(paths.TRAIN_SOURCE3_PATH, separator="\t", has_header=True)

        df_s1_proc = preprocess_polars_df(df_s1)
        df_s2_proc = preprocess_polars_df(df_s2)
        df_s3_proc = preprocess_polars_df(df_s3)
        df_cand_proc = pl.concat([df_s2_proc, df_s3_proc])

        logger.info("Step 2: Generating candidate pairs via multi-pass blocking...")
        candidate_pairs_df = self.blocker.block_candidates(df_s1_proc, df_cand_proc)
        
        # Also export training candidate pairs
        self.blocker.export_candidate_pairs(candidate_pairs_df, paths.CANDIDATE_PAIRS_PATH)

        logger.info("Step 3: Loading ground truth and assigning binary labels...")
        gt_map = load_ground_truth_map(paths.TRAIN_GROUND_TRUTH_PATH)

        # Build labels
        s1_ids = candidate_pairs_df["source1_entity_id"].to_list()
        cand_ids = candidate_pairs_df["candidate_entity_id"].to_list()

        labels = np.array([
            1 if cand_ids[i] in gt_map.get(s1_ids[i], set()) else 0
            for i in range(len(s1_ids))
        ], dtype=np.int32)

        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        logger.info(f"Dataset summary: {len(labels):,} candidate pairs (Positives: {num_pos:,}, Negatives: {num_neg:,})")

        logger.info("Step 4: Extracting feature matrix...")
        X, feature_names = FeatureEngine.build_feature_matrix(
            candidate_pairs_df, df_s1_proc, df_cand_proc
        )

        logger.info("Step 5: Training LightGBM with Group-K-Fold validation...")
        # Group by S1 entity to avoid data leakage
        groups = np.array(s1_ids)
        gkf = GroupKFold(n_splits=min(self.config.model.N_SPLITS, len(np.unique(groups))))

        oof_preds = np.zeros(len(labels), dtype=np.float32)
        models: List[lgb.Booster] = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, labels, groups=groups)):
            logger.info(f"Training Fold {fold + 1}...")
            X_train, y_train = X[train_idx], labels[train_idx]
            X_val, y_val = X[val_idx], labels[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)

            booster = lgb.train(
                params=self.config.model.LGBM_PARAMS,
                train_set=dtrain,
                valid_sets=[dtrain, dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )

            val_preds = booster.predict(X_val)
            oof_preds[val_idx] = val_preds
            models.append(booster)

        logger.info("Step 6: Optimizing probability threshold for F_0.5 metric...")
        best_thresh, best_f, best_metrics = find_optimal_threshold(
            labels, oof_preds, self.config.model.THRESHOLD_GRID, beta=self.config.model.F_BETA
        )

        logger.info("Step 7: Retraining final model on complete dataset...")
        dall = lgb.Dataset(X, label=labels, feature_name=feature_names)
        final_model = lgb.train(
            params=self.config.model.LGBM_PARAMS,
            train_set=dall,
        )

        # Save model booster
        final_model.save_model(str(paths.MODEL_PATH))
        logger.info(f"Model saved to: {paths.MODEL_PATH}")

        # Save metadata and best threshold
        metadata = {
            "best_threshold": float(best_thresh),
            "best_f_0_5_score": float(best_f),
            "validation_metrics": best_metrics,
            "feature_names": feature_names,
            "total_pairs_trained": int(len(labels)),
            "positive_pairs": int(num_pos),
            "negative_pairs": int(num_neg),
        }
        with open(paths.CONFIG_ARTIFACT_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved to: {paths.CONFIG_ARTIFACT_PATH}")

        # Clean memory
        del X, labels, oof_preds, df_s1_proc, df_s2_proc, df_s3_proc, df_cand_proc
        gc.collect()

        return metadata


if __name__ == "__main__":
    trainer = EntityResolutionTrainer()
    trainer.train_pipeline()
