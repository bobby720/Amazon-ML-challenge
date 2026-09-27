"""Inference pipeline for batch candidate blocking, scoring, and output generation."""

import json
import logging
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import polars as pl
import lightgbm as lgb

from .config import CONFIG, AppConfig
from .preprocess import preprocess_polars_df
from .blocking import MultiPassBlocker
from .feature_engine import FeatureEngine
from .postprocess import build_submission_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EntityResolutionInference:
    """End-to-end inference engine for test datasets."""

    def __init__(self, config: AppConfig = CONFIG):
        self.config = config
        self.blocker = MultiPassBlocker(self.config.blocking)

    def load_model_and_threshold(self) -> tuple[lgb.Booster, float]:
        """Load trained LightGBM booster and the optimal F_0.5 threshold."""
        model_path = self.config.paths.MODEL_PATH
        metadata_path = self.config.paths.CONFIG_ARTIFACT_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")

        logger.info(f"Loading model booster from: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))

        threshold = self.config.model.DEFAULT_THRESHOLD
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                threshold = meta.get("best_threshold", threshold)
                logger.info(f"Loaded optimal threshold from metadata: {threshold:.4f}")
        else:
            logger.warning(f"Metadata file not found. Using default threshold: {threshold:.4f}")

        return booster, threshold

    def predict_test_dataset(
        self,
        test_s1_path: Optional[Path] = None,
        test_s2_path: Optional[Path] = None,
        test_s3_path: Optional[Path] = None,
    ) -> pl.DataFrame:
        """Run blocking, feature computation, and scoring on test sources."""
        paths = self.config.paths
        s1_file = test_s1_path or paths.TEST_SOURCE1_PATH
        s2_file = test_s2_path or paths.TEST_SOURCE2_PATH
        s3_file = test_s3_path or paths.TEST_SOURCE3_PATH

        logger.info(f"Reading test sources:\n S1: {s1_file}\n S2: {s2_file}\n S3: {s3_file}")
        df_s1 = pl.read_csv(s1_file, separator="\t", has_header=True)
        df_s2 = pl.read_csv(s2_file, separator="\t", has_header=True)
        df_s3 = pl.read_csv(s3_file, separator="\t", has_header=True)

        # Preprocess test datasets
        logger.info("Preprocessing test entities...")
        df_s1_proc = preprocess_polars_df(df_s1)
        df_s2_proc = preprocess_polars_df(df_s2)
        df_s3_proc = preprocess_polars_df(df_s3)
        df_cand_proc = pl.concat([df_s2_proc, df_s3_proc])

        all_s1_ids = df_s1_proc["entity_id"].to_list()

        # Generate candidate pairs via multi-pass blocking
        logger.info("Executing candidate blocking on test set...")
        candidate_pairs_df = self.blocker.block_candidates(df_s1_proc, df_cand_proc)

        # Always export candidate pairs artifact
        self.blocker.export_candidate_pairs(candidate_pairs_df, paths.CANDIDATE_PAIRS_PATH)

        if candidate_pairs_df.height == 0:
            logger.warning("No candidate pairs generated during test blocking.")
            build_submission_file(all_s1_ids, None, paths.MATCHING_RESULTS_PATH)
            return candidate_pairs_df

        # Build feature matrix
        logger.info("Computing test pair features...")
        X_test, _ = FeatureEngine.build_feature_matrix(
            candidate_pairs_df, df_s1_proc, df_cand_proc
        )

        # Load model and score candidates
        booster, threshold = self.load_model_and_threshold()
        logger.info("Scoring candidate pairs with LightGBM booster...")
        probs = booster.predict(X_test)

        # Append probabilities to candidate pairs
        candidate_pairs_with_probs = candidate_pairs_df.with_columns([
            pl.Series("probability", probs)
        ])

        # Filter candidates above threshold
        logger.info(f"Filtering candidates with probability >= {threshold:.4f} (F_0.5 optimal)...")
        matched_pairs_df = candidate_pairs_with_probs.filter(pl.col("probability") >= threshold)
        logger.info(f"Retained {matched_pairs_df.height:,} matched pairs.")

        # Build final submission TSV
        build_submission_file(all_s1_ids, matched_pairs_df, paths.MATCHING_RESULTS_PATH)

        # Memory cleanup
        del df_s1_proc, df_s2_proc, df_s3_proc, df_cand_proc, X_test, probs
        gc.collect()

        return matched_pairs_df


if __name__ == "__main__":
    inference_runner = EntityResolutionInference()
    inference_runner.predict_test_dataset()
