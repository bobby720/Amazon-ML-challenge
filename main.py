"""Main CLI entrypoint for Amazon ML Challenge 2026: Multi-Source Business Entity Resolution."""

import argparse
import sys
import logging
from pathlib import Path

# Add code directory to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from business_entity_resolution.config import CONFIG
from business_entity_resolution.train import EntityResolutionTrainer
from business_entity_resolution.inference import EntityResolutionInference
from validate_submission import validate_candidate_pairs, validate_matching_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MainCLI")


def run_training():
    """Execute training and F_0.5 threshold optimization."""
    logger.info("Starting LightGBM training pipeline...")
    trainer = EntityResolutionTrainer()
    metadata = trainer.train_pipeline()
    logger.info(f"Training completed successfully! Optimal F_0.5 Threshold: {metadata['best_threshold']:.4f}")


def run_inference():
    """Execute candidate blocking and inference on test set."""
    logger.info("Starting test inference and candidate blocking...")
    inferer = EntityResolutionInference()
    inferer.predict_test_dataset()
    logger.info("Inference completed successfully. Submission files generated.")


def run_validation():
    """Validate output artifacts."""
    logger.info("Running submission validation integrity checks...")
    validate_candidate_pairs(CONFIG.paths.CANDIDATE_PAIRS_PATH)
    validate_matching_results(CONFIG.paths.MATCHING_RESULTS_PATH, CONFIG.paths.TEST_SOURCE1_PATH)
    logger.info("Validation complete.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Source Business Entity Resolution Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available pipeline stages")

    subparsers.add_parser("train", help="Run multi-pass blocking, feature generation, LightGBM training & F_0.5 tuning")
    subparsers.add_parser("infer", help="Generate candidates, compute features, score test set & build submission TSV")
    subparsers.add_parser("validate", help="Run strict integrity checks on submission TSVs")
    subparsers.add_parser("all", help="Run full pipeline: train -> infer -> validate")

    args = parser.parse_args()

    if args.command == "train":
        run_training()
    elif args.command == "infer":
        run_inference()
        run_validation()
    elif args.command == "validate":
        run_validation()
    elif args.command == "all":
        run_training()
        run_inference()
        run_validation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
