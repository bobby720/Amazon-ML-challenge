"""Configuration module for Multi-Source Business Entity Resolution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


@dataclass
class PathConfig:
    """Directory and file path definitions."""
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Raw training dataset paths
    TRAIN_SOURCE1_PATH: Path = DATA_DIR / "train_source1.tsv"
    TRAIN_SOURCE2_PATH: Path = DATA_DIR / "train_source2.tsv"
    TRAIN_SOURCE3_PATH: Path = DATA_DIR / "train_source3.tsv"
    TRAIN_GROUND_TRUTH_PATH: Path = DATA_DIR / "train_ground_truth.tsv"

    # Raw test dataset paths
    TEST_SOURCE1_PATH: Path = DATA_DIR / "test_source1.tsv"
    TEST_SOURCE2_PATH: Path = DATA_DIR / "test_source2.tsv"
    TEST_SOURCE3_PATH: Path = DATA_DIR / "test_source3.tsv"

    # Output artifact paths
    CANDIDATE_PAIRS_PATH: Path = OUTPUT_DIR / "candidate_pairs.tsv"
    MATCHING_RESULTS_PATH: Path = OUTPUT_DIR / "matching_results.tsv"
    MODEL_PATH: Path = MODELS_DIR / "lgbm_entity_resolver.txt"
    CONFIG_ARTIFACT_PATH: Path = MODELS_DIR / "pipeline_metadata.json"

    def ensure_directories(self) -> None:
        """Ensure all output and model directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BlockingConfig:
    """Parameters for deterministic inverted-index blocking."""
    MAX_CANDIDATES_PER_S1: int = 20
    MIN_TOKEN_LENGTH_NAME: int = 2
    BLOCKING_PASSES: List[str] = field(default_factory=lambda: [
        "exact_name",
        "name_2token_postal",
        "name_1token_location",
    ])
    CHUNK_SIZE: int = 100_000


@dataclass
class ModelConfig:
    """Hyperparameters and settings for LightGBM model training."""
    F_BETA: float = 0.5
    N_SPLITS: int = 5
    RANDOM_STATE: int = 42
    DEFAULT_THRESHOLD: float = 0.75
    THRESHOLD_GRID: np.ndarray = field(
        default_factory=lambda: np.round(np.linspace(0.50, 0.95, 46), 4)
    )

    LGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 500,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })


@dataclass
class AppConfig:
    """Master application configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        self.paths.ensure_directories()


CONFIG = AppConfig()
