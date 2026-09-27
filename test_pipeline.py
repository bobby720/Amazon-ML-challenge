import sys
import logging
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

import polars as pl
from business_entity_resolution.config import CONFIG
from business_entity_resolution.train import EntityResolutionTrainer
from business_entity_resolution.inference import EntityResolutionInference
from validate_submission import validate_candidate_pairs, validate_matching_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IntegrationTest")


def generate_synthetic_data(data_dir: Path):
    """Generate realistic synthetic multi-source business entity data for US, India, and France."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Source 1 (Reference ground entities)
    train_s1 = pl.DataFrame({
        "source1_entity_id": [f"S1-{i:04d}" for i in range(1, 11)],
        "business_name": [
            "Amazon Web Services Inc",
            "Starbucks Coffee Company",
            "Infosys Limited",
            "Tata Consultancy Services Pvt Ltd",
            "TotalEnergies SE",
            "LVMH Moët Hennessy Louis Vuitton SA",
            "Walmart Supercenter Inc",
            "Reliance Retail Limited",
            "Carrefour Market",
            "Microsoft Corporation",
        ],
        "business_address": [
            "410 Terry Ave N, Seattle, WA 98109",
            "2401 Utah Avenue South, Seattle, WA 98134",
            "Plot 44, Electronic City, Hosur Road, Bangalore, 560100",
            "TCS House, Raveline Street, Fort, Mumbai, 400001",
            "2 Place Jean Millier, La Defense 6, Courbevoie, 92400",
            "22 Avenue Montaigne, Paris, 75008",
            "702 SW 8th St, Bentonville, AR 72716",
            "Maker Chambers IV, Nariman Point, Mumbai, 400021",
            "93 Avenue de Paris, Massy, 91300",
            "One Microsoft Way, Redmond, WA 98052",
        ],
        "city": ["Seattle", "Seattle", "Bangalore", "Mumbai", "Courbevoie", "Paris", "Bentonville", "Mumbai", "Massy", "Redmond"],
        "state": ["WA", "WA", "Karnataka", "Maharashtra", "Île-de-France", "Île-de-France", "AR", "Maharashtra", "Île-de-France", "WA"],
        "country": ["US", "US", "IND", "IND", "FRA", "FRA", "USA", "India", "France", "US"],
        "postal_code": ["98109", "98134", "560100", "400001", "92400", "75008", "72716", "400021", "91300", "98052"],
    })

    # 2. Train Source 2 (Matches + variations)
    train_s2 = pl.DataFrame({
        "source2_entity_id": [f"S2-{i:04d}" for i in range(1, 11)],
        "business_name": [
            "Amazon Web Services",
            "Starbucks Coffee",
            "Infosys Ltd.",
            "Tata Consultancy Services",
            "Total Energies",
            "LVMH Louis Vuitton",
            "Walmart",
            "Reliance Retail",
            "Carrefour Supermarche",
            "Microsoft Corp",
        ],
        "business_address": [
            "410 Terry Avenue North, Seattle, WA",
            "2401 Utah Ave S, Seattle, WA 98134",
            "Electronic City, Hosur Rd, Bangalore, 560100",
            "Raveline St, Fort, Mumbai 400001",
            "2 Pl. Jean Millier, Courbevoie, 92400",
            "22 Ave Montaigne, Paris, 75008",
            "702 SW 8th Street, Bentonville, AR 72716",
            "Maker Chambers 4, Nariman Pt, Mumbai 400021",
            "93 Ave de Paris, Massy 91300",
            "1 Microsoft Way, Redmond, WA 98052",
        ],
        "city": ["Seattle", "Seattle", "Bangalore", "Mumbai", "Courbevoie", "Paris", "Bentonville", "Mumbai", "Massy", "Redmond"],
        "state": ["WA", "WA", "Karnataka", "Maharashtra", "IDF", "IDF", "AR", "MH", "IDF", "WA"],
        "country": ["US", "USA", "India", "IND", "FR", "France", "US", "IND", "FRA", "US"],
        "postal_code": ["98109", "98134", "560100", "400001", "92400", "75008", "72716", "400021", "91300", "98052"],
    })

    # 3. Train Source 3 (Additional matches + distractors)
    train_s3 = pl.DataFrame({
        "source3_entity_id": [f"S3-{i:04d}" for i in range(1, 11)],
        "business_name": [
            "AWS Cloud Solutions",
            "Starbucks Reserve",
            "Infosys Technologies",
            "TCS Tata",
            "Total Gas & Power",
            "Moët Hennessy LVMH",
            "Target Store 101",  # Distractor
            "D-Mart Avenue Supermarts",  # Distractor
            "Auchan Hypermarket",  # Distractor
            "Microsoft Azure",
        ],
        "business_address": [
            "410 Terry Ave N, Seattle, WA 98109",
            "2401 Utah Ave, Seattle, WA 98134",
            "Hosur Road, Electronic City, Bangalore 560100",
            "Fort, Mumbai 400001",
            "Courbevoie, 92400",
            "22 Av Montaigne, Paris 75008",
            "100 Target Way, Minneapolis, MN 55403",
            "Anjaneya Nagar, Mumbai 400076",
            "200 Rue de Paris, Lille 59000",
            "One Microsoft Way, Redmond, WA 98052",
        ],
        "city": ["Seattle", "Seattle", "Bangalore", "Mumbai", "Courbevoie", "Paris", "Minneapolis", "Mumbai", "Lille", "Redmond"],
        "state": ["WA", "WA", "Karnataka", "Maharashtra", "IDF", "IDF", "MN", "Maharashtra", "HDF", "WA"],
        "country": ["US", "US", "IND", "IND", "FRA", "FRA", "USA", "IND", "FRA", "US"],
        "postal_code": ["98109", "98134", "560100", "400001", "92400", "75008", "55403", "400076", "59000", "98052"],
    })

    # 4. Train Ground Truth
    train_gt = pl.DataFrame({
        "source1_entity_id": [f"S1-{i:04d}" for i in range(1, 11)],
        "matched_entity_ids": [
            "S2-0001,S3-0001",
            "S2-0002,S3-0002",
            "S2-0003,S3-0003",
            "S2-0004,S3-0004",
            "S2-0005,S3-0005",
            "S2-0006,S3-0006",
            "S2-0007",
            "S2-0008",
            "S2-0009",
            "S2-0010,S3-0010",
        ]
    })

    # Save training TSVs
    train_s1.write_csv(data_dir / "train_source1.tsv", separator="\t")
    train_s2.write_csv(data_dir / "train_source2.tsv", separator="\t")
    train_s3.write_csv(data_dir / "train_source3.tsv", separator="\t")
    train_gt.write_csv(data_dir / "train_ground_truth.tsv", separator="\t")

    # Test sources (simulated with same schema)
    test_s1 = train_s1
    test_s2 = train_s2
    test_s3 = train_s3

    test_s1.write_csv(data_dir / "test_source1.tsv", separator="\t")
    test_s2.write_csv(data_dir / "test_source2.tsv", separator="\t")
    test_s3.write_csv(data_dir / "test_source3.tsv", separator="\t")

    logger.info("Synthetic dataset created in 'data/' directory.")


if __name__ == "__main__":
    generate_synthetic_data(CONFIG.paths.DATA_DIR)
    
    logger.info("Executing Training Pipeline...")
    trainer = EntityResolutionTrainer()
    trainer.train_pipeline()

    logger.info("Executing Inference Pipeline on Test Set...")
    inferer = EntityResolutionInference()
    inferer.predict_test_dataset()

    logger.info("Executing Validation Integrity Check...")
    validate_candidate_pairs(CONFIG.paths.CANDIDATE_PAIRS_PATH)
    validate_matching_results(CONFIG.paths.MATCHING_RESULTS_PATH, CONFIG.paths.TEST_SOURCE1_PATH)

    logger.info("All pipeline stages executed & verified successfully!")
