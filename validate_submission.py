"""Comprehensive submission validator and integrity checker."""

import sys
import logging
from pathlib import Path
from typing import Set, List, Optional
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Validator")


def validate_candidate_pairs(
    candidate_pairs_path: Path,
) -> bool:
    """Validate format and integrity of output/candidate_pairs.tsv."""
    logger.info(f"Validating Candidate Pairs file: {candidate_pairs_path}")
    if not candidate_pairs_path.exists():
        raise FileNotFoundError(f"Missing file: {candidate_pairs_path}")

    df = pl.read_csv(candidate_pairs_path, separator="\t", has_header=True)
    expected_cols = ["source1_entity_id", "candidate_entity_id"]
    
    if df.columns != expected_cols:
        raise ValueError(f"Invalid columns in candidate_pairs.tsv. Expected {expected_cols}, got {df.columns}")

    # Check for empty IDs
    empty_s1 = df.filter(pl.col("source1_entity_id").is_null() | (pl.col("source1_entity_id") == "")).height
    empty_cand = df.filter(pl.col("candidate_entity_id").is_null() | (pl.col("candidate_entity_id") == "")).height
    
    if empty_s1 > 0 or empty_cand > 0:
        raise ValueError(f"Found empty entity IDs! empty_s1={empty_s1}, empty_cand={empty_cand}")

    # Check for self matches
    self_matches = df.filter(pl.col("source1_entity_id") == pl.col("candidate_entity_id")).height
    if self_matches > 0:
        raise ValueError(f"Found {self_matches} self-referencing pairs in candidate_pairs.tsv")

    # Check for duplicates
    dup_count = df.height - df.unique().height
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate candidate pairs. Consider running deduplication.")

    logger.info(f" Candidate Pairs validation PASSED! Total rows: {df.height:,}")
    return True


def validate_matching_results(
    matching_results_path: Path,
    test_source1_path: Optional[Path] = None,
) -> bool:
    """Validate format and integrity of output/matching_results.tsv against all challenge constraints."""
    logger.info(f"Validating Matching Results file: {matching_results_path}")
    if not matching_results_path.exists():
        raise FileNotFoundError(f"Missing file: {matching_results_path}")

    df = pl.read_csv(
        matching_results_path,
        separator="\t",
        has_header=True,
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    )

    expected_cols = ["source1_entity_id", "matched_entity_ids"]
    if df.columns != expected_cols:
        raise ValueError(f"Invalid columns in matching_results.tsv. Expected {expected_cols}, got {df.columns}")

    s1_ids = df["source1_entity_id"].to_list()
    
    # 1. Unique source1_entity_ids
    if len(s1_ids) != len(set(s1_ids)):
        duplicates = len(s1_ids) - len(set(s1_ids))
        raise ValueError(f"Found {duplicates} duplicate source1_entity_ids in submission!")

    # 2. Check complete coverage of test_source1 if available
    if test_source1_path and test_source1_path.exists():
        df_test_s1 = pl.read_csv(test_source1_path, separator="\t", has_header=True)
        test_s1_id_col = [c for c in df_test_s1.columns if "id" in c.lower()][0]
        test_s1_ids = set(df_test_s1[test_s1_id_col].cast(pl.Utf8).to_list())
        sub_s1_ids = set(s1_ids)

        missing = test_s1_ids - sub_s1_ids
        if missing:
            raise ValueError(f"Submission is missing {len(missing)} IDs from test_source1.tsv! Sample: {list(missing)[:5]}")
        
        extra = sub_s1_ids - test_s1_ids
        if extra:
            raise ValueError(f"Submission contains {len(extra)} extra IDs not present in test_source1.tsv! Sample: {list(extra)[:5]}")

    # 3. Check format of matched_entity_ids
    matched_strs = df["matched_entity_ids"].fill_null("").to_list()
    total_matched_links = 0

    for s1_id, match_str in zip(s1_ids, matched_strs):
        if not match_str or match_str == "None":
            continue
        
        cand_list = [c.strip() for c in match_str.split(",") if c.strip()]
        
        # Check no duplicate candidates within same source1
        if len(cand_list) != len(set(cand_list)):
            raise ValueError(f"Found duplicate candidates inside row for {s1_id}: {match_str}")

        # Check no self-referential or S1- IDs in matched candidates
        for cand in cand_list:
            if cand == s1_id or cand.startswith("S1-"):
                raise ValueError(f"Found illegal matched ID '{cand}' for source1 ID '{s1_id}'!")

        total_matched_links += len(cand_list)

    logger.info(
        f" Matching Results validation PASSED!\n"
        f"  - Total S1 Entities: {len(s1_ids):,}\n"
        f"  - S1 Entities with Matches: {sum(1 for m in matched_strs if m):,}\n"
        f"  - Total Entity Links: {total_matched_links:,}"
    )
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    cand_path = base_dir / "output" / "candidate_pairs.tsv"
    match_path = base_dir / "output" / "matching_results.tsv"
    test_s1 = base_dir / "data" / "test_source1.tsv"

    success = True
    try:
        if cand_path.exists():
            validate_candidate_pairs(cand_path)
        else:
            logger.warning(f"Candidate pairs file not found at {cand_path} (will be created during execution).")

        if match_path.exists():
            validate_matching_results(match_path, test_s1 if test_s1.exists() else None)
        else:
            logger.warning(f"Matching results file not found at {match_path} (will be created during execution).")

    except Exception as e:
        logger.error(f"❌ Validation Failed: {e}", exc_info=True)
        sys.exit(1)
