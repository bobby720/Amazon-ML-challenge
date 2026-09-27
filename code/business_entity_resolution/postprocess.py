"""Formatting predictions into submission TSVs and integrity validation."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Set, Union, Optional
import polars as pl

from .config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_submission_file(
    all_s1_ids: List[str],
    predicted_pairs_df: Optional[pl.DataFrame] = None,
    output_path: Union[str, Path] = CONFIG.paths.MATCHING_RESULTS_PATH,
) -> None:
    """
    Format predictions into matching_results.tsv satisfying all submission rules:
    - Exactly two columns: source1_entity_id\tmatched_entity_ids
    - Every source1_entity_id from test set appears exactly once.
    - Comma-separated list of matched candidate entity IDs (sorted by confidence or ID).
    - Blank string if no match found.
    - No duplicate IDs, no self-referential S1- IDs.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Dictionary mapping: s1_id -> list of matched candidate IDs
    matches_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    if predicted_pairs_df is not None and predicted_pairs_df.height > 0:
        # Sort by probability if available, otherwise by candidate_id
        sort_cols = []
        desc_flags = []
        if "probability" in predicted_pairs_df.columns:
            sort_cols.append("probability")
            desc_flags.append(True)
        sort_cols.append("candidate_entity_id")
        desc_flags.append(False)

        sorted_df = predicted_pairs_df.sort(by=sort_cols, descending=desc_flags)
        
        for row in sorted_df.iter_rows(named=True):
            s1_id = str(row["source1_entity_id"]).strip()
            cand_id = str(row["candidate_entity_id"]).strip()
            
            # Guard against self-reference or invalid values
            if not cand_id or cand_id == s1_id or cand_id.startswith("S1-"):
                continue

            if s1_id in matches_map:
                if cand_id not in matches_map[s1_id]:
                    matches_map[s1_id].append(cand_id)
            else:
                matches_map[s1_id] = [cand_id]

    logger.info(f"Writing {len(matches_map):,} records to submission TSV: {output_file}")
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in all_s1_ids:
            cand_list = matches_map.get(s1_id, [])
            matched_str = ",".join(cand_list)
            writer.writerow([s1_id, matched_str])

    logger.info("Submission file generated successfully.")
