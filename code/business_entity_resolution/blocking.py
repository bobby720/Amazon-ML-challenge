"""Multi-pass deterministic inverted-index blocking engine."""

import gc
import logging
from pathlib import Path
from typing import List, Optional, Union
import polars as pl

from .config import CONFIG, BlockingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiPassBlocker:
    """
    High-throughput multi-pass candidate blocker using Polars joins.
    Restricts search space by enforcing strict country matching and multi-pass inverted keys.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or CONFIG.blocking

    def _generate_pass_pairs(
        self,
        df_s1: pl.DataFrame,
        df_cand: pl.DataFrame,
        s1_keys: List[str],
        cand_keys: List[str],
        pass_weight: int,
    ) -> pl.DataFrame:
        """Execute a single blocking join pass on specified key expressions."""
        # Filter out empty or null key rows
        filter_s1 = pl.all_horizontal([pl.col(k).str.len_chars() > 0 for k in s1_keys])
        filter_cand = pl.all_horizontal([pl.col(k).str.len_chars() > 0 for k in cand_keys])

        sub_s1 = df_s1.filter(filter_s1).select(["entity_id"] + s1_keys)
        sub_cand = df_cand.filter(filter_cand).select(["entity_id"] + cand_keys)

        joined = sub_s1.join(
            sub_cand,
            left_on=s1_keys,
            right_on=cand_keys,
            how="inner"
        ).select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_right").alias("candidate_entity_id"),
            pl.lit(pass_weight).alias("blocking_score"),
        ])

        return joined

    def block_candidates(
        self,
        df_s1_processed: pl.DataFrame,
        df_candidates_processed: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Generate candidate pairs using multi-pass blocking keys.
        
        Blocking Passes:
        1. Exact Match on (clean_country, clean_name)
        2. (clean_country, name_token_2, clean_postal)
        3. (clean_country, name_token_1, clean_city)
        4. (clean_country, name_token_1, clean_state)
        """
        logger.info(f"Starting blocking for {df_s1_processed.height:,} S1 entities against {df_candidates_processed.height:,} candidate records.")
        
        pairs_list: List[pl.DataFrame] = []

        # Pass 1: Exact Name within Country (Weight 4)
        logger.info("Executing Pass 1: Exact Clean Name + Country...")
        p1 = self._generate_pass_pairs(
            df_s1_processed,
            df_candidates_processed,
            s1_keys=["clean_country", "clean_name"],
            cand_keys=["clean_country", "clean_name"],
            pass_weight=4,
        )
        if p1.height > 0:
            pairs_list.append(p1)
        logger.info(f"Pass 1 generated {p1.height:,} pairs.")

        # Pass 2: First 2 Name Tokens + Postal Code within Country (Weight 3)
        logger.info("Executing Pass 2: 2 Name Tokens + Postal + Country...")
        p2 = self._generate_pass_pairs(
            df_s1_processed,
            df_candidates_processed,
            s1_keys=["clean_country", "name_token_2", "clean_postal"],
            cand_keys=["clean_country", "name_token_2", "clean_postal"],
            pass_weight=3,
        )
        if p2.height > 0:
            pairs_list.append(p2)
        logger.info(f"Pass 2 generated {p2.height:,} pairs.")

        # Pass 3: First Name Token + City within Country (Weight 2)
        logger.info("Executing Pass 3: 1 Name Token + City + Country...")
        p3 = self._generate_pass_pairs(
            df_s1_processed,
            df_candidates_processed,
            s1_keys=["clean_country", "name_token_1", "clean_city"],
            cand_keys=["clean_country", "name_token_1", "clean_city"],
            pass_weight=2,
        )
        if p3.height > 0:
            pairs_list.append(p3)
        logger.info(f"Pass 3 generated {p3.height:,} pairs.")

        # Pass 4: First Name Token + State within Country (Weight 1)
        logger.info("Executing Pass 4: 1 Name Token + State + Country...")
        p4 = self._generate_pass_pairs(
            df_s1_processed,
            df_candidates_processed,
            s1_keys=["clean_country", "name_token_1", "clean_state"],
            cand_keys=["clean_country", "name_token_1", "clean_state"],
            pass_weight=1,
        )
        if p4.height > 0:
            pairs_list.append(p4)
        logger.info(f"Pass 4 generated {p4.height:,} pairs.")

        if not pairs_list:
            logger.warning("No candidate pairs found across all blocking passes!")
            return pl.DataFrame(schema={
                "source1_entity_id": pl.Utf8,
                "candidate_entity_id": pl.Utf8,
                "blocking_score": pl.Int32,
            })

        # Concatenate and aggregate multi-pass scores
        logger.info("Aggregating and deduplicating candidate pairs...")
        all_pairs = pl.concat(pairs_list)
        
        # Deduplicate and sum scores across passes for stronger candidates
        agg_pairs = all_pairs.group_by(["source1_entity_id", "candidate_entity_id"]).agg([
            pl.col("blocking_score").sum().alias("total_blocking_score")
        ])

        # Filter self-matches if any
        agg_pairs = agg_pairs.filter(
            pl.col("source1_entity_id") != pl.col("candidate_entity_id")
        )

        # Rank and limit top N candidates per source1_entity_id
        max_cands = self.config.MAX_CANDIDATES_PER_S1
        logger.info(f"Limiting to top {max_cands} candidates per S1 entity...")
        
        ranked = agg_pairs.with_columns(
            pl.col("total_blocking_score")
            .rank(method="ordinal", descending=True)
            .over("source1_entity_id")
            .alias("rank")
        ).filter(pl.col("rank") <= max_cands)

        final_pairs = ranked.select(["source1_entity_id", "candidate_entity_id", "total_blocking_score"])
        logger.info(f"Total candidate pairs generated after capping: {final_pairs.height:,}")

        # Clean up memory
        del pairs_list, all_pairs, agg_pairs, ranked
        gc.collect()

        return final_pairs

    @staticmethod
    def export_candidate_pairs(
        pairs_df: pl.DataFrame,
        output_path: Union[str, Path] = CONFIG.paths.CANDIDATE_PAIRS_PATH
    ) -> None:
        """Export candidate pairs to TSV format with standard schema."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        export_df = pairs_df.select([
            pl.col("source1_entity_id"),
            pl.col("candidate_entity_id")
        ])
        
        export_df.write_csv(out, separator="\t", include_header=True)
        logger.info(f"Candidate pairs successfully exported to: {out} ({export_df.height:,} rows)")
