"""Feature engineering module computing RapidFuzz string metrics and digit overlap."""

import re
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .preprocess import extract_numeric_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "addr_fuzz_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "city_fuzz_ratio",
    "state_fuzz_ratio",
    "name_char_len_diff",
    "name_char_len_ratio",
    "name_word_cnt_diff",
    "name_word_cnt_ratio",
    "addr_char_len_diff",
    "addr_char_len_ratio",
    "addr_word_cnt_diff",
    "addr_word_cnt_ratio",
    "digit_overlap_ratio_s1_in_cand",
    "digit_jaccard_similarity",
    "is_exact_name_match",
    "is_exact_postal_match",
    "is_exact_street_num_match",
    "is_exact_country_match",
    "is_exact_city_match",
    "total_blocking_score",
]


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_city: str,
    s1_state: str,
    s1_country: str,
    s1_postal: str,
    s1_street: str,
    cand_name: str,
    cand_addr: str,
    cand_city: str,
    cand_state: str,
    cand_country: str,
    cand_postal: str,
    cand_street: str,
    blocking_score: float = 0.0,
) -> List[float]:
    """Compute rich similarity and overlap metrics for a single entity pair."""
    # 1. RapidFuzz metrics on business name
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    name_sort_ratio = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_set_ratio = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_name, cand_name) / 100.0

    # 2. RapidFuzz metrics on address
    addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
    addr_set_ratio = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    addr_partial_ratio = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0

    # 3. Location metrics
    city_ratio = fuzz.ratio(s1_city, cand_city) / 100.0 if (s1_city and cand_city) else (1.0 if s1_city == cand_city else 0.0)
    state_ratio = fuzz.ratio(s1_state, cand_state) / 100.0 if (s1_state and cand_state) else (1.0 if s1_state == cand_state else 0.0)

    # 4. Length and word count differences & ratios
    len_s1_name = len(s1_name)
    len_cand_name = len(cand_name)
    name_len_diff = abs(len_s1_name - len_cand_name)
    name_len_ratio = (min(len_s1_name, len_cand_name) + 1.0) / (max(len_s1_name, len_cand_name) + 1.0)

    words_s1_name = len(s1_name.split())
    words_cand_name = len(cand_name.split())
    name_word_diff = abs(words_s1_name - words_cand_name)
    name_word_ratio = (min(words_s1_name, words_cand_name) + 1.0) / (max(words_s1_name, words_cand_name) + 1.0)

    len_s1_addr = len(s1_addr)
    len_cand_addr = len(cand_addr)
    addr_len_diff = abs(len_s1_addr - len_cand_addr)
    addr_len_ratio = (min(len_s1_addr, len_cand_addr) + 1.0) / (max(len_s1_addr, len_cand_addr) + 1.0)

    words_s1_addr = len(s1_addr.split())
    words_cand_addr = len(cand_addr.split())
    addr_word_diff = abs(words_s1_addr - words_cand_addr)
    addr_word_ratio = (min(words_s1_addr, words_cand_addr) + 1.0) / (max(words_s1_addr, words_cand_addr) + 1.0)

    # 5. Digit overlap and Jaccard similarity
    s1_digits = extract_numeric_tokens(s1_name + " " + s1_addr + " " + s1_postal)
    cand_digits = extract_numeric_tokens(cand_name + " " + cand_addr + " " + cand_postal)

    if s1_digits:
        overlap_cnt = len(s1_digits.intersection(cand_digits))
        digit_overlap_s1 = overlap_cnt / len(s1_digits)
        digit_jaccard = overlap_cnt / len(s1_digits.union(cand_digits))
    else:
        digit_overlap_s1 = 1.0 if not cand_digits else 0.5
        digit_jaccard = 1.0 if not cand_digits else 0.5

    # 6. Binary indicators
    is_exact_name = 1.0 if s1_name and s1_name == cand_name else 0.0
    is_exact_postal = 1.0 if s1_postal and s1_postal == cand_postal else 0.0
    is_exact_street = 1.0 if s1_street and s1_street == cand_street else 0.0
    is_exact_country = 1.0 if s1_country and s1_country == cand_country else 0.0
    is_exact_city = 1.0 if s1_city and s1_city == cand_city else 0.0

    return [
        name_ratio,
        name_sort_ratio,
        name_set_ratio,
        name_partial_ratio,
        addr_ratio,
        addr_set_ratio,
        addr_partial_ratio,
        city_ratio,
        state_ratio,
        float(name_len_diff),
        name_len_ratio,
        float(name_word_diff),
        name_word_ratio,
        float(addr_len_diff),
        addr_len_ratio,
        float(addr_word_diff),
        addr_word_ratio,
        digit_overlap_s1,
        digit_jaccard,
        is_exact_name,
        is_exact_postal,
        is_exact_street,
        is_exact_country,
        is_exact_city,
        float(blocking_score),
    ]


class FeatureEngine:
    """Batch feature generator for candidate entity pairs."""

    @staticmethod
    def build_feature_matrix(
        candidate_pairs_df: pl.DataFrame,
        df_s1_processed: pl.DataFrame,
        df_cand_processed: pl.DataFrame,
        batch_size: int = 50_000,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build numeric feature matrix from candidate pairs and processed entity tables.
        Returns (feature_matrix, feature_column_names).
        """
        logger.info(f"Extracting features for {candidate_pairs_df.height:,} candidate pairs...")

        # Join pair records with entity metadata
        s1_cols = [
            "entity_id", "clean_name", "clean_address", "clean_city",
            "clean_state", "clean_country", "clean_postal", "street_number"
        ]
        cand_cols = [
            "entity_id", "clean_name", "clean_address", "clean_city",
            "clean_state", "clean_country", "clean_postal", "street_number"
        ]

        # Left join for S1 attributes
        joined = candidate_pairs_df.join(
            df_s1_processed.select(s1_cols),
            left_on="source1_entity_id",
            right_on="entity_id",
            how="inner",
        )

        # Rename joined S1 cols
        joined = joined.rename({
            "clean_name": "s1_name",
            "clean_address": "s1_address",
            "clean_city": "s1_city",
            "clean_state": "s1_state",
            "clean_country": "s1_country",
            "clean_postal": "s1_postal",
            "street_number": "s1_street",
        })

        # Join for Candidate attributes
        joined = joined.join(
            df_cand_processed.select(cand_cols),
            left_on="candidate_entity_id",
            right_on="entity_id",
            how="inner",
        )

        joined = joined.rename({
            "clean_name": "cand_name",
            "clean_address": "cand_address",
            "clean_city": "cand_city",
            "clean_state": "cand_state",
            "clean_country": "cand_country",
            "clean_postal": "cand_postal",
            "street_number": "cand_street",
        })

        # Ensure blocking score column exists
        if "total_blocking_score" not in joined.columns:
            joined = joined.with_columns(pl.lit(1.0).alias("total_blocking_score"))

        num_rows = joined.height
        if num_rows == 0:
            return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), FEATURE_COLUMNS

        # Extract columns to Python lists for fast iteration
        s1_names = joined["s1_name"].to_list()
        s1_addrs = joined["s1_address"].to_list()
        s1_cities = joined["s1_city"].to_list()
        s1_states = joined["s1_state"].to_list()
        s1_countries = joined["s1_country"].to_list()
        s1_postals = joined["s1_postal"].to_list()
        s1_streets = joined["s1_street"].to_list()

        cand_names = joined["cand_name"].to_list()
        cand_addrs = joined["cand_address"].to_list()
        cand_cities = joined["cand_city"].to_list()
        cand_states = joined["cand_state"].to_list()
        cand_countries = joined["cand_country"].to_list()
        cand_postals = joined["cand_postal"].to_list()
        cand_streets = joined["cand_street"].to_list()
        b_scores = joined["total_blocking_score"].to_list()

        features_list = []
        for i in range(num_rows):
            f_vec = compute_pair_features(
                s1_names[i], s1_addrs[i], s1_cities[i], s1_states[i], s1_countries[i], s1_postals[i], s1_streets[i],
                cand_names[i], cand_addrs[i], cand_cities[i], cand_states[i], cand_countries[i], cand_postals[i], cand_streets[i],
                b_scores[i]
            )
            features_list.append(f_vec)

        feature_matrix = np.array(features_list, dtype=np.float32)
        logger.info(f"Feature matrix generated with shape: {feature_matrix.shape}")
        return feature_matrix, FEATURE_COLUMNS
