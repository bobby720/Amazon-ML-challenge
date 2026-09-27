"""Text preprocessing, normalization, and entity token extraction."""

import re
import logging
from typing import Set, Tuple, Optional
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Legal entity suffixes to remove from business names
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b(pvt\s+ltd|private\s+limited|inc|incorporated|llc|corp|corporation|ltd|limited|co|company|gmbh|sa|sarl|plc|llp|lp|assoc|associates)\b",
    flags=re.IGNORECASE
)

# Address abbreviations mapping
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\bno\b": "number",
    r"\bctr\b": "center",
    r"\bstr\b": "street",
}

# Country aliases normalization mapping
COUNTRY_MAPPING = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "in",
    "ind": "in",
    "india": "in",
    "fr": "fr",
    "fra": "fr",
    "france": "fr",
}

# Regex for postal/ZIP codes (supports US 5/9 digit, India 6 digit, France 5 digit, general alphanumerics)
POSTAL_CODE_REGEX = re.compile(r"\b([0-9]{5}(?:-[0-9]{4})?|[0-9]{6}|[a-z0-9]{3,8})\b", flags=re.IGNORECASE)

# Regex for building/street numbers
STREET_NUMBER_REGEX = re.compile(r"\b([0-9]{1,6}[a-z]?)\b", flags=re.IGNORECASE)

# Regex for any numeric sequence
NUMERIC_TOKENS_REGEX = re.compile(r"\b[0-9]+\b")


def clean_text(text: Optional[str]) -> str:
    """Lowercase text and strip extraneous punctuation while preserving alphanumeric characters."""
    if not text or text == "None" or text == "null":
        return ""
    # Convert to lowercase
    s = text.lower()
    # Replace common separators with spaces
    s = re.sub(r"[\t\r\n\-_/\\,;:|#&()+]", " ", s)
    # Strip extraneous non-alphanumeric characters (keep standard ASCII and unicode word characters)
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_business_name(name: Optional[str]) -> str:
    """Clean business name and strip legal entity suffixes."""
    cleaned = clean_text(name)
    if not cleaned:
        return ""
    # Strip legal entity suffixes
    normalized = LEGAL_SUFFIXES_REGEX.sub(" ", cleaned)
    # Collapse extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if normalized else cleaned


def normalize_address(address: Optional[str]) -> str:
    """Clean address and normalize standard street abbreviations."""
    cleaned = clean_text(address)
    if not cleaned:
        return ""
    
    # Replace abbreviations
    res = cleaned
    for pattern, full_word in ADDRESS_ABBREVIATIONS.items():
        res = re.sub(pattern, full_word, res)
    
    res = re.sub(r"\s+", " ", res).strip()
    return res


def normalize_country(country: Optional[str]) -> str:
    """Normalize country strings into standardized ISO-style codes."""
    cleaned = clean_text(country)
    if not cleaned:
        return "unknown"
    return COUNTRY_MAPPING.get(cleaned, cleaned)


def extract_postal_code(address: Optional[str], fallback_postal: Optional[str] = None) -> str:
    """Extract standard postal/PIN code from address or fallback postal column."""
    if fallback_postal and clean_text(fallback_postal):
        fp = clean_text(fallback_postal).replace(" ", "")
        if fp:
            return fp
    
    if not address:
        return ""
    
    # Try finding postal code in address
    matches = POSTAL_CODE_REGEX.findall(address)
    if matches:
        # Prefer longest match that looks like valid numeric code
        numeric_matches = [m for m in matches if any(c.isdigit() for c in m)]
        if numeric_matches:
            return numeric_matches[-1].replace(" ", "").lower()
        return matches[-1].replace(" ", "").lower()
    return ""


def extract_street_number(address: Optional[str]) -> str:
    """Extract building or street number from address string."""
    if not address:
        return ""
    matches = STREET_NUMBER_REGEX.findall(address)
    return matches[0] if matches else ""


def extract_numeric_tokens(text: Optional[str]) -> Set[str]:
    """Extract set of all numeric digit tokens from string."""
    if not text:
        return set()
    return set(NUMERIC_TOKENS_REGEX.findall(text))


def get_name_tokens(name: str) -> Tuple[str, str, str]:
    """
    Extract first token, first 2 tokens, and combined tokens for blocking keys.
    Returns: (token_1, token_1_2, full_compact)
    """
    tokens = name.split()
    t1 = tokens[0] if len(tokens) >= 1 else ""
    t2 = " ".join(tokens[:2]) if len(tokens) >= 2 else t1
    compact = "".join(tokens)
    return t1, t2, compact


def preprocess_polars_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply full preprocessing pipeline to a Polars DataFrame.
    Expected columns: entity_id (or source1_entity_id), business_name, business_address, 
                      and optional country, city, state, postal_code.
    """
    # Ensure all required text columns exist
    cols = df.columns
    id_col = [c for c in cols if "id" in c.lower()][0]
    
    name_col = "business_name" if "business_name" in cols else "name"
    addr_col = "business_address" if "business_address" in cols else "address"
    country_col = "country" if "country" in cols else None
    city_col = "city" if "city" in cols else None
    state_col = "state" if "state" in cols else None
    postal_col = "postal_code" if "postal_code" in cols else "zip" if "zip" in cols else None

    # Cast existing columns to string and fill nulls
    exprs = [
        pl.col(id_col).cast(pl.Utf8).alias("entity_id"),
        pl.col(name_col).fill_null("").cast(pl.Utf8).alias("raw_name"),
        pl.col(addr_col).fill_null("").cast(pl.Utf8).alias("raw_address"),
    ]
    
    if country_col:
        exprs.append(pl.col(country_col).fill_null("").cast(pl.Utf8).alias("raw_country"))
    else:
        exprs.append(pl.lit("").alias("raw_country"))

    if city_col:
        exprs.append(pl.col(city_col).fill_null("").cast(pl.Utf8).alias("raw_city"))
    else:
        exprs.append(pl.lit("").alias("raw_city"))

    if state_col:
        exprs.append(pl.col(state_col).fill_null("").cast(pl.Utf8).alias("raw_state"))
    else:
        exprs.append(pl.lit("").alias("raw_state"))

    if postal_col:
        exprs.append(pl.col(postal_col).fill_null("").cast(pl.Utf8).alias("raw_postal"))
    else:
        exprs.append(pl.lit("").alias("raw_postal"))

    clean_df = df.select(exprs)

    # Process normalized text and blocking tokens
    names = clean_df["raw_name"].to_list()
    addresses = clean_df["raw_address"].to_list()
    countries = clean_df["raw_country"].to_list()
    cities = clean_df["raw_city"].to_list()
    states = clean_df["raw_state"].to_list()
    postals = clean_df["raw_postal"].to_list()

    norm_names = [normalize_business_name(n) for n in names]
    norm_addrs = [normalize_address(a) for a in addresses]
    norm_countries = [normalize_country(c) for c in countries]
    norm_cities = [clean_text(c) for c in cities]
    norm_states = [clean_text(s) for s in states]
    
    extracted_postals = [
        extract_postal_code(a, p) for a, p in zip(addresses, postals)
    ]
    extracted_street_nums = [
        extract_street_number(a) for a in addresses
    ]

    t1_list = []
    t2_list = []
    compact_list = []
    for nm in norm_names:
        t1, t2, comp = get_name_tokens(nm)
        t1_list.append(t1)
        t2_list.append(t2)
        compact_list.append(comp)

    # Add processed columns to DataFrame
    return clean_df.with_columns([
        pl.Series("clean_name", norm_names),
        pl.Series("clean_address", norm_addrs),
        pl.Series("clean_country", norm_countries),
        pl.Series("clean_city", norm_cities),
        pl.Series("clean_state", norm_states),
        pl.Series("clean_postal", extracted_postals),
        pl.Series("street_number", extracted_street_nums),
        pl.Series("name_token_1", t1_list),
        pl.Series("name_token_2", t2_list),
        pl.Series("name_compact", compact_list),
    ])
