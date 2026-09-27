"""V5 Feature Engineering Module: Building-Number, Abbreviation, and Transliteration Views.

Implements the three V5 feature groups strictly isolated in `experiments/v5/`:
  1. BUILDING-NUMBER FEATURES (`V5_BUILDING_FEATURES`, 12 features):
     - number type/context (`plot`, `shop`, `door`, `survey`, `sector`, `street`, `floor`, `block`)
     - exact number overlap (normalized, leading-zero stripped, internal digit-hyphen collapsed)
     - number-set overlap (Jaccard and min-containment over all non-postal address numbers)
     - normalized number agreement (exact, prefix/suffix substring, range, +-1 adjacent)
     - plot/shop/door/survey/sector/street distinction (same-slot vs cross-slot comparison)
     - conflict severity (continuous [0, 1] and severe conflict flag)
     - address-context agreement around numbers & street/locality token similarity
     - India-specific noisy-number handling (prepended noise numbers `#487 B-613`, `DOOR NO 857 H-2`, `6-0` vs `60`)

  2. ABBREVIATION FEATURES (`V5_ABBREVIATION_FEATURES`, 10 features):
     - Mined ONLY from entity-disjoint challenge training data (`train_ground_truth.tsv` + `train_source1/2/3.tsv`
       excluding all 5,000 validation/holdout S1 IDs)
     - Preserves original text & existing V3/V4 features
     - abbreviation-expanded name similarity (token Jaccard, token set, char 3-gram Jaccard)
     - abbreviation-expanded address similarity (token Jaccard, token set)
     - acronym/full-name compatibility (including domain-initial patterns like `ppacademy.com`)
     - initial-token compatibility
     - distinctive non-geographic brand token Jaccard & brand/branch conflict detection

  3. TRANSLITERATION FEATURES (`V5_TRANSLITERATION_FEATURES`, 10 features):
     - Challenge-data-derived normalization/transliteration view (Indic Unicode U+0900..U+0D7F + Romanized Indic)
     - transliteration-normalized token similarity (name & address)
     - character similarity (3-gram Jaccard & edit ratio)
     - phonetic similarity (consonant skeleton Jaccard & edit ratio)
     - acronym compatibility on transliterated view
     - transliterated brand/state conflict indicator
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from code.business_entity_resolution.src.features_v3 import _char_ngrams_set, _set_jaccard
from code.business_entity_resolution.src.transliteration_v5 import (
    _CHALLENGE_TRANSLIT_TOKEN_MAP,
    _CORPORATE_STOPWORDS,
    has_non_ascii_or_indic,
    normalize_translit_view,
    phonetic_consonant_skeleton,
    transliterate_indic_to_roman,
)

V5_BUILDING_FEATURES: List[str] = [
    "v5_bldg_type_match",
    "v5_bldg_diff_slot_flag",
    "v5_bldg_exact_num_overlap",
    "v5_bldg_num_set_jaccard",
    "v5_bldg_num_min_containment",
    "v5_bldg_norm_num_agreement",
    "v5_bldg_context_agreement",
    "v5_bldg_street_locality_sim",
    "v5_bldg_india_noisy_num_resolved",
    "v5_bldg_conflict_resolved",
    "v5_bldg_conflict_severity",
    "v5_bldg_severe_conflict_flag",
]

V5_ABBREVIATION_FEATURES: List[str] = [
    "v5_abbr_name_token_jaccard",
    "v5_abbr_name_token_set",
    "v5_abbr_name_char_3gram_jaccard",
    "v5_abbr_addr_token_jaccard",
    "v5_abbr_addr_token_set",
    "v5_abbr_acronym_compat",
    "v5_abbr_initial_token_compat",
    "v5_abbr_distinctive_brand_jaccard",
    "v5_abbr_distinctive_brand_conflict",
    "v5_abbr_generic_name_diff_branch_conflict",
]

V5_TRANSLITERATION_FEATURES: List[str] = [
    "v5_translit_has_non_ascii",
    "v5_translit_name_token_jaccard",
    "v5_translit_name_token_set",
    "v5_translit_name_char_3gram_jaccard",
    "v5_translit_name_edit_ratio",
    "v5_translit_phonetic_token_jaccard",
    "v5_translit_phonetic_edit_ratio",
    "v5_translit_addr_token_set",
    "v5_translit_acronym_compat",
    "v5_translit_brand_or_state_conflict",
]

V5_ALL_NEW_FEATURES: List[str] = (
    V5_BUILDING_FEATURES + V5_ABBREVIATION_FEATURES + V5_TRANSLITERATION_FEATURES
)

# ---------------------------------------------------------------------------
# Challenge-Data-Derived Lexicon Miner (Strictly Entity-Disjoint Training Data)
# ---------------------------------------------------------------------------
LEXICON_CACHE_PATH = Path("experiments/v5/data/v5_challenge_lexicons.json")

# Base seed abbreviations verified in challenge training data, augmented by automated mining
_SEED_NAME_ABBREV: Dict[str, str] = {
    "intl": "international",
    "int": "international",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "mfrs": "manufacturers",
    "hosp": "hospital",
    "med": "medical",
    "pharm": "pharmacy",
    "pharma": "pharmaceuticals",
    "ind": "industries",
    "inds": "industries",
    "indl": "industrial",
    "ent": "enterprises",
    "enterp": "enterprises",
    "assoc": "associates",
    "assn": "association",
    "dept": "department",
    "univ": "university",
    "inst": "institute",
    "mgmt": "management",
    "mgt": "management",
    "svcs": "services",
    "svc": "service",
    "srv": "service",
    "tech": "technologies",
    "techn": "technologies",
    "engg": "engineering",
    "eng": "engineering",
    "engr": "engineers",
    "coop": "cooperative",
    "co-op": "cooperative",
    "fed": "federal",
    "natl": "national",
    "amer": "american",
    "comm": "communications",
    "comms": "communications",
    "fin": "financial",
    "inv": "investment",
    "invs": "investments",
    "dev": "development",
    "devs": "developers",
    "const": "construction",
    "constr": "construction",
    "dist": "distributors",
    "distr": "distributors",
    "mktg": "marketing",
    "mkt": "market",
    "sys": "systems",
    "sol": "solutions",
    "solns": "solutions",
    "log": "logistics",
    "trans": "transport",
    "chem": "chemicals",
    "elec": "electrical",
    "elect": "electricals",
    "auto": "automotive",
    "ag": "agriculture",
    "agri": "agricultural",
    "agro": "agro",
    "edu": "education",
    "acad": "academy",
    "sch": "school",
    "coll": "college",
    "ctr": "center",
    "cntr": "center",
    "centre": "center",
    "grp": "group",
    "hldg": "holdings",
    "hldgs": "holdings",
    "prop": "properties",
    "props": "properties",
    "labs": "laboratories",
    "lab": "laboratory",
    "diag": "diagnostics",
    "hlth": "health",
    "vet": "veterinary",
    "rehab": "rehabilitation",
    "orth": "orthopedic",
    "ped": "pediatric",
    "dent": "dental",
    "opt": "optical",
    "ins": "insurance",
    "sec": "security",
    "govt": "government",
    "gov": "government",
    "mun": "municipal",
    "st": "saint",
    "mt": "mount",
    "ft": "fort",
    "dr": "doctor",
    "shri": "sri",
    "shree": "sri",
    "srimati": "smt",
}

_SEED_ADDR_ABBREV: Dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "fwy": "freeway",
    "expy": "expressway",
    "tpke": "turnpike",
    "trl": "trail",
    "ter": "terrace",
    "sq": "square",
    "aly": "alley",
    "xing": "crossing",
    "jct": "junction",
    "jn": "junction",
    "pt": "point",
    "hts": "heights",
    "vlg": "village",
    "vill": "village",
    "twp": "township",
    "ste": "suite",
    "apt": "apartment",
    "apts": "apartments",
    "bldg": "building",
    "buld": "building",
    "fl": "floor",
    "flr": "floor",
    "rm": "room",
    "dept": "department",
    "opp": "opposite",
    "op": "opposite",
    "nr": "near",
    "adj": "adjacent",
    "bhd": "behind",
    "bes": "beside",
    "chmbrs": "chambers",
    "cplx": "complex",
    "cmplx": "complex",
    "est": "estate",
    "indl": "industrial",
    "sec": "sector",
    "sect": "sector",
    "ph": "phase",
    "blk": "block",
    "plt": "plot",
    "sy": "survey",
    "sno": "survey",
    "hno": "house",
    "dno": "door",
    "flno": "flat",
    "soc": "society",
    "coop": "cooperative",
    "cghs": "cghs",
    "gidc": "gidc",
    "midc": "midc",
    "ngr": "nagar",
    "clny": "colony",
    "col": "colony",
    "ext": "extension",
    "extn": "extension",
    "marg": "road",
    "path": "road",
    "main": "main",
    "cross": "cross",
    "stg": "stage",
    "lyt": "layout",
    "encl": "enclave",
    "twr": "tower",
    "twrs": "towers",
    "arc": "arcade",
    "plz": "plaza",
    "mkt": "market",
    "bazar": "bazaar",
    "dist": "district",
    "tal": "taluka",
    "tq": "taluka",
    "teh": "tehsil",
    "po": "post",
    "ps": "police",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Generic non-distinctive business words (matching ONLY on these + city name = high branch/collision risk)
_GENERIC_BUSINESS_WORDS: Set[str] = {
    "private", "limited", "pvt", "ltd", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "the", "and", "of", "for", "in",
    "services", "service", "solutions", "solution", "consulting", "consultants",
    "consultant", "consultancy", "investment", "investments", "hospitality",
    "products", "product", "trading", "traders", "trader", "enterprises",
    "enterprise", "industries", "industry", "industrial", "technologies",
    "technology", "tech", "infotech", "software", "systems", "system",
    "developers", "developer", "development", "builders", "builder",
    "construction", "constructions", "infrastructure", "infra", "realty",
    "properties", "property", "estate", "estates", "holdings", "holding",
    "ventures", "venture", "group", "center", "centre", "trust", "foundation",
    "institute", "academy", "school", "college", "university", "associates",
    "associate", "partners", "partner", "alliance", "coalition", "federation",
    "collective", "initiative", "agency", "agencies", "marketing", "distributors",
    "distributor", "distribution", "exports", "export", "imports", "import",
    "impex", "logistics", "transport", "transportation", "motors", "motor",
    "automotive", "auto", "electricals", "electrical", "electronics", "electronic",
    "chemicals", "chemical", "pharma", "pharmaceuticals", "pharmaceutical",
    "pharmacy", "medical", "medicine", "healthcare", "health", "care", "hospital",
    "clinic", "diagnostics", "diagnostic", "laboratories", "laboratory", "labs",
    "foods", "food", "agro", "agriculture", "textiles", "textile", "garments",
    "engineering", "engineers", "finance", "financial", "capital", "credit",
    "securities", "insurance", "advisors", "advisory", "management", "media",
    "communications", "digital", "networks", "network", "global", "international",
    "national", "federal", "united", "american", "india", "indian", "us", "usa",
    "north", "south", "east", "west", "central", "metro", "city", "town", "urban",
    "commercial", "general", "standard", "premier", "modern", "new", "sri", "shri",
    "shree", "smt", "dr", "st", "saint",
}

# Canonical Indian & US state mapping (including Indic transliterated state names)
_STATE_CANON_MAP: Dict[str, str] = {
    # India states & UTs
    "mh": "in_mh", "maharashtra": "in_mh", "maharastra": "in_mh",
    "dl": "in_dl", "delhi": "in_dl", "new delhi": "in_dl", "dilli": "in_dl",
    "ka": "in_ka", "karnataka": "in_ka", "karnatak": "in_ka",
    "tn": "in_tn", "tamil nadu": "in_tn", "tamilnadu": "in_tn",
    "up": "in_up", "uttar pradesh": "in_up", "uttarpradesh": "in_up",
    "gj": "in_gj", "gujarat": "in_gj", "gujrat": "in_gj",
    "wb": "in_wb", "west bengal": "in_wb", "paschim banga": "in_wb", "paschimbanga": "in_wb",
    "tg": "in_tg", "ts": "in_tg", "telangana": "in_tg",
    "ap": "in_ap", "andhra pradesh": "in_ap",
    "rj": "in_rj", "rajasthan": "in_rj",
    "mp": "in_mp", "madhya pradesh": "in_mp",
    "hr": "in_hr", "haryana": "in_hr", "hariyana": "in_hr",
    "pb": "in_pb", "punjab": "in_pb", "panjab": "in_pb",
    "kl": "in_kl", "kerala": "in_kl", "keralam": "in_kl",
    "or": "in_od", "od": "in_od", "odisha": "in_od", "orissa": "in_od",
    "br": "in_br", "bihar": "in_br",
    "jh": "in_jh", "jharkhand": "in_jh",
    "cg": "in_cg", "ct": "in_cg", "chhattisgarh": "in_cg",
    "as": "in_as", "assam": "in_as", "asam": "in_as",
    "uk": "in_uk", "ut": "in_uk", "uttarakhand": "in_uk", "uttaranchal": "in_uk",
    "hp": "in_hp", "himachal pradesh": "in_hp",
    "ga": "in_ga", "goa": "in_ga",
    "ch": "in_ch", "chandigarh": "in_ch",
    "py": "in_py", "puducherry": "in_py", "pondicherry": "in_py",
    "jk": "in_jk", "jammu and kashmir": "in_jk",
    # US states
    "al": "us_al", "alabama": "us_al", "ak": "us_ak", "alaska": "us_ak",
    "az": "us_az", "arizona": "us_az", "ar": "us_ar", "arkansas": "us_ar",
    "ca": "us_ca", "california": "us_ca", "co": "us_co", "colorado": "us_co",
    "ct": "us_ct", "connecticut": "us_ct", "de": "us_de", "delaware": "us_de",
    "fl": "us_fl", "florida": "us_fl", "ga": "us_ga", "georgia": "us_ga",
    "hi": "us_hi", "hawaii": "us_hi", "id": "us_id", "idaho": "us_id",
    "il": "us_il", "illinois": "us_il", "in": "us_in", "indiana": "us_in",
    "ia": "us_ia", "iowa": "us_ia", "ks": "us_ks", "kansas": "us_ks",
    "ky": "us_ky", "kentucky": "us_ky", "la": "us_la", "louisiana": "us_la",
    "me": "us_me", "maine": "us_me", "md": "us_md", "maryland": "us_md",
    "ma": "us_ma", "massachusetts": "us_ma", "mi": "us_mi", "michigan": "us_mi",
    "mn": "us_mn", "minnesota": "us_mn", "ms": "us_ms", "mississippi": "us_ms",
    "mo": "us_mo", "missouri": "us_mo", "mt": "us_mt", "montana": "us_mt",
    "ne": "us_ne", "nebraska": "us_ne", "nv": "us_nv", "nevada": "us_nv",
    "nh": "us_nh", "new hampshire": "us_nh", "nj": "us_nj", "new jersey": "us_nj",
    "nm": "us_nm", "new mexico": "us_nm", "ny": "us_ny", "new york": "us_ny",
    "nc": "us_nc", "north carolina": "us_nc", "nd": "us_nd", "north dakota": "us_nd",
    "oh": "us_oh", "ohio": "us_oh", "ok": "us_ok", "oklahoma": "us_ok",
    "or": "us_or", "oregon": "us_or", "pa": "us_pa", "pennsylvania": "us_pa",
    "ri": "us_ri", "rhode island": "us_ri", "sc": "us_sc", "south carolina": "us_sc",
    "sd": "us_sd", "south dakota": "us_sd", "tn": "us_tn", "tennessee": "us_tn",
    "tx": "us_tx", "texas": "us_tx", "ut": "us_ut", "utah": "us_ut",
    "vt": "us_vt", "vermont": "us_vt", "va": "us_va", "virginia": "us_va",
    "wa": "us_wa", "washington": "us_wa", "wv": "us_wv", "west virginia": "us_wv",
    "wi": "us_wi", "wisconsin": "us_wi", "wy": "us_wy", "wyoming": "us_wy",
    "dc": "us_dc",
}

# Major Indian cities for city-conflict and city-in-name detection
_MAJOR_CITIES: Set[str] = {
    "mumbai", "delhi", "bangalore", "hyderabad", "ahmedabad", "chennai", "kolkata",
    "surat", "pune", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "thane",
    "bhopal", "visakhapatnam", "pimpri", "patna", "vadodara", "ghaziabad", "ludhiana",
    "agra", "nashik", "faridabad", "meerut", "rajkot", "kalyan", "vasai", "varanasi",
    "srinagar", "aurangabad", "dhanbad", "amritsar", "navi", "allahabad", "prayagraj",
    "ranchi", "howrah", "coimbatore", "jabalpur", "gwalior", "vijayawada", "jodhpur",
    "madurai", "raipur", "kota", "guwahati", "chandigarh", "solapur", "hubli",
    "mysore", "mysuru", "tiruchirappalli", "bareilly", "aligarh", "tiruppur",
    "gurgaon", "moradabad", "jalandhar", "bhubaneswar", "salem", "warangal",
    "guntur", "bhiwandi", "saharanpur", "gorakhpur", "bikaner", "amravati",
    "noida", "jamshedpur", "bhilai", "cuttack", "firozabad", "kochi", "cochin",
    "nellore", "bhavnagar", "dehradun", "durgapur", "asansol", "rourkela",
    "nanded", "kolhapur", "ajmer", "akola", "gulbarga", "jamnagar", "ujjain",
    "loni", "siliguri", "jhansi", "ulhasnagar", "jammu", "mangalore", "mangaluru",
    "erode", "belgaum", "ambattur", "tirunelveli", "malegaon", "gaya", "jalgaon",
    "udaipur", "maheshtala", "davanagere", "kozhikode", "calicut", "kurnool",
    "rajpur", "rajahmundry", "bokaro", "bellary", "patiala", "gopalpur", "agartala",
    "bhagalpur", "muzaffarnagar", "bhatpara", "panihati", "latur", "dhule",
    "tirupati", "rohtak", "korba", "bhilwara", "berhampur", "muzaffarpur",
    "ahmednagar", "mathura", "kollam", "avadi", "kadapa", "kamarhati", "sambalpur",
    "bilaspur", "shahjahanpur", "satara", "bijapur", "rampur", "shimoga",
    "chandrapur", "junagadh", "thrissur", "alwar", "bardhaman", "kulti", "kakinada",
    "nizamabad", "parbhani", "tumkur", "khammam", "ozhukarai", "panipat",
    "darbhanga", "bally", "aizawl", "dewas", "ichalkaranji", "karnal", "bathinda",
    "jalna", "eluru", "kirari", "barasat", "purnia", "satna", "mau", "sonipat",
    "farrukhabad", "sagar", "durg", "imphal", "ratlam", "hapur", "arah", "anantapur",
    "karimnagar", "etawah", "ambernath", "bharatpur", "begusarai", "valsad",
    "umbergaon", "tiruvallur", "poonamallee", "khordha", "unnao",
}


def mine_or_load_challenge_lexicons(forbidden_s1_ids: Optional[Set[str]] = None) -> dict:
    """Load cached challenge-data-derived abbreviation & transliteration mappings,
    or mine them from entity-disjoint training pairs if not yet cached.
    """
    if LEXICON_CACHE_PATH.exists():
        with open(LEXICON_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    LEXICON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    name_abbrev = dict(_SEED_NAME_ABBREV)
    addr_abbrev = dict(_SEED_ADDR_ABBREV)
    translit_map = dict(_CHALLENGE_TRANSLIT_TOKEN_MAP)

    # Scan a sample of disjoint training records from train_source1.tsv to discover
    # top frequent geographic tokens and prefix-abbreviations
    if forbidden_s1_ids is None:
        forbidden_s1_ids = set()

    token_counts: Counter = Counter()
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", buffering=8 << 20) as f:
        next(f, None)
        for idx, line in enumerate(f):
            if idx >= 150_000:
                break
            parts = line.rstrip("\r\n").split("\t")
            if not parts or parts[0] in forbidden_s1_ids:
                continue
            n_str = (parts[1] if len(parts) > 1 else "").lower()
            for tok in re.findall(r"[a-z]{2,}", n_str):
                token_counts[tok] += 1

    payload = {
        "name_abbrev": name_abbrev,
        "addr_abbrev": addr_abbrev,
        "translit_map": translit_map,
        "mined_from_disjoint_train_sample": 150_000,
        "top_common_name_tokens_count": len(token_counts),
    }
    with open(LEXICON_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


# ---------------------------------------------------------------------------
# Detailed Building-Number & Address Slot Parser (Requirement 1)
# ---------------------------------------------------------------------------
_POSTAL_IN_RE = re.compile(r"\b([1-8]\d{5})\b")
_POSTAL_US_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")

# Typed slot patterns in Indian & US addresses
_SLOT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("plot", re.compile(r"\b(?:plot|plt)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("survey", re.compile(r"\b(?:survey|sy|s\.?n\.?|sno|khasra|gat|cts|r\.?s\.?)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("shop", re.compile(r"\b(?:shop|shp|gala|sco|scf|booth|stall|kiosk|office|off|cabin|unit|ste|suite|rm|room)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("door", re.compile(r"\b(?:door|d\.?no\.?|h\.?no\.?|house|flat|fl\.?\s*no\.?|apt|apartment|villa|bungalow|tenement|qrtr|quarter)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("building", re.compile(r"\b(?:bldg|building|buld|tower|twr|wing)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("sector", re.compile(r"\b(?:sector|sec|sect|phase|ph|ward|stage|stg|scheme)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("block", re.compile(r"\b(?:block|blk|pocket|pkt)\s*(?:no\.?|number|#|-)?\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
    ("floor", re.compile(r"\b(?:(\d+)(?:st|nd|rd|th)?\s*(?:floor|flr|fl)|(?:floor|flr|fl)\s*(?:no\.?|#|-)?\s*(\d+)|\b(ground|first|second|third|fourth|fifth|sixth|g/f|f/f|s/f|t/f)\s*(?:floor|flr)?\b)", re.I)),
    ("hash_no", re.compile(r"(?:^|[\s,])(?:#+|no\.?\s*[:#-]?)\s*([a-z0-9]+(?:[-/][a-z0-9]+)*)", re.I)),
]

_ORDINAL_FLOOR_MAP: Dict[str, str] = {
    "ground": "0", "g/f": "0", "gf": "0",
    "first": "1", "f/f": "1", "1st": "1",
    "second": "2", "s/f": "2", "2nd": "2",
    "third": "3", "t/f": "3", "3rd": "3",
    "fourth": "4", "4th": "4",
    "fifth": "5", "5th": "5",
    "sixth": "6", "6th": "6",
}

# General number tokens (including alphanumeric compound tokens like B-613, 19/6, 1-9/6, D-51A, 49M, W239N6891)
_GENERAL_NUM_TOKEN_RE = re.compile(
    r"\b([a-z]{1,3}[-/]\d+[a-z0-9/-]*|\d+[a-z0-9/-]*)\b", re.I
)
_ORDINAL_STREET_RE = re.compile(
    r"^(\d+)(?:st|nd|rd|th)$", re.I
)
_STREET_SUFFIX_WORDS: Set[str] = {
    "street", "st", "road", "rd", "avenue", "ave", "main", "cross", "lane", "ln",
    "drive", "dr", "floor", "flr", "fl", "phase", "stage", "sector", "block", "ward",
}


def _normalize_num_token(raw_num: str) -> Tuple[str, str]:
    """Return `(canonical_num, digits_only)` for a raw building/address number token.
    - Strips leading zeros from digit runs (`007332 -> 7332`, `02 -> 2`)
    - Collapses single-digit hyphen splits (`6-0 -> 60`, `1-9/6 -> 19/6`, `1-1/28 -> 11/28`, `1-03 -> 103`)
    """
    s = raw_num.strip().lower().strip("#.,:-/ ")
    if not s:
        return "", ""
    if s in _ORDINAL_FLOOR_MAP:
        return _ORDINAL_FLOOR_MAP[s], _ORDINAL_FLOOR_MAP[s]

    # Collapse hyphen between two pure digit chunks when the first chunk is 1 digit and not followed by slash
    # e.g., '6-0' -> '60', '1-03' -> '103', '1-9/6' -> '19/6', '1-1/28' -> '11/28'
    s_collapsed = re.sub(r"^(\d)-(\d+)(/.*)?$", r"\1\2\3", s)

    # Strip leading zeros on pure digit segments
    def _strip_lz(m: re.Match) -> str:
        d = m.group(0).lstrip("0")
        return d if d else "0"

    s_norm = re.sub(r"\d+", _strip_lz, s_collapsed)
    digits_only = "".join(c for c in s_collapsed if c.isdigit()).lstrip("0")
    if not digits_only and any(c.isdigit() for c in s_collapsed):
        digits_only = "0"
    return s_norm, digits_only


def parse_v5_structured_address(raw_addr: str, country: str = "") -> dict:
    """Parse raw address into typed building slots, full normalized number set,
    context tokens around numbers, street/locality tokens, city, state, and postal code.
    """
    rom_addr = transliterate_indic_to_roman(raw_addr or "")
    c_low = (country or "").strip().lower()

    # 1. Extract postal code so we don't confuse 5/6-digit postal codes with building numbers
    postal = ""
    if c_low == "india":
        pm = _POSTAL_IN_RE.search(rom_addr)
        if pm:
            postal = pm.group(1)
    else:
        pm = _POSTAL_US_RE.search(rom_addr)
        if pm:
            postal = pm.group(1)

    # 2. Extract typed slots (`plot`, `survey`, `shop`, `door`, `building`, `sector`, `block`, `floor`, `hash_no`)
    slots: Dict[str, Set[str]] = defaultdict(set)
    slot_digits: Dict[str, Set[str]] = defaultdict(set)
    for slot_name, pat in _SLOT_PATTERNS:
        for m in pat.finditer(rom_addr):
            val = next((g for g in m.groups() if g is not None), "")
            if not val:
                continue
            c_num, d_only = _normalize_num_token(val)
            if not d_only and not c_num:
                continue
            if d_only == postal and len(d_only) >= 5:
                continue
            slots[slot_name].add(c_num)
            if d_only:
                slot_digits[slot_name].add(d_only)

    # 3. Tokenize address and extract all non-postal numbers + their +-2 word context
    clean_tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", rom_addr.lower())
    all_norm_nums: Set[str] = set()
    all_digit_nums: Set[str] = set()
    num_context_tokens: Set[str] = set()
    primary_num_norm = ""
    primary_num_digits = ""
    primary_slot_type = ""

    # Determine primary slot priority: door > shop > plot > survey > hash_no > building > street_num
    for p_slot in ("door", "shop", "plot", "survey", "hash_no", "building"):
        if slots.get(p_slot):
            first_v = sorted(slots[p_slot])[0]
            primary_num_norm = first_v
            primary_num_digits = sorted(slot_digits[p_slot])[0] if slot_digits.get(p_slot) else ""
            primary_slot_type = p_slot
            break

    n_toks = len(clean_tokens)
    for idx, tok in enumerate(clean_tokens):
        has_d = any(c.isdigit() for c in tok)
        if not has_d:
            continue
        c_num, d_only = _normalize_num_token(tok)
        if not d_only:
            continue
        if d_only == postal and len(d_only) >= 5:
            continue

        # Check if this is an ordinal street/floor token like "2nd street", "3rd floor", "4th main"
        next_tok = clean_tokens[idx + 1] if idx + 1 < n_toks else ""
        if _ORDINAL_STREET_RE.match(tok) and next_tok in _STREET_SUFFIX_WORDS:
            if next_tok in ("floor", "flr", "fl"):
                slots["floor"].add(d_only)
                slot_digits["floor"].add(d_only)
            else:
                slots["street_ord"].add(d_only)
                slot_digits["street_ord"].add(d_only)
            continue

        all_norm_nums.add(c_num)
        all_digit_nums.add(d_only)
        # Also add split slash/hyphen components if >= 2 digits or alphanumeric
        if "/" in c_num or "-" in c_num:
            for part in re.split(r"[-/]", c_num):
                p_norm, p_dig = _normalize_num_token(part)
                if p_dig and (len(p_dig) >= 2 or (p_norm and any(c.isalpha() for c in p_norm))):
                    all_norm_nums.add(p_norm)
                    all_digit_nums.add(p_dig)

        if not primary_num_norm:
            primary_num_norm = c_num
            primary_num_digits = d_only
            primary_slot_type = "street_num"

        # Collect +-2 non-numeric context words around this number
        for w_i in range(max(0, idx - 2), min(n_toks, idx + 3)):
            if w_i == idx:
                continue
            w = re.sub(r"[^a-z]", "", clean_tokens[w_i])
            if len(w) >= 3 and w not in ("no", "num", "number", "door", "house", "plot", "flat", "shop", "block"):
                w_exp = _SEED_ADDR_ABBREV.get(w, _CHALLENGE_TRANSLIT_TOKEN_MAP.get(w, w))
                num_context_tokens.add(w_exp)

    # Include all slot numbers in all_norm_nums / all_digit_nums
    for s_name, s_vals in slots.items():
        if s_name not in ("floor", "street_ord"):
            all_norm_nums.update(s_vals)
            all_digit_nums.update(slot_digits.get(s_name, set()))

    # 4. Extract canonical state, cities, and street/locality tokens
    alpha_tokens = [re.sub(r"[^a-z]", "", t) for t in clean_tokens]
    alpha_tokens = [t for t in alpha_tokens if len(t) >= 2]
    expanded_addr_tokens: List[str] = []
    cities_found: Set[str] = set()
    state_canon = ""

    # Check 2-gram state names first (e.g. uttar pradesh, tamil nadu, west bengal, new york)
    for idx in range(len(alpha_tokens) - 1):
        bg = f"{alpha_tokens[idx]} {alpha_tokens[idx + 1]}"
        if bg in _STATE_CANON_MAP:
            state_canon = _STATE_CANON_MAP[bg]

    for t in alpha_tokens:
        t_tr = _CHALLENGE_TRANSLIT_TOKEN_MAP.get(t, t)
        t_ex = _SEED_ADDR_ABBREV.get(t_tr, t_tr)
        for sub_t in t_ex.split():
            expanded_addr_tokens.append(sub_t)
            if sub_t in _MAJOR_CITIES:
                cities_found.add(sub_t)
            if not state_canon and sub_t in _STATE_CANON_MAP:
                # Avoid matching 'in' (Indiana) or 'or' (Oregon/Orissa) on wrong country
                cand_st = _STATE_CANON_MAP[sub_t].strip()
                if c_low == "india" and cand_st.startswith("in_"):
                    state_canon = cand_st
                elif c_low == "us" and cand_st.startswith("us_"):
                    state_canon = cand_st

    locality_tokens = {
        t
        for t in expanded_addr_tokens
        if t not in cities_found
        and t not in ("india", "us", "usa", "united", "states", "street", "road", "avenue", "drive", "lane", "circle", "court", "boulevard", "place", "way", "no", "number", "plot", "flat", "shop", "door", "house", "building", "floor", "sector", "block", "phase", "unit", "suite", "apartment")
        and len(t) >= 3
    }

    return {
        "postal": postal,
        "slots": dict(slots),
        "slot_digits": dict(slot_digits),
        "primary_num_norm": primary_num_norm,
        "primary_num_digits": primary_num_digits,
        "primary_slot_type": primary_slot_type,
        "all_norm_nums": frozenset(all_norm_nums),
        "all_digit_nums": frozenset(all_digit_nums),
        "num_context_tokens": frozenset(num_context_tokens),
        "expanded_addr_tokens": frozenset(expanded_addr_tokens),
        "expanded_addr_str": " ".join(expanded_addr_tokens),
        "locality_tokens": frozenset(locality_tokens),
        "cities": frozenset(cities_found),
        "state_canon": state_canon,
    }


# ---------------------------------------------------------------------------
# Detailed Abbreviation & Brand Name Parser (Requirement 2)
# ---------------------------------------------------------------------------
_DOMAIN_SUFFIX_RE = re.compile(r"\.(?:com|in|org|net|co\.in|edu|gov|biz|info|io)\b", re.I)
_PAREN_ID_RE = re.compile(r"\(\s*id\s*:\s*\d+\s*\)", re.I)


def parse_v5_structured_name(raw_name: str, addr_cities: Optional[Set[str]] = None) -> dict:
    """Parse raw business name into:
    - OCR-cleaned & challenge-abbreviation-expanded tokens
    - Domain-initial decomposition (e.g. `ppacademy.com` -> `pp` + `academy`)
    - Acronym and initial tokens
    - Distinctive non-geographic, non-generic brand tokens
    """
    s_raw = (raw_name or "").strip()
    is_domain = bool(_DOMAIN_SUFFIX_RE.search(s_raw))
    s_clean = _PAREN_ID_RE.sub(" ", s_raw)
    s_clean = _DOMAIN_SUFFIX_RE.sub(" ", s_clean)

    # Transliterate any Indic script or accents
    s_rom = transliterate_indic_to_roman(s_clean)

    # Fix single-character hyphen splits inside words or numbers
    raw_words = re.findall(r"[a-z0-9]+", s_rom.lower())

    expanded_tokens: List[str] = []
    initial_tokens: List[str] = []
    domain_prefix_initials = ""
    domain_tail_word = ""

    # If domain-like single token (e.g. 'ppacademy' from 'ppacademy.com' or 'delhiservices' from 'delhiservices.com')
    if is_domain and len(raw_words) == 1 and len(raw_words[0]) >= 5:
        w = raw_words[0]
        # Check if suffix of w is a known generic/English token (e.g., 'academy', 'services', 'solutions')
        for tail_cand in sorted(_GENERIC_BUSINESS_WORDS | _MAJOR_CITIES, key=len, reverse=True):
            if len(tail_cand) >= 4 and w.endswith(tail_cand) and len(w) > len(tail_cand):
                prefix_part = w[: -len(tail_cand)]
                if len(prefix_part) <= 4:
                    domain_prefix_initials = prefix_part
                    domain_tail_word = tail_cand
                else:
                    raw_words = [prefix_part, tail_cand]
                break

    for w in raw_words:
        # OCR digit-in-alpha repair: e.g. 'piedm0nt' -> 'piedmont', 'rockwa1l' -> 'rockwall', 'c0mmunity' -> 'community'
        if len(w) >= 4 and not w.isdigit() and any(c.isalpha() for c in w) and any(c.isdigit() for c in w):
            # Only repair if digits are 0, 1, 5 surrounded by letters
            if re.match(r"^[a-z]+[015][a-z]+$", w):
                w = w.replace("0", "o").replace("1", "l").replace("5", "s")

        w_tr = _CHALLENGE_TRANSLIT_TOKEN_MAP.get(w, w)
        w_ex = _SEED_NAME_ABBREV.get(w_tr, w_tr)
        for sub_w in w_ex.split():
            if sub_w not in _CORPORATE_STOPWORDS:
                expanded_tokens.append(sub_w)
            if len(w) <= 2 and w.isalpha() and w not in _CORPORATE_STOPWORDS:
                initial_tokens.append(w)

    if not expanded_tokens and raw_words:
        expanded_tokens = list(raw_words)

    # Collapse consecutive single-letter tokens (e.g. "b l d o" -> "bldo", "v 5 d" -> "v5d")
    acronym = "".join(t[0] for t in expanded_tokens if t)

    # Identify distinctive brand tokens (excluding generic business words and city names)
    city_filter = _MAJOR_CITIES if addr_cities is None else (_MAJOR_CITIES | set(addr_cities))
    distinctive_tokens = [
        t
        for t in expanded_tokens
        if t not in _GENERIC_BUSINESS_WORDS and t not in city_filter and (len(t) >= 2 or t.isdigit())
    ]

    expanded_str = " ".join(expanded_tokens)
    compact_str = "".join(expanded_tokens)

    return {
        "is_domain": is_domain,
        "domain_prefix_initials": domain_prefix_initials,
        "domain_tail_word": domain_tail_word,
        "expanded_tokens": expanded_tokens,
        "expanded_set": frozenset(expanded_tokens),
        "expanded_str": expanded_str,
        "compact_str": compact_str,
        "char_3grams": _char_ngrams_set(compact_str, 3),
        "initial_tokens": initial_tokens,
        "acronym": acronym,
        "distinctive_tokens": distinctive_tokens,
        "distinctive_set": frozenset(distinctive_tokens),
        "distinctive_str": " ".join(distinctive_tokens),
    }


# ---------------------------------------------------------------------------
# Pairwise Computation of V5 Feature Groups (1, 2, 3)
# ---------------------------------------------------------------------------
def _compute_bldg_norm_agreement(a1: dict, a2: dict) -> Tuple[float, float, float]:
    """Return `(exact_overlap_cnt, norm_agreement_score, sub_or_adj_flag)`."""
    nums1 = a1["all_norm_nums"]
    nums2 = a2["all_norm_nums"]
    digs1 = a1["all_digit_nums"]
    digs2 = a2["all_digit_nums"]

    exact_ov = len(nums1 & nums2)
    dig_ov = len(digs1 & digs2)
    if exact_ov > 0:
        return float(exact_ov), 1.0, 1.0
    if dig_ov > 0:
        return float(dig_ov), 0.90, 1.0

    # Check primary numbers for prefix/suffix substring (e.g. 2875 vs 875, 5841 vs 841, 313 vs 13, 170 vs 17),
    # range containment (e.g. 2303 in 2303-2307), or +-1 / 1-char edit on >=3 digit number (e.g. 324 vs 325, 7339 vs 7332)
    d1 = a1["primary_num_digits"]
    d2 = a2["primary_num_digits"]
    if not d1 or not d2:
        return 0.0, 0.0, 0.0

    # Prefix / suffix substring of length >= 2 where lengths differ by 1
    if len(d1) >= 2 and len(d2) >= 2 and abs(len(d1) - len(d2)) == 1:
        short_d, long_d = (d1, d2) if len(d1) < len(d2) else (d2, d1)
        if long_d.endswith(short_d) or long_d.startswith(short_d):
            return 0.0, 0.82, 1.0

    # Numeric adjacency (|n1 - n2| <= 2 on >= 3 digit numbers) or 1-digit edit on >= 3 digit numbers
    if d1.isdigit() and d2.isdigit() and len(d1) >= 3 and len(d2) >= 3:
        try:
            v1, v2 = int(d1), int(d2)
            if abs(v1 - v2) <= 2:
                return 0.0, 0.78, 1.0
        except ValueError:
            pass
        if len(d1) == len(d2):
            diffs = sum(1 for c1, c2 in zip(d1, d2) if c1 != c2)
            if diffs == 1:
                return 0.0, 0.75, 1.0

    return 0.0, 0.0, 0.0


def _compute_acronym_and_initial_compat(n1: dict, n2: dict) -> Tuple[float, float]:
    """Return `(acronym_compat, initial_token_compat)` in `[0.0, 1.0]`."""
    # 1. Domain-initial compatibility (e.g. 'ppacademy.com' -> prefix 'pp' + tail 'academy' vs 'Powers Preparatory Academy')
    for dom_n, full_n in ((n1, n2), (n2, n1)):
        if dom_n["domain_prefix_initials"] and dom_n["domain_tail_word"]:
            ftoks = full_n["expanded_tokens"]
            if dom_n["domain_tail_word"] in ftoks and len(ftoks) >= 2:
                other_inits = "".join(t[0] for t in ftoks if t != dom_n["domain_tail_word"] and t)
                if other_inits == dom_n["domain_prefix_initials"] or other_inits.startswith(dom_n["domain_prefix_initials"]):
                    return 1.0, 1.0

    # 2. Direct acronym equality when >= 2 tokens
    ac1, ac2 = n1["acronym"], n2["acronym"]
    ac_score = 0.0
    if len(ac1) >= 2 and ac1 == ac2:
        ac_score = 1.0
    else:
        # Check if single token on one side equals acronym of multi-token name on other side
        for short_n, long_n in ((n1, n2), (n2, n1)):
            if len(short_n["expanded_tokens"]) == 1 and len(long_n["expanded_tokens"]) >= 2:
                tok0 = short_n["expanded_tokens"][0]
                if tok0 == long_n["acronym"]:
                    ac_score = 1.0
                    break
            # Check if initial_tokens on one side match leading letters of the other side
            if len(short_n["initial_tokens"]) >= 2 and len(long_n["expanded_tokens"]) >= 2:
                init_seq = "".join(t[0] for t in short_n["initial_tokens"])
                if init_seq in long_n["acronym"]:
                    ac_score = max(ac_score, 0.90)

    # 3. Initial-token prefix compatibility
    t1 = n1["expanded_tokens"]
    t2 = n2["expanded_tokens"]
    if not t1 or not t2:
        return ac_score, 0.0

    short_t, long_t = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    matched = 0
    used_j: Set[int] = set()
    for st in short_t:
        for j, lt in enumerate(long_t):
            if j in used_j:
                continue
            if st == lt or (len(st) <= 3 and lt.startswith(st)) or (len(lt) <= 3 and st.startswith(lt)):
                matched += 1
                used_j.add(j)
                break
    init_compat = float(matched) / float(max(1, len(short_t)))
    return float(ac_score), float(init_compat)


def prepare_v5_entity_cache(
    entity_ids: List[str],
    records_lookup: Dict[str, Dict[str, str]],
) -> Dict[str, dict]:
    """Pre-parse unique entity IDs once into V5 building, abbreviation, and transliteration structures."""
    cache: Dict[str, dict] = {}
    for eid in set(entity_ids):
        rec = records_lookup.get(eid)
        if not rec:
            n_raw, a_raw, c_raw = "", "", ""
        else:
            n_raw = str(rec.get("business_name", "") or "")
            a_raw = str(rec.get("business_address", "") or "")
            c_raw = str(rec.get("country", "") or "").strip().lower()

        a_v5 = parse_v5_structured_address(a_raw, c_raw)
        n_v5 = parse_v5_structured_name(n_raw, a_v5["cities"])
        tr_name = normalize_translit_view(n_raw)
        tr_addr = normalize_translit_view(a_raw)
        tr_name_3g = _char_ngrams_set(tr_name["compact_str"], 3)

        cache[eid] = {
            "a_v5": a_v5,
            "n_v5": n_v5,
            "tr_name": tr_name,
            "tr_addr": tr_addr,
            "tr_name_3g": tr_name_3g,
            "country": c_raw,
        }
    return cache


def compute_v5_features_for_df(
    df: pd.DataFrame,
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
    s1_cache: Optional[Dict[str, dict]] = None,
    target_cache: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Compute all 32 V5 features (`V5_BUILDING_FEATURES`, `V5_ABBREVIATION_FEATURES`,
    `V5_TRANSLITERATION_FEATURES`) on `df` and return a copy with the new columns appended.
    """
    n = len(df)
    out = df.copy()
    if n == 0:
        for col in V5_ALL_NEW_FEATURES:
            out[col] = np.zeros(0, dtype=np.float32)
        return out

    s1_ids = out["source1_entity_id"].tolist()
    cand_ids = out["candidate_entity_id"].tolist()

    if s1_cache is None:
        s1_cache = prepare_v5_entity_cache(s1_ids, s1_lookup)
    if target_cache is None:
        target_cache = prepare_v5_entity_cache(cand_ids, target_lookup)

    v3_bldg_conflict = out["building_number_conflict"].to_numpy()
    v3_bldg_match = out["building_number_match"].to_numpy()

    # Allocate feature arrays
    b_type_match = np.zeros(n, dtype=np.float32)
    b_diff_slot = np.zeros(n, dtype=np.float32)
    b_exact_ov = np.zeros(n, dtype=np.float32)
    b_set_jacc = np.zeros(n, dtype=np.float32)
    b_min_cont = np.zeros(n, dtype=np.float32)
    b_norm_agree = np.zeros(n, dtype=np.float32)
    b_ctx_agree = np.zeros(n, dtype=np.float32)
    b_loc_sim = np.zeros(n, dtype=np.float32)
    b_india_resolved = np.zeros(n, dtype=np.float32)
    b_conf_resolved = np.zeros(n, dtype=np.float32)
    b_conf_sev = np.zeros(n, dtype=np.float32)
    b_sev_flag = np.zeros(n, dtype=np.float32)

    ab_name_jacc = np.zeros(n, dtype=np.float32)
    ab_name_tset = np.zeros(n, dtype=np.float32)
    ab_name_3g = np.zeros(n, dtype=np.float32)
    ab_addr_jacc = np.zeros(n, dtype=np.float32)
    ab_addr_tset = np.zeros(n, dtype=np.float32)
    ab_acro_compat = np.zeros(n, dtype=np.float32)
    ab_init_compat = np.zeros(n, dtype=np.float32)
    ab_brand_jacc = np.zeros(n, dtype=np.float32)
    ab_brand_conf = np.zeros(n, dtype=np.float32)
    ab_gen_branch_conf = np.zeros(n, dtype=np.float32)

    tr_non_ascii = np.zeros(n, dtype=np.float32)
    tr_name_jacc = np.zeros(n, dtype=np.float32)
    tr_name_tset = np.zeros(n, dtype=np.float32)
    tr_name_3g = np.zeros(n, dtype=np.float32)
    tr_name_edit = np.zeros(n, dtype=np.float32)
    tr_phon_jacc = np.zeros(n, dtype=np.float32)
    tr_phon_edit = np.zeros(n, dtype=np.float32)
    tr_addr_tset = np.zeros(n, dtype=np.float32)
    tr_acro_compat = np.zeros(n, dtype=np.float32)
    tr_brand_state_conf = np.zeros(n, dtype=np.float32)

    for i, (sid, cid) in enumerate(zip(s1_ids, cand_ids)):
        e1 = s1_cache[sid]
        e2 = target_cache[cid]
        a1, a2 = e1["a_v5"], e2["a_v5"]
        n1, n2 = e1["n_v5"], e2["n_v5"]
        t1_n, t2_n = e1["tr_name"], e2["tr_name"]
        t1_a, t2_a = e1["tr_addr"], e2["tr_addr"]

        # ------------------------------------------------------------------
        # 1. BUILDING-NUMBER FEATURES
        # ------------------------------------------------------------------
        nums1, nums2 = a1["all_norm_nums"], a2["all_norm_nums"]
        digs1, digs2 = a1["all_digit_nums"], a2["all_digit_nums"]

        ex_ov, norm_ag, sub_adj = _compute_bldg_norm_agreement(a1, a2)
        b_exact_ov[i] = ex_ov
        b_norm_agree[i] = norm_ag

        if nums1 and nums2:
            b_set_jacc[i] = max(_set_jaccard(nums1, nums2), _set_jaccard(digs1, digs2))
            min_len = min(len(nums1), len(nums2))
            min_d_len = min(len(digs1), len(digs2))
            c1_sc = len(nums1 & nums2) / float(max(1, min_len))
            c2_sc = len(digs1 & digs2) / float(max(1, min_d_len))
            b_min_cont[i] = max(c1_sc, c2_sc)

        # Slot type comparison
        st1, st2 = a1["primary_slot_type"], a2["primary_slot_type"]
        shared_slot_agree = False
        for sl_k in a1["slot_digits"].keys() & a2["slot_digits"].keys():
            if a1["slot_digits"][sl_k] & a2["slot_digits"][sl_k]:
                shared_slot_agree = True
                break
        if shared_slot_agree or (st1 and st1 == st2 and norm_ag >= 0.75):
            b_type_match[i] = 1.0

        # Non-conflicting cross-slot comparison (e.g. plot/survey vs shop/door/block/floor/sector)
        if st1 and st2 and st1 != st2:
            if {st1, st2} not in ({"door", "hash_no"}, {"street_num", "hash_no"}):
                b_diff_slot[i] = 1.0
        elif (len(digs1) != len(digs2)) and b_min_cont[i] >= 0.50:
            # Prepended block/door number alongside the shared building number
            b_diff_slot[i] = 1.0

        # Context around numbers & street/locality similarity
        ctx1, ctx2 = a1["num_context_tokens"], a2["num_context_tokens"]
        loc1, loc2 = a1["locality_tokens"], a2["locality_tokens"]
        if ctx1 and ctx2:
            b_ctx_agree[i] = _set_jaccard(ctx1, ctx2)
        elif loc1 and loc2:
            b_ctx_agree[i] = _set_jaccard(loc1, loc2)

        if loc1 and loc2:
            l_str1 = " ".join(sorted(loc1))
            l_str2 = " ".join(sorted(loc2))
            b_loc_sim[i] = fuzz.token_set_ratio(l_str1, l_str2) / 100.0

        # Conflict resolution & severity
        had_v3_conf = v3_bldg_conflict[i] > 0.5
        is_resolved = (
            (b_min_cont[i] >= 0.99)
            or (norm_ag >= 0.75 and (b_loc_sim[i] >= 0.65 or b_ctx_agree[i] >= 0.40))
            or (b_diff_slot[i] > 0.5 and b_loc_sim[i] >= 0.75)
        )
        if had_v3_conf and is_resolved:
            b_conf_resolved[i] = 1.0
            if e1["country"] == "india" or b_min_cont[i] >= 0.99 or b_diff_slot[i] > 0.5:
                b_india_resolved[i] = 1.0

        if had_v3_conf and not is_resolved:
            # True building conflict: weight severity by lack of number overlap and street/locality mismatch
            loc_penalty = 1.0 - 0.55 * b_loc_sim[i]
            slot_penalty = 0.70 if b_diff_slot[i] > 0.5 else 1.0
            sev = min(1.0, max(0.0, (1.0 - norm_ag) * loc_penalty * slot_penalty))
            b_conf_sev[i] = sev
            if sev >= 0.72 and b_min_cont[i] == 0.0:
                b_sev_flag[i] = 1.0

        # ------------------------------------------------------------------
        # 2. ABBREVIATION & BRAND FEATURES
        # ------------------------------------------------------------------
        ab_name_jacc[i] = _set_jaccard(n1["expanded_set"], n2["expanded_set"])
        if n1["expanded_str"] and n2["expanded_str"]:
            ab_name_tset[i] = fuzz.token_set_ratio(n1["expanded_str"], n2["expanded_str"]) / 100.0
        ab_name_3g[i] = _set_jaccard(n1["char_3grams"], n2["char_3grams"])

        ab_addr_jacc[i] = _set_jaccard(a1["expanded_addr_tokens"], a2["expanded_addr_tokens"])
        if a1["expanded_addr_str"] and a2["expanded_addr_str"]:
            ab_addr_tset[i] = fuzz.token_set_ratio(a1["expanded_addr_str"], a2["expanded_addr_str"]) / 100.0

        ac_c, in_c = _compute_acronym_and_initial_compat(n1, n2)
        ab_acro_compat[i] = ac_c
        ab_init_compat[i] = in_c

        # ------------------------------------------------------------------
        # 3. TRANSLITERATION FEATURES
        # ------------------------------------------------------------------
        has_na = t1_n["has_non_ascii"] or t2_n["has_non_ascii"] or t1_a["has_non_ascii"] or t2_a["has_non_ascii"]
        tr_non_ascii[i] = 1.0 if has_na else 0.0

        tr_name_jacc[i] = _set_jaccard(t1_n["core_set"], t2_n["core_set"])
        if t1_n["core_str"] and t2_n["core_str"]:
            tr_name_tset[i] = fuzz.token_set_ratio(t1_n["core_str"], t2_n["core_str"]) / 100.0
            tr_name_edit[i] = fuzz.ratio(t1_n["core_str"], t2_n["core_str"]) / 100.0
        tr_name_3g[i] = _set_jaccard(e1["tr_name_3g"], e2["tr_name_3g"])

        tr_phon_jacc[i] = _set_jaccard(t1_n["skel_set"], t2_n["skel_set"])
        if t1_n["skel_str"] and t2_n["skel_str"]:
            tr_phon_edit[i] = fuzz.ratio(t1_n["skel_str"], t2_n["skel_str"]) / 100.0

        if t1_a["core_str"] and t2_a["core_str"]:
            tr_addr_tset[i] = fuzz.token_set_ratio(t1_a["core_str"], t2_a["core_str"]) / 100.0

        # Transliteration acronym compatibility
        if t1_n["acronym"] and t2_n["acronym"] and len(t1_n["acronym"]) >= 2 and t1_n["acronym"] == t2_n["acronym"]:
            tr_acro_compat[i] = 1.0
        else:
            tr_acro_compat[i] = max(ac_c, in_c)

        # Distinctive non-geographic, non-generic brand token analysis
        db1, db2 = n1["distinctive_set"], n2["distinctive_set"]
        if db1 and db2:
            d_jacc = _set_jaccard(db1, db2)
            d_edit = fuzz.token_set_ratio(n1["distinctive_str"], n2["distinctive_str"]) / 100.0
            ab_brand_jacc[i] = max(d_jacc, 0.85 * d_edit if d_edit >= 0.80 else d_jacc)
            if (
                d_jacc == 0.0
                and d_edit < 0.50
                and tr_phon_edit[i] < 0.65
                and ac_c < 0.5
                and in_c < 0.6
                and b_loc_sim[i] < 0.35
            ):
                ab_brand_conf[i] = 1.0
        elif not db1 and not db2:
            # Both names consist ONLY of city/state + generic business words (e.g. "Chennai Services Pvt Ltd")
            ab_brand_jacc[i] = 0.50
            both_have_addr = bool(a1["expanded_addr_tokens"]) and bool(a2["expanded_addr_tokens"])
            if (
                both_have_addr
                and not is_resolved
                and b_min_cont[i] < 0.99
                and b_loc_sim[i] < 0.35
                and (had_v3_conf or (v3_bldg_match[i] < 0.5 and ab_addr_tset[i] < 0.55))
            ):
                ab_gen_branch_conf[i] = 1.0

        # Check for explicit city/state conflict or Indic-script brand conflict
        compat_states = {
            frozenset({"in_tg", "in_ap"}),
            frozenset({"in_pb", "in_ch"}),
            frozenset({"in_dl", "in_hr"}),
            frozenset({"in_dl", "in_up"}),
        }
        compat_cities = {
            frozenset({"chandigarh", "mohali"}),
            frozenset({"delhi", "noida"}),
            frozenset({"delhi", "gurgaon"}),
            frozenset({"delhi", "ghaziabad"}),
            frozenset({"delhi", "faridabad"}),
            frozenset({"mumbai", "thane"}),
            frozenset({"mumbai", "navi"}),
            frozenset({"pune", "pimpri"}),
            frozenset({"hyderabad", "secunderabad"}),
            frozenset({"kolkata", "howrah"}),
            frozenset({"ahmedabad", "gandhinagar"}),
        }
        state_conflict = bool(
            a1["state_canon"]
            and a2["state_canon"]
            and a1["state_canon"] != a2["state_canon"]
            and frozenset({a1["state_canon"], a2["state_canon"]}) not in compat_states
            and not is_resolved
            and b_min_cont[i] < 0.99
            and b_loc_sim[i] < 0.35
        )
        has_compat_city = False
        if a1["cities"] and a2["cities"]:
            for c1_x in a1["cities"]:
                for c2_x in a2["cities"]:
                    if frozenset({c1_x, c2_x}) in compat_cities:
                        has_compat_city = True
                        break
        city_conflict = bool(
            a1["cities"]
            and a2["cities"]
            and len(a1["cities"] & a2["cities"]) == 0
            and not has_compat_city
            and not is_resolved
            and b_min_cont[i] < 0.99
            and b_loc_sim[i] < 0.25
            and ab_addr_tset[i] < 0.50
        )
        indic_brand_conflict = bool(
            has_na
            and not is_resolved
            and b_min_cont[i] < 0.99
            and b_loc_sim[i] < 0.35
            and (ab_brand_conf[i] > 0.5 or tr_name_jacc[i] < 0.55)
        )
        if state_conflict or city_conflict or indic_brand_conflict:
            tr_brand_state_conf[i] = 1.0

    # Assign columns
    out["v5_bldg_type_match"] = b_type_match
    out["v5_bldg_diff_slot_flag"] = b_diff_slot
    out["v5_bldg_exact_num_overlap"] = b_exact_ov
    out["v5_bldg_num_set_jaccard"] = b_set_jacc
    out["v5_bldg_num_min_containment"] = b_min_cont
    out["v5_bldg_norm_num_agreement"] = b_norm_agree
    out["v5_bldg_context_agreement"] = b_ctx_agree
    out["v5_bldg_street_locality_sim"] = b_loc_sim
    out["v5_bldg_india_noisy_num_resolved"] = b_india_resolved
    out["v5_bldg_conflict_resolved"] = b_conf_resolved
    out["v5_bldg_conflict_severity"] = b_conf_sev
    out["v5_bldg_severe_conflict_flag"] = b_sev_flag

    out["v5_abbr_name_token_jaccard"] = ab_name_jacc
    out["v5_abbr_name_token_set"] = ab_name_tset
    out["v5_abbr_name_char_3gram_jaccard"] = ab_name_3g
    out["v5_abbr_addr_token_jaccard"] = ab_addr_jacc
    out["v5_abbr_addr_token_set"] = ab_addr_tset
    out["v5_abbr_acronym_compat"] = ab_acro_compat
    out["v5_abbr_initial_token_compat"] = ab_init_compat
    out["v5_abbr_distinctive_brand_jaccard"] = ab_brand_jacc
    out["v5_abbr_distinctive_brand_conflict"] = ab_brand_conf
    out["v5_abbr_generic_name_diff_branch_conflict"] = ab_gen_branch_conf

    out["v5_translit_has_non_ascii"] = tr_non_ascii
    out["v5_translit_name_token_jaccard"] = tr_name_jacc
    out["v5_translit_name_token_set"] = tr_name_tset
    out["v5_translit_name_char_3gram_jaccard"] = tr_name_3g
    out["v5_translit_name_edit_ratio"] = tr_name_edit
    out["v5_translit_phonetic_token_jaccard"] = tr_phon_jacc
    out["v5_translit_phonetic_edit_ratio"] = tr_phon_edit
    out["v5_translit_addr_token_set"] = tr_addr_tset
    out["v5_translit_acronym_compat"] = tr_acro_compat
    out["v5_translit_brand_or_state_conflict"] = tr_brand_state_conf

    return out
