"""V4 Normalization Module: Unicode/Diacritic Folding, Acronym Extraction, and Challenge-Derived Abbreviations.

Preserves all V3 normalized fields and the original representation (`raw_name`, `raw_addr`,
`orig_norm`), while adding:
  1. Full Unicode NFKC/NFKD + diacritic folding + Indic ISCII transliteration
  2. Acronym extraction (`acronym`) and single-letter initial collapsing (`collapsed_initials`)
  3. Challenge-data-derived business & address abbreviation expansion (`abbrev_name_norm`,
     `abbrev_name_tokens`, `abbrev_addr_norm`, `abbrev_addr_tokens`)
  4. Country-aware 3-digit postal prefix (`postal_prefix3`) for regional proximity
"""

from functools import lru_cache
import re
from typing import Dict, List, Set, Tuple
import unicodedata

from code.business_entity_resolution.src.normalization_v3 import (
    RE_ALNUM,
    SALUTATION_STOPWORDS,
    SUFFIX_MAP,
    TRANSLIT_TABLE,
    normalize_name_v3,
    parse_address_v3,
)

# Challenge-data-derived business name abbreviations (mined from US/India/France training records)
CHALLENGE_NAME_ABBR: Dict[str, str] = {
    "intl": "international",
    "int": "international",
    "natl": "national",
    "assn": "association",
    "assoc": "associates",
    "bros": "brothers",
    "mfg": "manufacturing",
    "mfrs": "manufacturers",
    "mfr": "manufacturer",
    "dist": "distributors",
    "distr": "distribution",
    "ent": "enterprises",
    "enterp": "enterprises",
    "entp": "enterprises",
    "svcs": "services",
    "svc": "service",
    "srv": "service",
    "srvs": "services",
    "tech": "technologies",
    "techn": "technology",
    "sys": "systems",
    "sol": "solutions",
    "soln": "solutions",
    "mgmt": "management",
    "mgt": "management",
    "mktg": "marketing",
    "hosp": "hospital",
    "med": "medical",
    "pharm": "pharmacy",
    "pharma": "pharmaceuticals",
    "coop": "cooperative",
    "dept": "department",
    "univ": "university",
    "inst": "institute",
    "ind": "industries",
    "inds": "industries",
    "indl": "industrial",
    "engg": "engineering",
    "eng": "engineering",
    "engr": "engineering",
    "const": "construction",
    "constr": "construction",
    "elec": "electrical",
    "elect": "electronics",
    "auto": "automotive",
    "comm": "communications",
    "comms": "communications",
    "tele": "telecom",
    "fin": "financial",
    "inv": "investments",
    "prop": "properties",
    "props": "properties",
    "dev": "development",
    "log": "logistics",
    "trans": "transport",
    "trdg": "trading",
    "trd": "trading",
    "impex": "import export",
    "chem": "chemicals",
    "chems": "chemicals",
    "tex": "textiles",
    "ag": "agency",
    "agy": "agency",
    "ets": "etablissements",
    "etab": "etablissements",
    "ste": "societe",
    "cie": "compagnie",
    "st": "saint",
    "ste": "sainte",
    "ctr": "center",
    "cntr": "center",
    "centre": "center",
    "grp": "group",
    "hldg": "holdings",
    "hldgs": "holdings",
    "labs": "laboratories",
    "lab": "laboratory",
    "vet": "veterinary",
    "dent": "dental",
    "opt": "optical",
    "ins": "insurance",
    "acc": "accounting",
    "acct": "accounting",
    "adv": "advertising",
    "cons": "consulting",
    "consult": "consulting",
    "govt": "government",
    "fed": "federal",
    "reg": "regional",
    "gen": "general",
    "std": "standard",
    "qual": "quality",
    "prem": "premier",
    "advncd": "advanced",
    "am": "american",
    "amer": "american",
    "indn": "indian",
    "fr": "france",
}

# Challenge-data-derived address abbreviations (US / India / France)
CHALLENGE_ADDR_ABBR: Dict[str, str] = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "drv": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "flr": "floor",
    "fl": "floor",
    "rm": "room",
    "dept": "department",
    "no": "number",
    "opp": "opposite",
    "nr": "near",
    "bh": "behind",
    "ch": "chawl",
    "ngr": "nagar",
    "col": "colony",
    "soc": "society",
    "ind": "industrial",
    "est": "estate",
    "sec": "sector",
    "ph": "phase",
    "mkt": "market",
    "marg": "road",
    "cross": "crossroad",
    "ext": "extension",
    "extn": "extension",
    "lyt": "layout",
    " layout": "layout",
    "rte": "route",
    "imp": "impasse",
    "all": "allee",
    "res": "residence",
    "bat": "batiment",
    "esc": "escalier",
    "etg": "etage",
    "zi": "zone industrielle",
    "za": "zone artisanale",
    "zac": "zone amenagement concerte",
    "bp": "boite postale",
    "twp": "township",
    "vlg": "village",
    "dist": "district",
    "tal": "taluka",
    "tq": "taluka",
    "po": "post office",
    "ps": "police station",
    "jct": "junction",
    "jn": "junction",
    "cir": "circle",
    "chk": "chowk",
    "chwk": "chowk",
    "rd": "road",
    "fl": "floor",
    "gr": "ground",
    "bsmt": "basement",
    "mezz": "mezzanine",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}


def fold_unicode_diacritics(text: str) -> str:
    """Fold Unicode characters, strip combining diacritical marks, and apply Indic ISCII transliteration."""
    if not text:
        return ""
    trans = text.translate(TRANSLIT_TABLE)
    nfkd = unicodedata.normalize("NFKD", trans)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return stripped.lower()


def _collapse_consecutive_initials(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Collapse adjacent single-letter alphabetic tokens into an acronym token (e.g. ('h','d','f','c','bank') -> ('hdfc','bank'))."""
    if not tokens:
        return ()
    out: List[str] = []
    buf: List[str] = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(t)
    if buf:
        out.append("".join(buf))
    return tuple(out)


def _extract_acronym(tokens: Tuple[str, ...]) -> str:
    """Extract multi-token acronym from core business tokens.
    Also returns a single token if the core name itself is a short 2..6 char token (potential acronym).
    """
    if not tokens:
        return ""
    collapsed = _collapse_consecutive_initials(tokens)
    if len(collapsed) == 1:
        tok = collapsed[0]
        if 2 <= len(tok) <= 6 and tok.isalpha():
            return tok
        return ""
    letters = [t[0] for t in collapsed if t and t[0].isalpha()]
    if len(letters) >= 2:
        return "".join(letters)
    return ""


@lru_cache(maxsize=250_000)
def normalize_name_v4(raw_name: str) -> Dict[str, object]:
    """V4 business name normalization: preserves original representation and V3 outputs while adding
    Unicode/diacritic folding, acronym extraction, collapsed initials, and challenge-derived abbreviation expansion.
    """
    v3_info = normalize_name_v3(raw_name)
    core_tokens: Tuple[str, ...] = v3_info["core_tokens"]

    # Collapse consecutive single-letter initials (e.g. "j p morgan" -> "jpmorgan")
    collapsed_tokens = _collapse_consecutive_initials(core_tokens)
    collapsed_norm = " ".join(collapsed_tokens)

    # Extract acronym of multi-word core name (or short acronym token)
    acronym = _extract_acronym(core_tokens)
    multi_word_initials = (
        "".join(t[0] for t in collapsed_tokens if t and t[0].isalpha())
        if len(collapsed_tokens) >= 2
        else ""
    )

    # Expand challenge-data-derived business abbreviations on collapsed core tokens
    expanded_toks: List[str] = []
    for t in collapsed_tokens:
        rep = CHALLENGE_NAME_ABBR.get(t, t)
        if " " in rep:
            expanded_toks.extend(rep.split())
        else:
            expanded_toks.append(rep)
    abbrev_tokens = tuple(expanded_toks)
    abbrev_norm = " ".join(abbrev_tokens)
    abbrev_compact = "".join(abbrev_tokens)

    return {
        "raw_name": raw_name or "",
        "orig_norm": v3_info["orig_norm"],
        "lower_norm": v3_info["lower_norm"],
        "compact_norm": v3_info["compact_norm"],
        "core_name_norm": v3_info["core_name_norm"],
        "tokens": v3_info["tokens"],
        "core_tokens": core_tokens,
        "suffix_class": v3_info["suffix_class"],
        "phonetic_tokens": v3_info["phonetic_tokens"],
        "phonetic_str": v3_info["phonetic_str"],
        "collapsed_tokens": collapsed_tokens,
        "collapsed_norm": collapsed_norm,
        "acronym": acronym,
        "multi_word_initials": multi_word_initials,
        "abbrev_tokens": abbrev_tokens,
        "abbrev_norm": abbrev_norm,
        "abbrev_compact": abbrev_compact,
    }


@lru_cache(maxsize=250_000)
def parse_address_v4(raw_addr: str, country: str = "") -> Dict[str, object]:
    """V4 address normalization: preserves original representation and V3 structured fields while adding
    challenge-data-derived address abbreviation expansion and 3-digit postal prefix.
    """
    v3_addr = parse_address_v3(raw_addr, country)
    addr_tokens: Tuple[str, ...] = v3_addr["tokens"]

    expanded_toks: List[str] = []
    for t in addr_tokens:
        rep = CHALLENGE_ADDR_ABBR.get(t, t)
        if " " in rep:
            expanded_toks.extend(rep.split())
        else:
            expanded_toks.append(rep)
    abbrev_tokens = tuple(expanded_toks)
    abbrev_norm = " ".join(abbrev_tokens)

    postal = str(v3_addr["postal_code"] or "")
    postal_prefix3 = postal[:3] if len(postal) >= 3 else ""

    return {
        "raw_addr": raw_addr or "",
        "orig_norm": v3_addr["orig_norm"],
        "lower_norm": v3_addr["lower_norm"],
        "tokens": addr_tokens,
        "building_num": v3_addr["building_num"],
        "building_conf": v3_addr["building_conf"],
        "unit_num": v3_addr["unit_num"],
        "unit_conf": v3_addr["unit_conf"],
        "postal_code": postal,
        "postal_conf": v3_addr["postal_conf"],
        "postal_prefix3": postal_prefix3,
        "state_code": v3_addr["state_code"],
        "numeric_tokens": v3_addr["numeric_tokens"],
        "abbrev_tokens": abbrev_tokens,
        "abbrev_norm": abbrev_norm,
    }
