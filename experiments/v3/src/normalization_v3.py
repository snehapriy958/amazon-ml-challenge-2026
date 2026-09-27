"""V3 Normalization, Country-Aware Address Parsing, Suffix Handling & Phonetic Encoding.

All normalization is non-destructive and produces structured representations for:
1. Business Name:
   - orig_norm: baseline ASCII alphanumeric tokenization
   - lower_norm: accent-stripped + Indic-transliterated lowercase normalized text
   - compact_norm: space/punctuation/handle-stripped compact string
   - core_name_norm: suffix/salutation-stripped business name
   - core_tokens: tuple of meaningful business name tokens
   - suffix_class: canonical legal suffix group (e.g. 'ltd', 'llc', 'inc', 'corp', 'llp', 'partnership')
   - phonetic_tokens: Double-Metaphone / consonant-skeleton codes of core tokens
2. Business Address:
   - lower_norm: accent-stripped + Indic-transliterated + state/street-expanded address
   - building_num: primary street/building/plot/house number (with confidence flag)
   - unit_num: extracted apartment/suite/unit/floor/gala/shop number (with confidence flag)
   - postal_code: country-aware ZIP5 / 6-digit PIN / 5-digit French postal code (with confidence flag)
   - state_code: canonical US state / Indian state / region code
   - numeric_tokens: set of normalized numeric tokens in address
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple


def _build_global_translit_table() -> Dict[int, str]:
    """Build a fast str.translate table that:
    1. Strips Latin accents/diacritics (e.g. Á -> a, í -> i, é -> e, ç -> c)
    2. Transliterates all 9 ISCII-aligned Indic Unicode blocks (U+0900..U+0D7F:
       Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam)
       into lowercase ASCII phonetic equivalents.
    """
    table: Dict[int, str] = {}

    # 1. Latin-1 Supplement & Latin Extended-A/B (U+00C0..U+024F)
    for cp in range(0x00C0, 0x0250):
        ch = chr(cp)
        decomp = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii").lower()
        if decomp:
            table[cp] = decomp

    # 2. ISCII-aligned Indic blocks: 0x0900 to 0x0D7F in steps of 0x80
    # Offsets within each 128-byte block:
    iscii_offsets: Dict[int, str] = {
        0x01: "n", 0x02: "n", 0x03: "h",
        # Independent vowels
        0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
        0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
        0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
        # Consonants
        0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
        0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
        0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
        0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
        0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
        0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
        0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
        # Dependent vowel signs (matras)
        0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
        0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai",
        0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
        # Virama / halant (suppresses inherent vowel)
        0x4D: "",
        # Nukta & extras
        0x3C: "", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z",
        0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
        # Indic digits 0..9
        0x66: "0", 0x67: "1", 0x68: "2", 0x69: "3", 0x6A: "4",
        0x6B: "5", 0x6C: "6", 0x6D: "7", 0x6E: "8", 0x6F: "9",
    }

    for block_base in range(0x0900, 0x0D80, 0x0080):
        for offset, lat in iscii_offsets.items():
            table[block_base + offset] = lat

    return table


TRANSLIT_TABLE = _build_global_translit_table()
RE_ALNUM = re.compile(r"[a-z0-9]+")
RE_DIGITS = re.compile(r"\d+")
RE_ID_NOISE = re.compile(r"\(\s*id\s*:?\s*[\(\d\)]+\s*\)", re.IGNORECASE)
RE_URL_AFFIX = re.compile(r"^(?:https?://)?(?:www\.)?|(?:\.(?:com|in|org|net|co\.in|fr|us|io|biz))$", re.IGNORECASE)

# Canonical Business Suffixes -> Suffix Class
SUFFIX_MAP: Dict[str, str] = {
    "pvt": "pvt_ltd",
    "private": "pvt_ltd",
    "ltd": "ltd",
    "limited": "ltd",
    "limted": "ltd",
    "lmtd": "ltd",
    "llc": "llc",
    "l.l.c": "llc",
    "pllc": "llc",
    "inc": "inc",
    "incorporated": "inc",
    "corp": "corp",
    "corporation": "corp",
    "co": "co",
    "company": "co",
    "cie": "co",
    "llp": "llp",
    "plc": "plc",
    "pc": "pc",
    "pa": "pc",
    "gmbh": "corp",
    "kgaa": "corp",
    "sarl": "llc",
    "sas": "corp",
    "sa": "corp",
    "eurl": "llc",
    "snc": "llp",
}

SALUTATION_STOPWORDS: Set[str] = {
    "mr", "mrs", "ms", "dr", "shri", "sri", "smt", "kumari", "m", "s", "messrs", "the", "of", "and"
}

# US State Full Name -> 2-letter Code
US_STATES: Dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
US_STATE_CODES: Set[str] = set(US_STATES.values())

# India State Full Name / Alias -> 2-letter Code
INDIA_STATES: Dict[str, str] = {
    "maharashtra": "mh", "karnataka": "ka", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "andhra pradesh": "ap", "delhi": "dl", "new delhi": "dl",
    "uttar pradesh": "up", "uttarpradesh": "up", "gujarat": "gj", "gujrat": "gj",
    "west bengal": "wb", "westbengal": "wb", "rajasthan": "rj", "madhya pradesh": "mp",
    "kerala": "kl", "Haryana": "hr", "haryana": "hr", "punjab": "pb", "bihar": "br",
    "odisha": "or", "orissa": "or", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg",
    "uttarakhand": "uk", "himachal pradesh": "hp", "goa": "ga", "chandigarh": "ch",
    "jammu and kashmir": "jk", "puducherry": "py", "pondicherry": "py", "tripura": "tr",
    "meghalaya": "ml", "manipur": "mn", "nagaland": "nl", "mizoram": "mz", "sikkim": "sk",
    "arunachal pradesh": "ar",
}
INDIA_STATE_CODES: Set[str] = set(INDIA_STATES.values())

# Common street suffix normalization
STREET_ABBR: Dict[str, str] = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct",
    "circle": "cir", "highway": "hwy", "parkway": "pkwy", "place": "pl", "square": "sq",
    "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "flr", "fl": "flr",
    "number": "no", "nagar": "ngr", "colony": "col", "society": "soc", "industrial": "ind",
    "sector": "sec", "phase": "ph",
}


from functools import lru_cache

@lru_cache(maxsize=250_000)
def phonetic_token_code(tok: str) -> str:
    """Fast phonetic / transliteration-invariant code for a single lowercase token.
    Collapses vowel variations, aspirated consonants (bh->b, dh->d, kh->k, gh->g, ph->f, th->t),
    sibilants (sh/z/s -> s), v/w, and duplicate letters.
    """
    if not tok:
        return ""
    if tok[0].isdigit():
        return tok.lstrip("0") or "0"
    s = tok.lower()
    # Multi-char phonetic replacements
    s = (
        s.replace("ph", "f")
        .replace("bh", "b")
        .replace("dh", "d")
        .replace("gh", "g")
        .replace("kh", "k")
        .replace("th", "t")
        .replace("sh", "s")
        .replace("chh", "c")
        .replace("ch", "c")
        .replace("ck", "k")
        .replace("qu", "k")
        .replace("w", "v")
        .replace("z", "s")
        .replace("x", "ks")
        .replace("y", "i")
    )
    first = s[0]
    # Keep first character (collapsed if vowel -> 'a'), strip internal vowels
    if first in "aeiou":
        first = "a"
    rest = [c for c in s[1:] if c not in "aeiou"]
    # Collapse adjacent duplicates
    out = [first]
    for c in rest:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def normalize_name_v3(raw_name: str) -> Dict[str, object]:
    """Compute non-destructive normalized business name representations."""
    if not raw_name or not isinstance(raw_name, str):
        return {
            "orig_norm": "",
            "lower_norm": "",
            "compact_norm": "",
            "core_name_norm": "",
            "tokens": (),
            "core_tokens": (),
            "suffix_class": "",
            "phonetic_tokens": (),
            "phonetic_str": "",
        }

    # Baseline original ASCII normalization
    orig_toks = RE_ALNUM.findall(raw_name.lower())
    orig_norm = " ".join(orig_toks)

    # Clean ID noise like '(ID: (2394))' and URL affixes
    cleaned = RE_ID_NOISE.sub(" ", raw_name)
    cleaned = RE_URL_AFFIX.sub("", cleaned.strip())
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]

    # Transliterate accents & Indic scripts
    trans = cleaned.translate(TRANSLIT_TABLE).lower()
    toks = RE_ALNUM.findall(trans)
    lower_norm = " ".join(toks)
    compact_norm = "".join(toks)

    # Separate legal suffixes and salutations without destroying core name
    core_toks: List[str] = []
    suffixes_found: Set[str] = set()
    for idx, t in enumerate(toks):
        if t in SUFFIX_MAP:
            suffixes_found.add(SUFFIX_MAP[t])
        elif t in SALUTATION_STOPWORDS and (idx == 0 or len(toks) > 2):
            continue
        else:
            core_toks.append(t)

    if not core_toks and toks:
        core_toks = list(toks)

    # Canonical suffix class (e.g. pvt + ltd -> pvt_ltd)
    if "pvt_ltd" in suffixes_found:
        suffix_class = "pvt_ltd"
    elif "ltd" in suffixes_found:
        suffix_class = "ltd"
    elif "llc" in suffixes_found:
        suffix_class = "llc"
    elif "inc" in suffixes_found:
        suffix_class = "inc"
    elif "corp" in suffixes_found:
        suffix_class = "corp"
    elif "llp" in suffixes_found:
        suffix_class = "llp"
    elif "plc" in suffixes_found:
        suffix_class = "plc"
    elif "pc" in suffixes_found:
        suffix_class = "pc"
    elif "co" in suffixes_found:
        suffix_class = "co"
    else:
        suffix_class = ""

    core_name_norm = " ".join(core_toks)
    phon_toks = tuple(phonetic_token_code(t) for t in core_toks if t)
    phon_str = " ".join(phon_toks)

    return {
        "orig_norm": orig_norm,
        "lower_norm": lower_norm,
        "compact_norm": compact_norm,
        "core_name_norm": core_name_norm,
        "tokens": tuple(toks),
        "core_tokens": tuple(core_toks),
        "suffix_class": suffix_class,
        "phonetic_tokens": phon_toks,
        "phonetic_str": phon_str,
    }


# Country-aware regexes for address parsing
RE_US_ZIP = re.compile(r"\b(\d{5})(?:-(\d{4}))?\b")
RE_INDIA_PIN = re.compile(r"\b([1-8]\d{5})\b")
RE_FRANCE_POSTAL = re.compile(r"\b(\d{5})\b")

# Building / plot / house number patterns
RE_INDIA_PLOT = re.compile(
    r"(?:plot|h\.?\s*no|house|door|d\.?\s*no|flat|shop|survey|sy|s\.?\s*no|wz|hn|gala|office|no\.?)\s*[:\-\.]?\s*([a-z]?\d+[a-z0-9/\-\.]*)",
    re.IGNORECASE,
)
RE_UNIT_NUM = re.compile(
    r"(?:unit|ste|suite|apt|apartment|flr|floor|fl|room|rm|gala|shop)\s*[:\-\.#]?\s*([0-9]+[a-z]?|[a-z]?[0-9]+)",
    re.IGNORECASE,
)
RE_ORDINAL_STREET = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
RE_LEADING_COMP_NUM = re.compile(r"(?:^|,\s*)(\d+(?:[\./\-][a-z0-9]+)+|\d+)\b")

INDIA_STATES_EXT: Dict[str, str] = {
    **INDIA_STATES,
    "uttara pradesha": "up",
    "gujarata": "gj",
    "maharastra": "mh",
}


def _norm_num_str(num_str: str) -> str:
    """Normalize a numeric or compound building string like '01375' -> '1375', '1839/7' -> '1839/7'."""
    parts = RE_DIGITS.findall(num_str)
    if not parts:
        return ""
    norm_parts = [p.lstrip("0") or "0" for p in parts]
    return "/".join(norm_parts)


def parse_address_v3(raw_addr: str, country: str = "") -> Dict[str, object]:
    """Country-aware address normalization and structured numeric/postal/state extraction."""
    if not raw_addr or not isinstance(raw_addr, str) or not raw_addr.strip():
        return {
            "orig_norm": "",
            "lower_norm": "",
            "tokens": (),
            "building_num": "",
            "building_conf": 0,
            "unit_num": "",
            "unit_conf": 0,
            "postal_code": "",
            "postal_conf": 0,
            "state_code": "",
            "numeric_tokens": frozenset(),
        }

    c_low = (country or "").strip().lower()
    orig_toks = RE_ALNUM.findall(raw_addr.lower())
    orig_norm = " ".join(orig_toks)

    # Transliterate accents & Indic scripts (so 'ગુજરાત' -> 'gujarat', 'उत्तर प्रदेश' -> 'uttara pradesha')
    trans = raw_addr.translate(TRANSLIT_TABLE).lower()

    # 1. Extract Postal Code (Country-Aware)
    postal_code = ""
    postal_conf = 0
    if c_low in ("india", "in"):
        m_pin = RE_INDIA_PIN.findall(trans)
        if m_pin:
            postal_code = m_pin[-1]
            postal_conf = 2
    elif c_low in ("us", "usa", "united states"):
        for m in RE_US_ZIP.finditer(trans):
            if m.start() > 4:
                postal_code = m.group(1)
                postal_conf = 2 if m.group(2) is None else 3
    elif c_low in ("france", "fr"):
        m_fr = RE_FRANCE_POSTAL.findall(trans)
        if m_fr:
            postal_code = m_fr[-1]
            postal_conf = 2
    else:
        m_pin = RE_INDIA_PIN.findall(trans)
        if m_pin:
            postal_code = m_pin[-1]
            postal_conf = 1

    # 2. Extract Unit / Floor / Suite Number
    unit_num = ""
    unit_conf = 0
    m_unit = RE_UNIT_NUM.search(trans)
    if m_unit:
        u_digits = RE_DIGITS.findall(m_unit.group(1))
        if u_digits:
            unit_num = u_digits[0].lstrip("0") or "0"
            unit_conf = 1

    # 3. Extract Primary Building / Street / Plot Number
    building_num = ""
    building_conf = 0

    if c_low in ("india", "in"):
        m_plot = RE_INDIA_PLOT.search(trans)
        if m_plot:
            b_cand = _norm_num_str(m_plot.group(1))
            if b_cand and b_cand != postal_code:
                building_num = b_cand
                building_conf = 2

    if not building_num:
        m_comp = RE_LEADING_COMP_NUM.search(trans.strip())
        if m_comp:
            b_cand = _norm_num_str(m_comp.group(1))
            if b_cand and b_cand != postal_code and b_cand != unit_num:
                building_num = b_cand
                building_conf = 2 if m_comp.start() == 0 else 1

    # 4. Tokenize & Normalize Street Suffixes + O(1) State Lookup
    raw_tokens = RE_ALNUM.findall(trans)
    norm_tokens: List[str] = []
    num_tokens: Set[str] = set()
    state_code = ""
    n_toks = len(raw_tokens)

    is_us = c_low in ("us", "usa")
    is_in = c_low in ("india", "in")

    for idx, t in enumerate(raw_tokens):
        if t.isdigit():
            nt = t.lstrip("0") or "0"
            norm_tokens.append(nt)
            if nt != postal_code:
                num_tokens.add(nt)
                if not building_num and len(nt) <= 5:
                    building_num = nt
                    building_conf = 1
        elif any(ch.isdigit() for ch in t):
            norm_tokens.append(t)
            if not RE_ORDINAL_STREET.match(t):
                for d in RE_DIGITS.findall(t):
                    nd = d.lstrip("0") or "0"
                    num_tokens.add(nd)
                    if not building_num:
                        building_num = nd
                        building_conf = 1
        else:
            mapped = STREET_ABBR.get(t, t)
            norm_tokens.append(mapped)
            if not state_code:
                if is_us:
                    if t in US_STATES:
                        state_code = US_STATES[t]
                    elif t in US_STATE_CODES:
                        state_code = t
                    elif idx + 1 < n_toks:
                        bg = f"{t} {raw_tokens[idx+1]}"
                        if bg in US_STATES:
                            state_code = US_STATES[bg]
                elif is_in:
                    if t in INDIA_STATES_EXT:
                        state_code = INDIA_STATES_EXT[t]
                    elif t in INDIA_STATE_CODES:
                        state_code = t
                    elif idx + 1 < n_toks:
                        bg = f"{t} {raw_tokens[idx+1]}"
                        if bg in INDIA_STATES_EXT:
                            state_code = INDIA_STATES_EXT[bg]

    dedup_tokens: List[str] = []
    for t in norm_tokens:
        if not dedup_tokens or dedup_tokens[-1] != t:
            dedup_tokens.append(t)

    lower_norm = " ".join(dedup_tokens)

    return {
        "orig_norm": orig_norm,
        "lower_norm": lower_norm,
        "tokens": tuple(dedup_tokens),
        "building_num": building_num,
        "building_conf": building_conf,
        "unit_num": unit_num,
        "unit_conf": unit_conf,
        "postal_code": postal_code,
        "postal_conf": postal_conf,
        "state_code": state_code,
        "numeric_tokens": frozenset(num_tokens),
    }


if __name__ == "__main__":
    # Quick unit verification on required test case:
    # "31 Eagle Park Circle" vs "46 Eagle Park Circle"
    p1 = parse_address_v3("Banner Elk, 31 Eagle Park Circle, NC", "us")
    p2 = parse_address_v3("Banner Elk, 46 Eagle Park Circle, North Carolina", "us")
    print("Test 1 (31 vs 46 Eagle Park Circle):")
    print("  p1:", p1)
    print("  p2:", p2)
    assert p1["building_num"] == "31" and p2["building_num"] == "46"
    assert p1["state_code"] == "nc" and p2["state_code"] == "nc"
    assert p1["building_num"] != p2["building_num"], "Must generate strong conflict!"

    # Test 2: Indic script transliteration
    n_tel = normalize_name_v3("వన్ సొల్యూషన్స్ ప్రైవేట్ లిమిటెడ్")
    a_guj = parse_address_v3("Ahmedabad, Ahmadabad City, 31 Aditya Plaza, ગુજરાત", "india")
    print("Test 2 (Telugu name & Gujarati state):")
    print("  Telugu name transliterated:", n_tel["lower_norm"], "| core:", n_tel["core_name_norm"])
    print("  Gujarati address transliterated:", a_guj["lower_norm"], "| building:", a_guj["building_num"], "| state:", a_guj["state_code"])
