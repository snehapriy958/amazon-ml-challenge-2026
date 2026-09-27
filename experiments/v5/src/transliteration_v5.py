"""V5 Challenge-Data-Derived Multilingual & Indic Transliteration Module.

Preserves raw/original normalization while adding a challenge-data-derived
transliteration view across:
  1. All 9 Indic Unicode script blocks in the challenge dataset (U+0900..U+0D7F:
     Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam),
     which share the standard ISCII 128-codepoint block layout.
  2. Romanized Indic phonetic/transliteration variations mined from disjoint
     challenge training ground-truth pairs (vowel elongation, aspirated stops,
     sibilant/labial equivalences, geminate consonants, and historical/official
     Indian city & state aliases).
  3. Phonetic consonant skeletons and transliteration-aware acronym compatibility.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# 1. Universal ISCII-Aligned Indic Unicode Block Transliterator (U+0900..U+0D7F)
# ---------------------------------------------------------------------------
# All 9 major Indic scripts in Unicode (0x0900..0x0D7F) follow the exact same
# 0x80 (128) codepoint offset table derived from ISCII.

_INDIC_INDEPENDENT_VOWELS: Dict[int, str] = {
    0x04: "a",
    0x05: "a",
    0x06: "a",
    0x07: "i",
    0x08: "i",
    0x09: "u",
    0x0A: "u",
    0x0B: "ri",
    0x0C: "li",
    0x0D: "e",
    0x0E: "e",
    0x0F: "e",
    0x10: "ai",
    0x11: "o",
    0x12: "o",
    0x13: "o",
    0x14: "au",
    0x60: "ri",
    0x61: "li",
    0x72: "a",
}

_INDIC_MATRAS: Dict[int, str] = {
    0x3E: "a",
    0x3F: "i",
    0x40: "i",
    0x41: "u",
    0x42: "u",
    0x43: "ri",
    0x44: "ri",
    0x45: "e",
    0x46: "e",
    0x47: "e",
    0x48: "ai",
    0x49: "o",
    0x4A: "o",
    0x4B: "o",
    0x4C: "au",
    0x55: "e",
    0x56: "ai",
    0x57: "au",
    0x62: "i",
    0x63: "i",
}

_INDIC_CONSONANTS: Dict[int, str] = {
    0x15: "k",
    0x16: "kh",
    0x17: "g",
    0x18: "gh",
    0x19: "ng",
    0x1A: "ch",
    0x1B: "chh",
    0x1C: "j",
    0x1D: "jh",
    0x1E: "ny",
    0x1F: "t",
    0x20: "th",
    0x21: "d",
    0x22: "dh",
    0x23: "n",
    0x24: "t",
    0x25: "th",
    0x26: "d",
    0x27: "dh",
    0x28: "n",
    0x29: "n",
    0x2A: "p",
    0x2B: "ph",
    0x2C: "b",
    0x2D: "bh",
    0x2E: "m",
    0x2F: "y",
    0x30: "r",
    0x31: "r",
    0x32: "l",
    0x33: "l",
    0x34: "zh",
    0x35: "v",
    0x36: "sh",
    0x37: "sh",
    0x38: "s",
    0x39: "h",
    # Precomposed nukta consonants (Devanagari/Gurmukhi/Bengali/Odia)
    0x58: "q",
    0x59: "kh",
    0x5A: "g",
    0x5B: "z",
    0x5C: "r",
    0x5D: "rh",
    0x5E: "f",
    0x5F: "y",
    0x79: "z",
    0x7A: "y",
    0x7B: "g",
    0x7C: "j",
    0x7E: "d",
    0x7F: "b",
    0x71: "v",  # Odia/Gurmukhi wa/va
}

_MALAYALAM_CHILLU: Dict[int, str] = {
    0x0D7A: "n",
    0x0D7B: "n",
    0x0D7C: "r",
    0x0D7D: "l",
    0x0D7E: "l",
    0x0D7F: "k",
}

_VIRAMA_OFFSET = 0x4D
_NUKTA_OFFSET = 0x3C
_ANUSVARA_OFFSETS = {0x01, 0x02, 0x70}  # chandrabindu, anusvara, gurmukhi tippi
_VISARGA_OFFSET = 0x03
_GURMUKHI_ADDAK_CP = 0x0A71


def has_non_ascii_or_indic(text: str) -> bool:
    """Return True if `text` contains any non-ASCII or Indic Unicode character."""
    if not text:
        return False
    for ch in text:
        if ord(ch) >= 128:
            return True
    return False


def transliterate_indic_to_roman(text: str) -> str:
    """Deterministically transliterate Indic Unicode scripts (U+0900..U+0D7F)
    and non-ASCII Latin diacritics into lowercase ASCII phonetic Roman script.
    """
    if not text:
        return ""
    if text.isascii():
        return text.lower()

    out: List[str] = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        cp = ord(ch)

        # Fast path for ASCII characters
        if cp < 128:
            out.append(ch.lower())
            i += 1
            continue

        # Malayalam atomic chillu consonants (no inherent vowel)
        if cp in _MALAYALAM_CHILLU:
            out.append(_MALAYALAM_CHILLU[cp])
            i += 1
            continue

        # Malayalam geminate rra (0x0D31 + 0x0D4D + 0x0D31) -> pronounced 'tt' ('t')
        if cp == 0x0D31 and i + 2 < n and ord(text[i + 1]) == 0x0D4D and ord(text[i + 2]) == 0x0D31:
            j = i + 3
            if j < n and 0x0900 <= ord(text[j]) <= 0x0D7F:
                noff = (ord(text[j]) - 0x0900) & 0x7F
                if noff == _VIRAMA_OFFSET:
                    out.append("t")
                    i = j + 1
                    continue
                if noff in _INDIC_MATRAS:
                    out.append("t" + _INDIC_MATRAS[noff])
                    i = j + 1
                    continue
            out.append("ta")
            i = j
            continue

        # Gurmukhi Addak (gemination marker)
        if cp == _GURMUKHI_ADDAK_CP:
            i += 1
            continue

        # Zero-width joiner / non-joiner
        if cp in (0x200C, 0x200D):
            i += 1
            continue

        # Indic Unicode blocks: U+0900 .. U+0D7F
        if 0x0900 <= cp <= 0x0D7F:
            offset = (cp - 0x0900) & 0x7F
            is_tamil = 0x0B80 <= cp <= 0x0BFF

            # Indic digits 0..9 (offset 0x66..0x6F)
            if 0x66 <= offset <= 0x6F:
                out.append(chr(ord("0") + (offset - 0x66)))
                i += 1
                continue

            # Independent vowels
            if offset in _INDIC_INDEPENDENT_VOWELS:
                out.append(_INDIC_INDEPENDENT_VOWELS[offset])
                i += 1
                continue

            # Standalone matras (if encountered without preceding consonant)
            if offset in _INDIC_MATRAS:
                out.append(_INDIC_MATRAS[offset])
                i += 1
                continue

            # Anusvara / Chandrabindu / Gurmukhi Tippi -> nasal 'n' (or 'm' before true labial p/b/m)
            if offset in _ANUSVARA_OFFSETS:
                next_nasal = "n"
                if i + 1 < n:
                    ncp = ord(text[i + 1])
                    if 0x0900 <= ncp <= 0x0D7F:
                        noff = (ncp - 0x0900) & 0x7F
                        if noff in (0x2A, 0x2C, 0x2D, 0x2E):
                            next_nasal = "m"
                out.append(next_nasal)
                i += 1
                continue

            if offset == _VISARGA_OFFSET:
                out.append("h")
                i += 1
                continue

            # Consonants
            if offset in _INDIC_CONSONANTS:
                base_c = _INDIC_CONSONANTS[offset]
                # In Tamil, 0x1A ('ca') represents 's' in loanwords unless geminated ('cc')
                if is_tamil and offset == 0x1A:
                    base_c = "s"
                j = i + 1
                # Check optional nukta (0x3C)
                if j < n and (0x0900 <= ord(text[j]) <= 0x0D7F) and ((ord(text[j]) - 0x0900) & 0x7F) == _NUKTA_OFFSET:
                    if offset == 0x1C:
                        base_c = "z"
                    elif offset == 0x2B:
                        base_c = "f"
                    elif offset == 0x15:
                        base_c = "q"
                    j += 1

                # Inspect what follows the consonant to decide vowel attachment
                if j < n:
                    ncp = ord(text[j])
                    if 0x0900 <= ncp <= 0x0D7F:
                        noff = (ncp - 0x0900) & 0x7F
                        if noff == _VIRAMA_OFFSET:
                            # Explicit virama/halant suppresses inherent vowel
                            out.append(base_c)
                            i = j + 1
                            continue
                        if noff in _INDIC_MATRAS:
                            out.append(base_c)
                            out.append(_INDIC_MATRAS[noff])
                            i = j + 1
                            continue
                    elif not text[j].isalnum():
                        # Word-final consonant: suppress inherent final schwa 'a' for clean English borrowing alignment
                        out.append(base_c)
                        i = j
                        continue
                else:
                    # String-final consonant: suppress final schwa
                    out.append(base_c)
                    i = j
                    continue

                # Default: consonant + inherent short vowel 'a'
                out.append(base_c + "a")
                i = j
                continue

            # Other punctuation/signs inside Indic block
            i += 1
            continue

        # Fallback for Latin accents / other Unicode scripts: NFKD ASCII fold
        folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii").lower()
        if folded:
            out.append(folded)
        i += 1

    return "".join(out)



# ---------------------------------------------------------------------------
# 2. Challenge-Data-Derived Transliteration & Loanword Equivalences
# ---------------------------------------------------------------------------
# Derived from entity-disjoint training GT pairs where English words are written
# in Indic scripts or alternative Roman transliterations.
_CHALLENGE_TRANSLIT_TOKEN_MAP: Dict[str, str] = {
    # Corporate / legal suffixes transliterated from Indic scripts
    "praivet": "private",
    "praibhet": "private",
    "prayvet": "private",
    "praiveta": "private",
    "pra": "private",
    "pvt": "private",
    "limited": "limited",
    "limiteda": "limited",
    "limitted": "limited",
    "li": "limited",
    "ltd": "limited",
    "elelpi": "llp",
    "elpi": "llp",
    "kampani": "company",
    "kampni": "company",
    "karporet": "corporate",
    "karporeshan": "corporation",
    "korporeshan": "corporation",
    # Common business / industry loanwords in Indic scripts
    "indastrij": "industries",
    "indastriz": "industries",
    "indastri": "industry",
    "인dastrij": "industries",
    "entarpraijej": "enterprises",
    "entarpraizes": "enterprises",
    "entarpraij": "enterprise",
    "teknolojij": "technologies",
    "teknolaji": "technology",
    "teknoloji": "technology",
    "teknalaji": "technology",
    "teknaloji": "technology",
    "inphotek": "infotech",
    "inaphotek": "infotech",
    "infotek": "infotech",
    "inphrastrakchar": "infrastructure",
    "imphrastrakchar": "infrastructure",
    "infrastrakchar": "infrastructure",
    "inphra": "infra",
    "sarvisej": "services",
    "sarvises": "services",
    "sarvis": "service",
    "solyushans": "solutions",
    "solushans": "solutions",
    "kansaltants": "consultants",
    "kansaltansi": "consultancy",
    "investament": "investment",
    "investment": "investment",
    "invesment": "investment",
    "invecssment": "investment",
    "hospitaliti": "hospitality",
    "hospitailiti": "hospitality",
    "hospital": "hospital",
    "prodakts": "products",
    "prodakt": "product",
    "bildars": "builders",
    "piltars": "builders",
    "bildar": "builder",
    "divalapars": "developers",
    "devalapars": "developers",
    "intarneshanal": "international",
    "intaraneshanal": "international",
    "treding": "trading",
    "tredars": "traders",
    "impeks": "impex",
    "eksports": "exports",
    "lojisticks": "logistics",
    "lojistiks": "logistics",
    "fainans": "finance",
    "phainans": "finance",
    "pharama": "pharma",
    "pharmasyutikals": "pharmaceuticals",
    "kemikals": "chemicals",
    "endjiniyaring": "engineering",
    "injiniyaring": "engineering",
    "konstrukshan": "construction",
    "kanstrakshan": "construction",
    "asosiyets": "associates",
    "markiting": "marketing",
    "distribyutars": "distributors",
    "manufakcharing": "manufacturing",
    "tekstails": "textiles",
    "garments": "garments",
    "motors": "motors",
    "motars": "motors",
    "elektrikals": "electricals",
    "elektroniks": "electronics",
    "sistams": "systems",
    "netvarks": "networks",
    "venshars": "ventures",
    "venchars": "ventures",
    "holdings": "holdings",
    "propartij": "properties",
    "riyalti": "realty",
    "fuds": "foods",
    "phuds": "foods",
    "speshaliti": "speciality",
    "heltcare": "healthcare",
    "helthkeyar": "healthcare",
    "medikal": "medical",
    "merdiceien": "medicine",
    "edukeshan": "education",
    "ekedami": "academy",
    "instityut": "institute",
    "phaondeshan": "foundation",
    "faundeshan": "foundation",
    "sosayati": "society",
    "sosaiti": "society",
    "kooperativ": "cooperative",
    # Common English brand words transliterated in Indic targets
    "hai": "high",
    "hvait": "white",
    "vhait": "white",
    "vait": "white",
    "yunaitad": "united",
    "yunaited": "united",
    "star": "star",
    "suprim": "supreme",
    "gret": "great",
    "parphekt": "perfect",
    "paraphekt": "perfect",
    "parfekt": "perfect",
    "sevan": "seven",
    "mai": "my",
    "silvar": "silver",
    "goldan": "golden",
    "daimand": "diamond",
    "royal": "royal",
    "globel": "global",
    "global": "global",
    "neshanal": "national",
    "sentral": "central",
    "modarn": "modern",
    "primiyar": "premier",
    "pioneer": "pioneer",
    "payoniyar": "pioneer",
    "payonir": "pioneer",
    "payoniar": "pioneer",
    "stendard": "standard",
    "sun": "sun",
    "surya": "surya",
    "sury": "surya",
    "lakshmi": "lakshmi",
    "laxmi": "lakshmi",
    "laksmi": "lakshmi",
    "ganesh": "ganesh",
    "ganesa": "ganesh",
    "krishna": "krishna",
    "krushna": "krishna",
    "krsna": "krishna",
    "vishnu": "vishnu",
    "bisnu": "vishnu",
    "mahadev": "mahadev",
    "mahadeb": "mahadev",
    "balaji": "balaji",
    "tirupati": "tirupati",
    "venkateshvara": "venkateswara",
    "venkateswara": "venkateswara",
    "agarwal": "agarwal",
    "aggarwal": "agarwal",
    "agrawal": "agarwal",
    "choudhary": "chaudhary",
    "chaudhari": "chaudhary",
    "chowdhury": "chaudhary",
    "choudhury": "chaudhary",
    "mukherjee": "mukherjee",
    "mukhopadhyay": "mukherjee",
    "banerjee": "banerjee",
    "bandyopadhyay": "banerjee",
    "chatterjee": "chatterjee",
    "chattopadhyay": "chatterjee",
    # Indian Cities & States (including transliterated Indic names & historical aliases)
    "bombay": "mumbai",
    "mumbai": "mumbai",
    "bangalore": "bangalore",
    "bengaluru": "bangalore",
    "bangaluru": "bangalore",
    "calcutta": "kolkata",
    "kolkata": "kolkata",
    "madras": "chennai",
    "chennai": "chennai",
    "gurgaon": "gurgaon",
    "gurugram": "gurgaon",
    "calicut": "kozhikode",
    "kozhikode": "kozhikode",
    "trivandrum": "thiruvananthapuram",
    "thiruvananthapuram": "thiruvananthapuram",
    "cochin": "kochi",
    "kochi": "kochi",
    "baroda": "vadodara",
    "vadodara": "vadodara",
    "mysore": "mysuru",
    "mysuru": "mysuru",
    "mangalore": "mangaluru",
    "mangaluru": "mangaluru",
    "pondicherry": "puducherry",
    "puducherry": "puducherry",
    "benares": "varanasi",
    "banaras": "varanasi",
    "varanasi": "varanasi",
    "allahabad": "prayagraj",
    "prayagraj": "prayagraj",
    "orissa": "odisha",
    "odisha": "odisha",
    "uttaranchal": "uttarakhand",
    "uttarakhand": "uttarakhand",
    "ahmadabad": "ahmedabad",
    "ahmedabad": "ahmedabad",
    "secunderabad": "hyderabad",
    "dilli": "delhi",
    "deli": "delhi",
    "maharastra": "maharashtra",
    "gujrat": "gujarat",
    "karnatak": "karnataka",
    "tamilnadu": "tamil",
    "paschimbanga": "west bengal",
    "paschim": "west",
    "banga": "bengal",
    "uttar": "uttar",
    "pradesh": "pradesh",
}

_CORPORATE_STOPWORDS: Set[str] = {
    "private", "limited", "pvt", "ltd", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "the", "and", "of",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_GEMINATE_RE = re.compile(r"([bcdfghjklmnpqrstvwxz])\1+")


def _canonicalize_roman_translit_token(tok: str) -> str:
    """Map a transliterated token to its canonical challenge-data-derived form."""
    if not tok:
        return ""
    if tok in _CHALLENGE_TRANSLIT_TOKEN_MAP:
        return _CHALLENGE_TRANSLIT_TOKEN_MAP[tok]

    # Normalize common Indic-English phonetic spellings
    t = tok
    # OCR digit-in-alpha fixes (e.g. piedm0nt -> piedmont, rockwa1l -> rockwall)
    if len(t) >= 4 and not t.isdigit():
        has_alpha = any(c.isalpha() for c in t)
        if has_alpha:
            t = t.replace("0", "o").replace("1", "l")
            if t in _CHALLENGE_TRANSLIT_TOKEN_MAP:
                return _CHALLENGE_TRANSLIT_TOKEN_MAP[t]

    # Strip trailing Indic schwa 'a' on long loanwords if it matches known stem
    if len(t) >= 5 and t.endswith("a") and t[:-1] in _CHALLENGE_TRANSLIT_TOKEN_MAP:
        return _CHALLENGE_TRANSLIT_TOKEN_MAP[t[:-1]]

    return t


def phonetic_consonant_skeleton(tok: str) -> str:
    """Compute a transliteration-invariant consonant skeleton for a token.
    Collapses Indic aspirated stops (bh->b, dh->d, gh->g, kh->k, ph->f, th->t, sh->s, chh->ch),
    sibilant/labial equivalences (v->w, z->j, c->k, q->k), geminate consonants, and internal vowels.
    """
    if not tok:
        return ""
    if tok.isdigit():
        return tok.lstrip("0") or "0"

    s = tok.lower()
    # Canonicalize digraphs before stripping vowels
    s = (
        s.replace("chh", "c")
        .replace("ch", "c")
        .replace("sh", "s")
        .replace("ph", "f")
        .replace("bh", "b")
        .replace("dh", "d")
        .replace("gh", "g")
        .replace("kh", "k")
        .replace("th", "t")
        .replace("wh", "w")
        .replace("zh", "j")
        .replace("ck", "k")
        .replace("qu", "k")
        .replace("x", "ks")
        .replace("v", "w")
        .replace("z", "j")
        .replace("y", "i")
    )
    # Normalize initial/internal 'c'
    out_chars = []
    for idx, ch in enumerate(s):
        if ch == "c":
            if idx + 1 < len(s) and s[idx + 1] in ("e", "i"):
                out_chars.append("s")
            else:
                out_chars.append("k")
        elif ch == "q":
            out_chars.append("k")
        else:
            out_chars.append(ch)
    s = "".join(out_chars)

    # Collapse geminates
    s = _GEMINATE_RE.sub(r"\1", s)

    # Keep first character (canonicalized if vowel), strip internal vowels
    if not s:
        return ""
    first = s[0]
    if first in "aeiou":
        first = "a"
    rest = [c for c in s[1:] if c not in "aeiou"]
    skel = first + "".join(rest)
    return _GEMINATE_RE.sub(r"\1", skel)


def normalize_translit_view(raw_text: str) -> dict:
    """Produce the V5 transliteration-normalized view for a name or address string."""
    has_non_ascii = has_non_ascii_or_indic(raw_text)
    rom = transliterate_indic_to_roman(raw_text or "")
    cleaned = _NON_ALNUM_RE.sub(" ", rom)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    raw_tokens = cleaned.split() if cleaned else []
    canon_tokens: List[str] = []
    for t in raw_tokens:
        mapped = _canonicalize_roman_translit_token(t)
        if mapped:
            for sub_t in mapped.split():
                canon_tokens.append(sub_t)

    core_tokens = [t for t in canon_tokens if t not in _CORPORATE_STOPWORDS]
    if not core_tokens and canon_tokens:
        core_tokens = list(canon_tokens)

    skel_tokens = [phonetic_consonant_skeleton(t) for t in core_tokens if len(t) >= 2 or t.isdigit()]
    skel_tokens = [s for s in skel_tokens if s]

    core_str = " ".join(core_tokens)
    compact_str = "".join(core_tokens)
    skel_str = "".join(skel_tokens)
    acronym = "".join(t[0] for t in core_tokens if t and t[0].isalnum())

    return {
        "has_non_ascii": has_non_ascii,
        "canon_tokens": canon_tokens,
        "core_tokens": core_tokens,
        "core_set": frozenset(core_tokens),
        "core_str": core_str,
        "compact_str": compact_str,
        "skel_tokens": skel_tokens,
        "skel_set": frozenset(skel_tokens),
        "skel_str": skel_str,
        "acronym": acronym,
    }
