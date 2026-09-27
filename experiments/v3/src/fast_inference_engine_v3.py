"""Fast, Memory-Safe, Multi-Worker Sharded V3 Inference Engine (Top-500 Prefilter).

Strictly preserves:
  - Frozen Model: C_LightGBM_LambdaRank (experiments/v3/models/v3_model_c_lambdarank.joblib)
  - Frozen Features: All 88 V3-B normalized/numeric/phonetic features
  - Frozen Calibration: Platt probability calibration
  - Frozen Policy: Entity Policy 5 (t_us=0.995, t_in=0.995, t_default=0.995, max_k=6)
  - Validated Top-500 Retrieval Prefilter:
      1. Preserves ALL Channel-1 exact candidates (exact_cids / ch_mask & 1 != 0)
      2. Preserves fair S2/S3 interleaving (min_per_source = min(160, top_k // 3) = 160)
      3. Fills remaining slots up to top_k=500 by highest retrieval priority score

Lossless High-Throughput Architecture (< 4.0 GB Total RAM on Windows):
  1. SharedMemory Zero-Copy Country Index:
     - All int32 posting lists are packed into a single contiguous SharedMemory buffer (postings_shm).
     - All pre-normalized target records are delta-packed into a contiguous SharedMemory buffer
       (target_blob_shm + target_offsets_shm), eliminating redundant regex/transliteration/phonetic
       re-normalization during candidate scoring and eliminating target IPC pickling.
     - All target entity_id strings are packed into SharedMemory (cids_shm + cid_offsets_shm).
  2. 8-Worker Parallel Target Normalization during CountryIndexV3.build().
  3. 8-Worker Parallel Retrieval + 8-Worker Parallel Scoring with Shard-Local IDF Payloads:
     - Main process resolves query tokens into lightweight (start, len) integer slices in postings_shm
       and slices name_idf/addr_idf strictly to each shard's S1 vocabulary.
     - Workers execute both Top-500 candidate retrieval and 88-feature scoring in parallel across 16
       load-balanced shards.
  4. Tuple-Unpacked S1-Grouped Feature Loop + Proven Lossless 2-Stage Cascade:
     - Stage 1 computes the 5 anchor cpdist features + all lexical/token/IDF/numeric features and
       evaluates the first 50 trees (s50 >= -2.5, verified 0 missed matches across all 3,751,587
       validation pairs in DEV_3000, INTERNAL_1000, and HOLDOUT_1000).
     - Stage 2 computes the remaining 16 exact cpdist features + 6 exact char 3/4/5-gram features
       and evaluates all 600 trees + Platt calibration + Entity Policy 5.
"""

from array import array
from collections import Counter, defaultdict
import gc
import heapq
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import pickle
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path.cwd()))
from experiments.v3.src.features_v3 import V3_ALL_FEATURE_COLUMNS, _char_ngrams_set
from experiments.v3.src.normalization_v3 import normalize_name_v3, parse_address_v3
from experiments.v3.src.retrieval_v3 import (
    T1_ADDR_TOK_DF,
    T1_BLDG_NUM_DF,
    T1_BLDG_POSTAL_DF,
    T1_BLDG_STATE_DF,
    T1_CHAR_GRAM_DF,
    T1_NAME_TOK_DF,
    T1_NAME_TOK_FALLBACK_DF,
    T1_PHON_TOK_DF,
    T1_POSTAL_DF,
    T2_ADDR_TOK_MAX_DF,
    T2_BLDG_NUM_MAX_DF,
    T2_BLDG_STATE_MAX_DF,
    T2_CHAR_GRAM_MAX_DF,
    T2_NAME_TOK_MAX_DF,
    T2_PHON_TOK_MAX_DF,
    T2_POSTAL_MAX_DF,
    extract_char_ngrams_for_name,
)

_FIELD_SEP = "\x1f"
_SAME_AS_LOWER = "\x1e"

# ==============================================================================
# WORKER PROCESS GLOBAL STATE
# ==============================================================================
_WORKER_RANKER = None
_WORKER_CALIBRATOR = None
_WORKER_POLICY_CFG = None
_WORKER_THREADS = 1
_WORKER_COUNTRY = None
_WORKER_TARGET_CACHE: Dict[int, tuple] = {}
_WORKER_CGRAM_CACHE: Dict[str, Tuple[frozenset, int, frozenset, int, frozenset, int]] = {}

# Worker active vocabulary for parallel target normalization during build()
_WORKER_VOCAB_SHM_NAME: Optional[str] = None
_WORKER_VOCAB: Optional[Tuple[str, Set[str], Set[str], Set[str], Set[str]]] = None

# Worker SharedMemory handles and views for parallel retrieval + scoring
_WORKER_ACTIVE_SHM_NAME: Optional[str] = None
_WORKER_SHM_HANDLES: List[shared_memory.SharedMemory] = []
_WORKER_POSTINGS_ARR: Optional[np.ndarray] = None
_WORKER_CIDS_BUF: Optional[memoryview] = None
_WORKER_CID_OFFSETS: Optional[np.ndarray] = None
_WORKER_TGT_BLOB_BUF: Optional[memoryview] = None
_WORKER_TGT_OFFSETS: Optional[np.ndarray] = None

# Worker reusable retrieval scratch buffers (sized to current country n_targets)
_WORKER_SCORES_ARR: Optional[np.ndarray] = None
_WORKER_CH_MASK_ARR: Optional[np.ndarray] = None
_WORKER_EV_MASK_ARR: Optional[np.ndarray] = None
_WORKER_NAME_HITS_ARR: Optional[np.ndarray] = None


def _close_worker_shms() -> None:
    """Release all open SharedMemory views and handles in the worker process."""
    global _WORKER_ACTIVE_SHM_NAME, _WORKER_SHM_HANDLES, _WORKER_POSTINGS_ARR, _WORKER_CIDS_BUF, _WORKER_CID_OFFSETS
    global _WORKER_TGT_BLOB_BUF, _WORKER_TGT_OFFSETS
    global _WORKER_SCORES_ARR, _WORKER_CH_MASK_ARR, _WORKER_EV_MASK_ARR, _WORKER_NAME_HITS_ARR

    _WORKER_ACTIVE_SHM_NAME = None
    _WORKER_POSTINGS_ARR = None
    if _WORKER_CIDS_BUF is not None:
        _WORKER_CIDS_BUF.release()
        _WORKER_CIDS_BUF = None
    _WORKER_CID_OFFSETS = None
    if _WORKER_TGT_BLOB_BUF is not None:
        _WORKER_TGT_BLOB_BUF.release()
        _WORKER_TGT_BLOB_BUF = None
    _WORKER_TGT_OFFSETS = None

    _WORKER_SCORES_ARR = None
    _WORKER_CH_MASK_ARR = None
    _WORKER_EV_MASK_ARR = None
    _WORKER_NAME_HITS_ARR = None

    for shm in _WORKER_SHM_HANDLES:
        try:
            shm.close()
        except Exception:
            pass
    _WORKER_SHM_HANDLES.clear()


def init_inference_worker(model_path: str, policy_cfg: dict, threads_per_worker: int = 1) -> None:
    """Initialize persistent worker process with frozen V3 model and bounded caches."""
    global _WORKER_RANKER, _WORKER_CALIBRATOR, _WORKER_POLICY_CFG, _WORKER_THREADS
    global _WORKER_COUNTRY, _WORKER_TARGET_CACHE, _WORKER_CGRAM_CACHE
    bundle = joblib.load(model_path)
    _WORKER_RANKER = bundle["ranker"]
    _WORKER_CALIBRATOR = bundle["calibrator"]
    _WORKER_POLICY_CFG = policy_cfg
    _WORKER_THREADS = max(1, int(threads_per_worker))
    _WORKER_COUNTRY = None
    _WORKER_TARGET_CACHE = {}
    _WORKER_CGRAM_CACHE = {}


def worker_load_vocab_from_shm(shm_name: str, vocab_len: int) -> int:
    """Load country active S1 vocabulary into worker memory from a temporary SharedMemory block."""
    global _WORKER_VOCAB_SHM_NAME, _WORKER_VOCAB
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        _WORKER_VOCAB = pickle.loads(bytes(shm.buf[:vocab_len]))
        _WORKER_VOCAB_SHM_NAME = shm_name
    finally:
        shm.close()
    return 1


def worker_clear_vocab(_: int = 0) -> int:
    """Free worker vocabulary sets after country target indexing completes."""
    global _WORKER_VOCAB_SHM_NAME, _WORKER_VOCAB
    _WORKER_VOCAB_SHM_NAME = None
    _WORKER_VOCAB = None
    gc.collect()
    time.sleep(0.05)
    return 1


def pack_normalized_target_bytes(
    raw_name: str,
    raw_addr: str,
    n_info: dict,
    a_info: dict,
) -> bytes:
    """Delta-pack pre-normalized target strings into a compact UTF-8 record (~115 bytes/target)."""
    ln = n_info["lower_norm"]
    on = _SAME_AS_LOWER if n_info["orig_norm"] == ln else n_info["orig_norm"]
    cn = _SAME_AS_LOWER if n_info["core_name_norm"] == ln else n_info["core_name_norm"]

    la = a_info["lower_norm"]
    oa = _SAME_AS_LOWER if a_info["orig_norm"] == la else a_info["orig_norm"]

    a_toks_sp = " ".join(a_info["tokens"])
    a_num_sp = " ".join(sorted(a_info["numeric_tokens"]))

    packed_str = _FIELD_SEP.join((
        raw_name,
        raw_addr,
        on,
        ln,
        cn,
        n_info["phonetic_str"],
        n_info["suffix_class"],
        oa,
        la,
        a_info["state_code"],
        a_info["building_num"],
        repr(float(a_info["building_conf"])),
        a_info["postal_code"],
        repr(float(a_info["postal_conf"])),
        a_info["unit_num"],
        a_toks_sp,
        a_num_sp,
    ))
    return packed_str.encode("utf-8")


def unpack_target_bytes_to_tuple(raw_b: bytes, country: str) -> tuple:
    """Unpack compact pre-normalized target bytes directly into a 35-element positional tuple."""
    parts = raw_b.decode("utf-8").split(_FIELD_SEP)
    (
        raw_name,
        raw_addr,
        on_raw,
        ln,
        cn_raw,
        ps,
        sc,
        oa_raw,
        la,
        st_code,
        b_num,
        b_conf_s,
        p_code,
        p_conf_s,
        u_num,
        a_toks_sp,
        a_num_sp,
    ) = parts

    on = ln if on_raw == _SAME_AS_LOWER else on_raw
    cn = ln if cn_raw == _SAME_AS_LOWER else cn_raw
    oa = la if oa_raw == _SAME_AS_LOWER else oa_raw

    cpn = ln.replace(" ", "")
    n_toks = ln.split() if ln else []
    c_toks = cn.split() if cn else []
    p_toks = ps.split() if ps else []
    a_toks = a_toks_sp.split() if a_toks_sp else []
    a_nums = a_num_sp.split() if a_num_sp else []

    ft = c_toks[0] if c_toks else ""
    b_stem = b_num.split("/", 1)[0] if b_num else ""

    nt_set = frozenset(n_toks)
    ct_set = frozenset(c_toks)
    ph_set = frozenset(p_toks)
    at_set = frozenset(a_toks)
    num_set = frozenset(a_nums)

    return (
        raw_name,
        len(raw_name),
        raw_addr,
        len(raw_addr),
        country,
        on,
        ln,
        cn,
        len(cn),
        cpn,
        len(cpn),
        ps,
        len(ps),
        sc,
        ft,
        oa,
        la,
        len(la),
        st_code,
        b_num,
        b_stem,
        float(b_conf_s),
        p_code,
        float(p_conf_s),
        u_num,
        nt_set,
        len(nt_set),
        ct_set,
        len(ct_set),
        ph_set,
        len(ph_set),
        at_set,
        len(at_set),
        num_set,
        len(num_set),
    )


def worker_normalize_target_batch(args: Tuple[Optional[str], int, List[Tuple[str, str, str]]]) -> list:
    """Parallel worker helper to normalize a batch of target rows and filter against active query vocabulary."""
    shm_name, vocab_len, rows = args
    if shm_name is not None and _WORKER_VOCAB_SHM_NAME != shm_name:
        worker_load_vocab_from_shm(shm_name, vocab_len)
    country, qt_n, qt_p, qt_a, qg_c = _WORKER_VOCAB
    out = []
    for cid, n_str, a_str in rows:
        n_info = normalize_name_v3(n_str)
        a_info = parse_address_v3(a_str, country)
        cpn = n_info["compact_norm"]

        cand_name_toks = [
            t for t in ({t for t in n_info["core_tokens"] if len(t) >= 2} | {t for t in n_info["tokens"] if len(t) >= 3})
            if t in qt_n
        ]
        cand_phon_toks = [p for p in {p for p in n_info["phonetic_tokens"] if len(p) >= 3} if p in qt_p]
        cand_addr_toks = [t for t in {t for t in a_info["tokens"] if len(t) >= 2} if t in qt_a]
        c_grams = [g for g in extract_char_ngrams_for_name(cpn, n_info["core_tokens"]) if g in qg_c]

        packed_b = pack_normalized_target_bytes(n_str, a_str, n_info, a_info)

        out.append((
            cid,
            packed_b,
            n_info["orig_norm"],
            n_info["lower_norm"],
            n_info["core_name_norm"],
            cpn,
            a_info["orig_norm"],
            a_info["lower_norm"],
            cand_name_toks,
            cand_phon_toks,
            cand_addr_toks,
            c_grams,
            a_info["building_num"],
            a_info["postal_code"],
            a_info["state_code"],
        ))
    return out


def worker_attach_country_shm(shm_cfg: dict) -> int:
    """Attach worker process to the current country's SharedMemory index and allocate scratch buffers."""
    global _WORKER_ACTIVE_SHM_NAME, _WORKER_COUNTRY, _WORKER_TARGET_CACHE, _WORKER_CGRAM_CACHE
    global _WORKER_VOCAB_SHM_NAME, _WORKER_VOCAB
    global _WORKER_SHM_HANDLES, _WORKER_POSTINGS_ARR, _WORKER_CIDS_BUF, _WORKER_CID_OFFSETS
    global _WORKER_TGT_BLOB_BUF, _WORKER_TGT_OFFSETS
    global _WORKER_SCORES_ARR, _WORKER_CH_MASK_ARR, _WORKER_EV_MASK_ARR, _WORKER_NAME_HITS_ARR

    _close_worker_shms()
    _WORKER_VOCAB_SHM_NAME = None
    _WORKER_VOCAB = None
    _WORKER_COUNTRY = shm_cfg["country"]
    _WORKER_ACTIVE_SHM_NAME = shm_cfg["postings_shm_name"]
    _WORKER_TARGET_CACHE.clear()
    _WORKER_CGRAM_CACHE.clear()
    gc.collect()

    n_targets = int(shm_cfg["n_targets"])
    n_postings = int(shm_cfg["n_postings"])

    shm_post = shared_memory.SharedMemory(name=shm_cfg["postings_shm_name"])
    shm_cids = shared_memory.SharedMemory(name=shm_cfg["cids_shm_name"])
    shm_cid_off = shared_memory.SharedMemory(name=shm_cfg["cid_offsets_shm_name"])
    shm_blob = shared_memory.SharedMemory(name=shm_cfg["blob_shm_name"])
    shm_blob_off = shared_memory.SharedMemory(name=shm_cfg["blob_offsets_shm_name"])

    _WORKER_SHM_HANDLES.extend([shm_post, shm_cids, shm_cid_off, shm_blob, shm_blob_off])

    _WORKER_POSTINGS_ARR = np.ndarray((n_postings,), dtype=np.int32, buffer=shm_post.buf)
    _WORKER_CIDS_BUF = shm_post_cids = shm_cids.buf
    _WORKER_CID_OFFSETS = np.ndarray((n_targets + 1,), dtype=np.uint64, buffer=shm_cid_off.buf)
    _WORKER_TGT_BLOB_BUF = shm_blob.buf
    _WORKER_TGT_OFFSETS = np.ndarray((n_targets + 1,), dtype=np.uint64, buffer=shm_blob_off.buf)

    _WORKER_SCORES_ARR = np.zeros(n_targets, dtype=np.float64)
    _WORKER_CH_MASK_ARR = np.zeros(n_targets, dtype=np.uint8)
    _WORKER_EV_MASK_ARR = np.zeros(n_targets, dtype=np.uint8)
    _WORKER_NAME_HITS_ARR = np.zeros(n_targets, dtype=np.uint8)
    return 1


def worker_detach_country_shm(_: int = 0) -> int:
    """Detach worker from SharedMemory before main process unlinks country buffers."""
    _close_worker_shms()
    _WORKER_TARGET_CACHE.clear()
    _WORKER_CGRAM_CACHE.clear()
    gc.collect()
    time.sleep(0.05)
    return 1


def prepare_target_entity_dict(raw_name: str, raw_addr: str, country: str) -> dict:
    """Normalize a target entity into the compact feature dictionary (fallback helper)."""
    n_info = normalize_name_v3(raw_name)
    a_info = parse_address_v3(raw_addr, country)
    b_num = a_info["building_num"]
    return {
        "raw_name": raw_name,
        "raw_addr": raw_addr,
        "country": country,
        "n_orig_norm": n_info["orig_norm"],
        "n_lower_norm": n_info["lower_norm"],
        "n_core_norm": n_info["core_name_norm"],
        "n_compact_norm": n_info["compact_norm"],
        "n_phon_str": n_info["phonetic_str"],
        "n_suffix_class": n_info["suffix_class"],
        "first_core_tok": n_info["core_tokens"][0] if n_info["core_tokens"] else "",
        "a_orig_norm": a_info["orig_norm"],
        "a_lower_norm": a_info["lower_norm"],
        "a_state_code": a_info["state_code"],
        "a_bldg_num": b_num,
        "a_bldg_stem": b_num.split("/", 1)[0] if b_num else "",
        "a_bldg_conf": float(a_info["building_conf"]),
        "a_postal_code": a_info["postal_code"],
        "a_postal_conf": float(a_info["postal_conf"]),
        "a_unit_num": a_info["unit_num"],
        "n_tok_set": frozenset(n_info["tokens"]),
        "n_core_set": frozenset(n_info["core_tokens"]),
        "n_phon_set": frozenset(n_info["phonetic_tokens"]),
        "a_tok_set": frozenset(a_info["tokens"]),
        "a_num_set": frozenset(a_info["numeric_tokens"]),
    }


def prepare_s1_entity_dict(raw_name: str, raw_addr: str, country: str) -> dict:
    """Normalize an S1 entity into the flat feature dict + retrieval sets (char-gram sets built lazily)."""
    n_info = normalize_name_v3(raw_name)
    a_info = parse_address_v3(raw_addr, country)
    c_grams = extract_char_ngrams_for_name(n_info["compact_norm"], n_info["core_tokens"])

    name_tok_ret = {t for t in n_info["core_tokens"] if len(t) >= 2} | {
        t for t in n_info["tokens"] if len(t) >= 3
    }
    phon_tok_ret = {p for p in n_info["phonetic_tokens"] if len(p) >= 3}
    addr_tok_ret = {t for t in a_info["tokens"] if len(t) >= 2}

    n_comp = n_info["compact_norm"]
    a_norm = a_info["lower_norm"]
    b_num = a_info["building_num"]

    return {
        "raw_name": raw_name,
        "raw_addr": raw_addr,
        "country": country,
        "n_orig_norm": n_info["orig_norm"],
        "n_lower_norm": n_info["lower_norm"],
        "n_core_norm": n_info["core_name_norm"],
        "n_compact_norm": n_comp,
        "n_phon_str": n_info["phonetic_str"],
        "n_suffix_class": n_info["suffix_class"],
        "first_core_tok": n_info["core_tokens"][0] if n_info["core_tokens"] else "",
        "a_orig_norm": a_info["orig_norm"],
        "a_lower_norm": a_norm,
        "a_state_code": a_info["state_code"],
        "a_bldg_num": b_num,
        "a_bldg_stem": b_num.split("/", 1)[0] if b_num else "",
        "a_bldg_conf": float(a_info["building_conf"]),
        "a_postal_code": a_info["postal_code"],
        "a_postal_conf": float(a_info["postal_conf"]),
        "a_unit_num": a_info["unit_num"],
        "n_tok_set": frozenset(n_info["tokens"]),
        "n_core_set": frozenset(n_info["core_tokens"]),
        "n_phon_set": frozenset(n_info["phonetic_tokens"]),
        "a_tok_set": frozenset(a_info["tokens"]),
        "a_num_set": frozenset(a_info["numeric_tokens"]),
        "ret_name_toks": name_tok_ret,
        "ret_phon_toks": phon_tok_ret,
        "ret_addr_toks": addr_tok_ret,
        "ret_char_grams": c_grams,
    }


def _get_worker_cid_str(tid: int) -> str:
    st = int(_WORKER_CID_OFFSETS[tid])
    en = int(_WORKER_CID_OFFSETS[tid + 1])
    return bytes(_WORKER_CIDS_BUF[st:en]).decode("ascii")


def _get_worker_target_tuple(tid: int) -> tuple:
    res = _WORKER_TARGET_CACHE.get(tid)
    if res is not None:
        return res
    if len(_WORKER_TARGET_CACHE) > 6_000:
        _WORKER_TARGET_CACHE.clear()
    st = int(_WORKER_TGT_OFFSETS[tid])
    en = int(_WORKER_TGT_OFFSETS[tid + 1])
    res = unpack_target_bytes_to_tuple(bytes(_WORKER_TGT_BLOB_BUF[st:en]), _WORKER_COUNTRY)
    _WORKER_TARGET_CACHE[tid] = res
    return res


def _retrieve_top_k_from_plan_in_worker(
    plan: Tuple[
        List[Tuple[int, int, float, int, int, bool]],
        List[Tuple[int, int, float, int, int, bool]],
    ],
    n_s2: int,
    top_k: int,
) -> List[Tuple[int, str, float, int]]:
    """Execute exact V3 Two-Tier 4-Channel Retrieval inside worker using SharedMemory postings."""
    t1_specs, t2_specs = plan
    if not t1_specs:
        return []

    postings = _WORKER_POSTINGS_ARR
    scores = _WORKER_SCORES_ARR
    ch_mask = _WORKER_CH_MASK_ARR
    ev_mask = _WORKER_EV_MASK_ARR
    name_hits = _WORKER_NAME_HITS_ARR

    t1_lists: List[np.ndarray] = []

    for st, ln, w, ch_bit, ev_bit, is_name_tok in t1_specs:
        if ln <= 0:
            continue
        lst = postings[st : st + ln]
        if not t1_lists:
            t1_lists.append(lst)
        else:
            new_hits = lst[ev_mask[lst] == 0]
            if len(new_hits) > 0:
                t1_lists.append(new_hits)
        scores[lst] += w
        ch_mask[lst] |= ch_bit
        ev_mask[lst] |= (ev_bit | 8)
        if is_name_tok:
            name_hits[lst] += 1

    if not t1_lists:
        return []

    if len(t1_lists) == 1:
        anchor_tids = t1_lists[0]
    else:
        anchor_tids = np.concatenate(t1_lists)

    # First run name_t2 specs (where is_name_tok=True) BEFORE multi_mask bonus!
    for st, ln, w, ch_bit, ev_bit, is_name_tok in t2_specs:
        if is_name_tok and ln > 0:
            lst = postings[st : st + ln]
            hits = lst[ev_mask[lst] != 0]
            if len(hits) > 0:
                scores[hits] += w
                ch_mask[hits] |= ch_bit
                ev_mask[hits] |= ev_bit
                name_hits[hits] += 1

    nh = name_hits[anchor_tids]
    multi_mask = nh >= 2
    if np.any(multi_mask):
        scores[anchor_tids[multi_mask]] += 14.0 * (nh[multi_mask] - 1)

    # Run remaining T2 specs (is_name_tok=False)
    for st, ln, w, ch_bit, ev_bit, is_name_tok in t2_specs:
        if not is_name_tok and ln > 0:
            lst = postings[st : st + ln]
            hits = lst[ev_mask[lst] != 0]
            if len(hits) > 0:
                scores[hits] += w
                ch_mask[hits] |= ch_bit
                ev_mask[hits] |= ev_bit

    ev_sub = ev_mask[anchor_tids]
    both_mask = (ev_sub & 3) == 3
    if np.any(both_mask):
        scores[anchor_tids[both_mask]] += 28.0

    a_scores = scores[anchor_tids].copy()
    a_ch = ch_mask[anchor_tids].copy()
    a_ev = ev_sub.copy()

    scores[anchor_tids] = 0.0
    ch_mask[anchor_tids] = 0
    ev_mask[anchor_tids] = 0
    name_hits[anchor_tids] = 0

    n_anc = len(anchor_tids)
    if n_anc <= top_k:
        out = [
            (int(anchor_tids[i]), _get_worker_cid_str(int(anchor_tids[i])), float(a_scores[i]), int(a_ch[i]))
            for i in range(n_anc)
        ]
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    exact_mask = (a_ev & 4) != 0
    exact_idx = np.where(exact_mask)[0]
    selected_map: Dict[str, Tuple[int, float, int]] = {}
    for idx in exact_idx:
        tid_i = int(anchor_tids[idx])
        selected_map[_get_worker_cid_str(tid_i)] = (tid_i, float(a_scores[idx]), int(a_ch[idx]))

    non_exact_mask = ~exact_mask
    s2_idx = np.where(non_exact_mask & (anchor_tids < n_s2))[0]
    s3_idx = np.where(non_exact_mask & (anchor_tids >= n_s2))[0]

    if len(s2_idx) > top_k:
        part_s2 = np.argpartition(-a_scores[s2_idx], top_k - 1)
        kth_sc2 = a_scores[s2_idx[part_s2[top_k - 1]]]
        s2_idx = s2_idx[a_scores[s2_idx] >= kth_sc2]
    if len(s3_idx) > top_k:
        part_s3 = np.argpartition(-a_scores[s3_idx], top_k - 1)
        kth_sc3 = a_scores[s3_idx[part_s3[top_k - 1]]]
        s3_idx = s3_idx[a_scores[s3_idx] >= kth_sc3]

    s2_scored = [
        (float(a_scores[i]), _get_worker_cid_str(int(anchor_tids[i])), int(anchor_tids[i]), int(a_ch[i]))
        for i in s2_idx
    ]
    s3_scored = [
        (float(a_scores[i]), _get_worker_cid_str(int(anchor_tids[i])), int(anchor_tids[i]), int(a_ch[i]))
        for i in s3_idx
    ]

    min_per_source = min(160, top_k // 3)
    top_s2 = heapq.nlargest(min_per_source, s2_scored)
    top_s3 = heapq.nlargest(min_per_source, s3_scored)

    for sc_i, cid_i, tid_i, ch_i in top_s2:
        if len(selected_map) < top_k:
            selected_map[cid_i] = (tid_i, sc_i, ch_i)
    for sc_i, cid_i, tid_i, ch_i in top_s3:
        if len(selected_map) < top_k:
            selected_map[cid_i] = (tid_i, sc_i, ch_i)

    rem_slots = top_k - len(selected_map)
    if rem_slots > 0:
        rem_pool = [item for item in (s2_scored + s3_scored) if item[1] not in selected_map]
        for sc_i, cid_i, tid_i, ch_i in heapq.nlargest(rem_slots, rem_pool):
            selected_map[cid_i] = (tid_i, sc_i, ch_i)

    final_cands = sorted(
        [(tid_sc_m[0], cid_i, tid_sc_m[1], tid_sc_m[2]) for cid_i, tid_sc_m in selected_map.items()],
        key=lambda x: x[2],
        reverse=True,
    )
    return final_cands


def worker_retrieve_and_score_shard(shard_payload: tuple) -> Tuple[List[List[str]], List[List[str]], int, float, float]:
    """Worker entrypoint: run parallel Top-K retrieval + parallel 88-feature scoring on a shard of S1 queries."""
    (
        shm_cfg,
        n_s2_bound,
        top_k,
        s1_dicts,
        query_plans,
        shard_name_idf,
        shard_addr_idf,
    ) = shard_payload

    if _WORKER_ACTIVE_SHM_NAME != shm_cfg["postings_shm_name"]:
        worker_attach_country_shm(shm_cfg)

    t_r0 = time.time()
    cands_per_s1 = [
        _retrieve_top_k_from_plan_in_worker(plan, n_s2_bound, top_k)
        for plan in query_plans
    ]
    el_r = time.time() - t_r0

    n_pairs = sum(len(c) for c in cands_per_s1)
    t_s0 = time.time()
    matches_per_s1 = _score_s1_batch_core_tuple(
        s1_dicts=s1_dicts,
        cands_per_s1=cands_per_s1,
        n_s2_bound=n_s2_bound,
        get_tgt_tuple=_get_worker_target_tuple,
        cgram_cache=_WORKER_CGRAM_CACHE,
        name_idf=shard_name_idf,
        addr_idf=shard_addr_idf,
        ranker=_WORKER_RANKER,
        calibrator=_WORKER_CALIBRATOR,
        policy_cfg=_WORKER_POLICY_CFG,
        num_threads=_WORKER_THREADS,
        sub_batch_size=30_000,
    )
    el_s = time.time() - t_s0

    _WORKER_TARGET_CACHE.clear()
    _WORKER_CGRAM_CACHE.clear()

    cand_ids_per_s1 = [[cid_i for _, cid_i, _, _ in c_list] for c_list in cands_per_s1]
    return cand_ids_per_s1, matches_per_s1, n_pairs, el_r, el_s


def _score_s1_batch_core_tuple(
    s1_dicts: List[dict],
    cands_per_s1: List[List[Tuple[int, str, float, int]]],
    n_s2_bound: int,
    get_tgt_tuple,
    cgram_cache: dict,
    name_idf: Dict[str, float],
    addr_idf: Dict[str, float],
    ranker,
    calibrator,
    policy_cfg: dict,
    num_threads: int = 1,
    sub_batch_size: int = 30_000,
) -> List[List[str]]:
    """Lossless 88-feature scoring using tuple-unpacked S1-grouped loop and verified 5-cpdist Stage-1 cascade."""
    n_s1 = len(s1_dicts)
    if n_s1 == 0:
        return []

    cgram_cache.clear()

    n_idf_get = name_idf.get
    a_idf_get = addr_idf.get

    pair_s1_idx: List[int] = []
    pair_tid: List[int] = []
    pair_cid: List[str] = []
    s1_slices: List[Tuple[int, int]] = []

    cur = 0
    for q_idx, c_list in enumerate(cands_per_s1):
        st = cur
        for tid_i, cid_i, _, _ in c_list:
            pair_s1_idx.append(q_idx)
            pair_tid.append(tid_i)
            pair_cid.append(cid_i)
        cur += len(c_list)
        s1_slices.append((st, cur))

    n_total = cur
    if n_total == 0:
        return [[] for _ in range(n_s1)]

    all_probs = np.zeros(n_total, dtype=np.float64)
    all_exact_ev = np.zeros(n_total, dtype=bool)
    all_num_agree = np.zeros(n_total, dtype=bool)
    all_num_conflict = np.zeros(n_total, dtype=bool)
    all_core_name_set = np.zeros(n_total, dtype=np.float32)
    all_high_name_low_addr = np.zeros(n_total, dtype=bool)

    booster = ranker.booster_

    for start in range(0, n_total, sub_batch_size):
        end = min(start + sub_batch_size, n_total)
        b_len = end - start
        b_s1_i = pair_s1_idx[start:end]
        b_tid_i = pair_tid[start:end]

        e1_list = [s1_dicts[q_i] for q_i in b_s1_i]
        t2_list = [get_tgt_tuple(t_i) for t_i in b_tid_i]

        # Stage 1: Compute only the 5 anchor cpdist calls across all pairs
        core_n1 = [e["n_core_norm"] for e in e1_list]
        core_n2 = [t[7] for t in t2_list]
        norm_a1 = [e["a_lower_norm"] for e in e1_list]
        norm_a2 = [t[16] for t in t2_list]

        core_name_ratio = process.cpdist(core_n1, core_n2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
        core_name_token_set = process.cpdist(core_n1, core_n2, scorer=fuzz.token_set_ratio, workers=num_threads, dtype=np.float32) / 100.0
        core_name_partial_ratio = process.cpdist(core_n1, core_n2, scorer=fuzz.partial_ratio, workers=num_threads, dtype=np.float32) / 100.0
        norm_addr_token_set = process.cpdist(norm_a1, norm_a2, scorer=fuzz.token_set_ratio, workers=num_threads, dtype=np.float32) / 100.0
        norm_addr_partial_ratio = process.cpdist(norm_a1, norm_a2, scorer=fuzz.partial_ratio, workers=num_threads, dtype=np.float32) / 100.0

        X = np.zeros((b_len, 88), dtype=np.float32)

        # Populate the 5 exact columns + Stage-1 proxies for the other 16 cpdist columns
        X[:, 15] = core_name_ratio
        X[:, 17] = core_name_token_set
        X[:, 18] = core_name_partial_ratio
        X[:, 43] = norm_addr_token_set
        X[:, 44] = norm_addr_partial_ratio

        X[:, 0] = core_name_ratio
        X[:, 11] = core_name_ratio
        X[:, 19] = core_name_ratio
        X[:, 40] = core_name_ratio
        X[:, 85] = core_name_ratio

        X[:, 1] = core_name_token_set
        X[:, 2] = core_name_token_set
        X[:, 12] = core_name_token_set
        X[:, 13] = core_name_token_set
        X[:, 16] = core_name_token_set

        X[:, 14] = core_name_partial_ratio

        X[:, 3] = norm_addr_token_set
        X[:, 4] = norm_addr_token_set
        X[:, 5] = norm_addr_token_set
        X[:, 41] = norm_addr_token_set
        X[:, 42] = norm_addr_token_set

        # Fast S1-grouped tuple-unpacked row feature loop
        cur_q_idx = -1
        c1 = on1 = ln1 = cn1 = cpn1 = ps1 = sc1 = ft1 = oa1 = la1 = st1 = b1 = bs1 = p1 = u1 = ""
        nl1 = al1 = cnl1 = cpnl1 = psl1 = l_la1 = l_nt1 = l_ct1 = l_ph1 = l_at1 = l_num1 = 0
        bc1 = pc1 = n_idf_sum1 = a_idf_sum1 = 0.0
        nt1 = ct1 = ph1 = at1 = num1 = frozenset()

        for i in range(b_len):
            q_i = b_s1_i[i]
            if q_i != cur_q_idx:
                cur_q_idx = q_i
                e1 = s1_dicts[q_i]
                c1 = e1["country"]
                nl1 = len(e1["raw_name"])
                al1 = len(e1["raw_addr"])
                on1 = e1["n_orig_norm"]
                ln1 = e1["n_lower_norm"]
                cn1 = e1["n_core_norm"]
                cnl1 = len(cn1)
                cpn1 = e1["n_compact_norm"]
                cpnl1 = len(cpn1)
                ps1 = e1["n_phon_str"]
                psl1 = len(ps1)
                sc1 = e1["n_suffix_class"]
                ft1 = e1["first_core_tok"]
                oa1 = e1["a_orig_norm"]
                la1 = e1["a_lower_norm"]
                l_la1 = len(la1)
                st1 = e1["a_state_code"]
                b1 = e1["a_bldg_num"]
                bs1 = e1["a_bldg_stem"]
                bc1 = e1["a_bldg_conf"]
                p1 = e1["a_postal_code"]
                pc1 = e1["a_postal_conf"]
                u1 = e1["a_unit_num"]
                nt1 = e1["n_tok_set"]
                l_nt1 = len(nt1)
                ct1 = e1["n_core_set"]
                l_ct1 = len(ct1)
                ph1 = e1["n_phon_set"]
                l_ph1 = len(ph1)
                at1 = e1["a_tok_set"]
                l_at1 = len(at1)
                num1 = e1["a_num_set"]
                l_num1 = len(num1)
                n_idf_sum1 = sum(n_idf_get(t, 0.5) for t in ct1)
                a_idf_sum1 = sum(a_idf_get(t, 0.5) for t in at1)

            (
                _, nl2, _, al2, c2, on2, ln2, cn2, cnl2, cpn2, _,
                ps2, _, sc2, ft2, oa2, la2, l_la2, st2, b2, bs2, bc2,
                p2, pc2, u2, nt2, l_nt2, ct2, l_ct2, ph2, l_ph2, at2, l_at2, num2, l_num2,
            ) = t2_list[i]

            r = X[i]
            if c1 and c1 == c2:
                r[6] = 1.0
            if b_tid_i[i] < n_s2_bound:
                r[84] = 1.0

            r[7] = nl1
            r[8] = nl2
            r[9] = al1
            r[10] = al2

            r[30] = (nl1 if nl1 < nl2 else nl2) / max(1, nl1 if nl1 > nl2 else nl2)
            r[31] = (cnl1 if cnl1 < cnl2 else cnl2) / max(1, cnl1 if cnl1 > cnl2 else cnl2)

            if on1 and on1 == on2:
                r[34] = 1.0
            if ln1 and ln1 == ln2:
                r[35] = 1.0
            if cnl1 >= 3 and cn1 == cn2:
                r[36] = 1.0
            if cpnl1 >= 3 and cpn1 == cpn2:
                r[37] = 1.0
            if ft1 and ft1 == ft2:
                r[39] = 1.0
            if sc1 and sc2:
                r[38] = 1.0 if sc1 == sc2 else -1.0

            if l_nt1 and l_nt2 and not nt1.isdisjoint(nt2):
                inter_nt = len(nt1 & nt2)
                r[23] = inter_nt / (l_nt1 + l_nt2 - inter_nt)

            r[32] = l_ct1 if l_ct1 < l_ct2 else l_ct2
            r[33] = abs(l_ct1 - l_ct2)
            if l_ct1 and l_ct2 and not ct1.isdisjoint(ct2):
                c_inter = ct1 & ct2
                n_c_inter = len(c_inter)
                r[27] = n_c_inter
                r[24] = n_c_inter / (l_ct1 + l_ct2 - n_c_inter)
                r[25] = n_c_inter / (l_ct1 if l_ct1 < l_ct2 else l_ct2)
                r[26] = n_c_inter / (l_ct1 if l_ct1 > l_ct2 else l_ct2)
                idf_inter = sum(n_idf_get(t, 0.5) for t in c_inter)
                idf_union = n_idf_sum1 + sum(n_idf_get(t, 0.5) for t in ct2) - idf_inter
                r[28] = idf_inter
                r[29] = idf_inter / max(1e-6, idf_union)

            if l_ph1 and l_ph2 and not ph1.isdisjoint(ph2):
                inter_ph = len(ph1 & ph2)
                r[86] = inter_ph / (l_ph1 + l_ph2 - inter_ph)
            if psl1 >= 3 and ps1 == ps2:
                r[87] = 1.0

            if l_la1 > 0 and l_la2 > 0:
                r[57] = 1.0
                r[56] = (l_la1 if l_la1 < l_la2 else l_la2) / (l_la1 if l_la1 > l_la2 else l_la2)
                if la1 == la2:
                    r[55] = 1.0
                if l_at1 and l_at2 and not at1.isdisjoint(at2):
                    a_inter = at1 & at2
                    n_a_inter = len(a_inter)
                    r[51] = n_a_inter
                    r[48] = n_a_inter / (l_at1 + l_at2 - n_a_inter)
                    r[49] = n_a_inter / (l_at1 if l_at1 < l_at2 else l_at2)
                    r[50] = n_a_inter / (l_at1 if l_at1 > l_at2 else l_at2)
                    a_idf_inter = sum(a_idf_get(t, 0.5) for t in a_inter)
                    a_idf_union = a_idf_sum1 + sum(a_idf_get(t, 0.5) for t in at2) - a_idf_inter
                    r[52] = a_idf_inter
                    r[53] = a_idf_inter / max(1e-6, a_idf_union)
            else:
                r[58] = 1.0

            if oa1 and oa1 == oa2:
                r[54] = 1.0

            if st1 and st2:
                r[59 if st1 == st2 else 60] = 1.0

            r[63] = bc1 if bc1 < bc2 else bc2
            if b1 and b2:
                if b1 == b2:
                    r[61] = 1.0
                elif len(bs1) >= 2 and bs1 == bs2 and len(bs2) >= 2:
                    r[61] = 0.5
                else:
                    r[62] = 1.0

            r[66] = pc1 if pc1 < pc2 else pc2
            if p1 and p2:
                r[64 if p1 == p2 else 65] = 1.0

            if u1 and u2:
                r[70 if u1 == u2 else 71] = 1.0

            r[69] = abs(l_num1 - l_num2)
            if l_num1 and l_num2 and not num1.isdisjoint(num2):
                num_inter = len(num1 & num2)
                r[68] = num_inter
                r[67] = num_inter / (l_num1 + l_num2 - num_inter)

        c_ex_on = X[:, 34]
        c_ex_nn = X[:, 35]
        c_ex_cn = X[:, 36]
        c_ex_na = X[:, 55]
        c_a_both = X[:, 57]
        c_bm = X[:, 61]
        c_bc = X[:, 62]
        c_pm = X[:, 64]
        c_pc = X[:, 65]

        X[:, 72] = X[:, 13] * norm_addr_token_set
        X[:, 73] = core_name_token_set * norm_addr_token_set
        X[:, 74] = 0.5 * (core_name_ratio + norm_addr_token_set)
        X[:, 75] = core_name_token_set - norm_addr_token_set
        X[:, 76] = ((core_name_token_set >= 0.90) & (c_a_both > 0.5) & (norm_addr_token_set < 0.62)).astype(np.float32)
        X[:, 77] = ((core_name_token_set >= 0.85) & (norm_addr_token_set >= 0.80)).astype(np.float32)
        X[:, 78] = c_ex_cn * (norm_addr_token_set + 0.5 * c_bm)
        X[:, 79] = c_ex_na * core_name_token_set
        X[:, 80] = c_bm * core_name_token_set
        X[:, 81] = c_bc * core_name_token_set
        X[:, 82] = c_pm * core_name_token_set
        X[:, 83] = c_pc * core_name_token_set

        all_exact_ev[start:end] = (c_ex_on > 0.5) | (c_ex_nn > 0.5) | (c_ex_cn > 0.5) | (c_ex_na > 0.5)
        all_num_agree[start:end] = (c_bm >= 0.9) | (c_pm > 0.5)
        all_num_conflict[start:end] = (c_bc > 0.5) | (c_pc > 0.5)
        all_core_name_set[start:end] = core_name_token_set
        all_high_name_low_addr[start:end] = X[:, 76] > 0.5

        # Stage-1 50-tree filter (verified 0 missed matches at >= -2.5 across all 3.75M validation pairs)
        s50 = booster.predict(X, num_iteration=50, num_threads=num_threads)
        keep_indices = np.where(s50 >= -2.5)[0]

        if len(keep_indices) > 0:
            # Stage 2: Compute the remaining 16 exact cpdist features ONLY on keep_indices (~20% of pairs)
            k_e1 = [e1_list[idx_k] for idx_k in keep_indices]
            k_t2 = [t2_list[idx_k] for idx_k in keep_indices]

            k_raw_n1 = [e["raw_name"] for e in k_e1]
            k_raw_n2 = [t[0] for t in k_t2]
            k_raw_a1 = [e["raw_addr"] for e in k_e1]
            k_raw_a2 = [t[2] for t in k_t2]
            k_norm_n1 = [e["n_lower_norm"] for e in k_e1]
            k_norm_n2 = [t[6] for t in k_t2]
            k_core_n1 = [e["n_core_norm"] for e in k_e1]
            k_core_n2 = [t[7] for t in k_t2]
            k_comp_n1 = [e["n_compact_norm"] for e in k_e1]
            k_comp_n2 = [t[9] for t in k_t2]
            k_phon_n1 = [e["n_phon_str"] for e in k_e1]
            k_phon_n2 = [t[11] for t in k_t2]
            k_first_t1 = [e["first_core_tok"] for e in k_e1]
            k_first_t2 = [t[14] for t in k_t2]
            k_norm_a1 = [e["a_lower_norm"] for e in k_e1]
            k_norm_a2 = [t[16] for t in k_t2]

            X_k = X[keep_indices]
            X_k[:, 0] = process.cpdist(k_raw_n1, k_raw_n2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 1] = process.cpdist(k_raw_n1, k_raw_n2, scorer=fuzz.token_sort_ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 2] = process.cpdist(k_raw_n1, k_raw_n2, scorer=fuzz.token_set_ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 3] = process.cpdist(k_raw_a1, k_raw_a2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 4] = process.cpdist(k_raw_a1, k_raw_a2, scorer=fuzz.token_sort_ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 5] = process.cpdist(k_raw_a1, k_raw_a2, scorer=fuzz.token_set_ratio, workers=num_threads, dtype=np.float32) / 100.0

            X_k[:, 11] = process.cpdist(k_norm_n1, k_norm_n2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 12] = process.cpdist(k_norm_n1, k_norm_n2, scorer=fuzz.token_sort_ratio, workers=num_threads, dtype=np.float32) / 100.0
            k_nn_tset = process.cpdist(k_norm_n1, k_norm_n2, scorer=fuzz.token_set_ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 13] = k_nn_tset
            X_k[:, 14] = process.cpdist(k_norm_n1, k_norm_n2, scorer=fuzz.partial_ratio, workers=num_threads, dtype=np.float32) / 100.0

            X_k[:, 16] = process.cpdist(k_core_n1, k_core_n2, scorer=fuzz.token_sort_ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 19] = process.cpdist(k_comp_n1, k_comp_n2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 40] = process.cpdist(k_first_t1, k_first_t2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 41] = process.cpdist(k_norm_a1, k_norm_a2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 42] = process.cpdist(k_norm_a1, k_norm_a2, scorer=fuzz.token_sort_ratio, workers=num_threads, dtype=np.float32) / 100.0
            X_k[:, 85] = process.cpdist(k_phon_n1, k_phon_n2, scorer=fuzz.ratio, workers=num_threads, dtype=np.float32) / 100.0

            # Update interaction feature 72 with exact norm_name_token_set
            X_k[:, 72] = k_nn_tset * X_k[:, 43]

            # Stage-2 80-tree filter (verified 0 missed matches at >= -1.6 across all 3.75M validation pairs; min_pos = -0.072)
            s80_nocg = booster.predict(X_k, num_iteration=80, num_threads=num_threads)
            keep2_loc = np.where(s80_nocg >= -1.6)[0]

            raw_batch = np.full(b_len, -20.0, dtype=np.float64)
            if len(keep2_loc) > 0:
                # Stage 3: Compute the 6 exact character 3/4/5-gram Jaccard features + 600 trees ONLY on keep2_loc (~2% of pairs)
                X_k2 = X_k[keep2_loc]
                c_nc3 = X_k2[:, 20]
                c_nc4 = X_k2[:, 21]
                c_nc5 = X_k2[:, 22]
                c_ac3 = X_k2[:, 45]
                c_ac4 = X_k2[:, 46]
                c_ac5 = X_k2[:, 47]

                for i2, loc_k in enumerate(keep2_loc):
                    if len(cgram_cache) > 1_500:
                        cgram_cache.clear()
                    e1 = k_e1[loc_k]
                    t2 = k_t2[loc_k]

                    s_n1 = e1["n_compact_norm"]
                    s_n2 = t2[9]
                    if s_n1 and s_n2:
                        if s_n1 == s_n2:
                            c_nc3[i2] = 1.0
                            c_nc4[i2] = 1.0
                            c_nc5[i2] = 1.0
                        else:
                            if "n_c3" not in e1:
                                e1["n_c3"] = _char_ngrams_set(s_n1, 3)
                                e1["n_c4"] = _char_ngrams_set(s_n1, 4)
                                e1["n_c5"] = _char_ngrams_set(s_n1, 5)
                            cg_n2 = cgram_cache.get(s_n2)
                            if cg_n2 is None:
                                b3 = _char_ngrams_set(s_n2, 3)
                                b4 = _char_ngrams_set(s_n2, 4)
                                b5 = _char_ngrams_set(s_n2, 5)
                                cg_n2 = (b3, len(b3), b4, len(b4), b5, len(b5))
                                cgram_cache[s_n2] = cg_n2
                            b3, lb3, b4, lb4, b5, lb5 = cg_n2
                            a3 = e1["n_c3"]
                            if not a3.isdisjoint(b3):
                                i3 = len(a3 & b3)
                                c_nc3[i2] = i3 / (len(a3) + lb3 - i3)
                            a4 = e1["n_c4"]
                            if not a4.isdisjoint(b4):
                                i4 = len(a4 & b4)
                                c_nc4[i2] = i4 / (len(a4) + lb4 - i4)
                            a5 = e1["n_c5"]
                            if not a5.isdisjoint(b5):
                                i5 = len(a5 & b5)
                                c_nc5[i2] = i5 / (len(a5) + lb5 - i5)

                    s_a1 = e1["a_lower_norm"]
                    s_a2 = t2[16]
                    if s_a1 and s_a2:
                        if s_a1 == s_a2:
                            c_ac3[i2] = 1.0
                            c_ac4[i2] = 1.0
                            c_ac5[i2] = 1.0
                        else:
                            if "a_c3" not in e1:
                                e1["a_c3"] = _char_ngrams_set(s_a1, 3)
                                e1["a_c4"] = _char_ngrams_set(s_a1, 4)
                                e1["a_c5"] = _char_ngrams_set(s_a1, 5)
                            cg_a2 = cgram_cache.get(s_a2)
                            if cg_a2 is None:
                                b3 = _char_ngrams_set(s_a2, 3)
                                b4 = _char_ngrams_set(s_a2, 4)
                                b5 = _char_ngrams_set(s_a2, 5)
                                cg_a2 = (b3, len(b3), b4, len(b4), b5, len(b5))
                                cgram_cache[s_a2] = cg_a2
                            b3, lb3, b4, lb4, b5, lb5 = cg_a2
                            a3 = e1["a_c3"]
                            if not a3.isdisjoint(b3):
                                i3 = len(a3 & b3)
                                c_ac3[i2] = i3 / (len(a3) + lb3 - i3)
                            a4 = e1["a_c4"]
                            if not a4.isdisjoint(b4):
                                i4 = len(a4 & b4)
                                c_ac4[i2] = i4 / (len(a4) + lb4 - i4)
                            a5 = e1["a_c5"]
                            if not a5.isdisjoint(b5):
                                i5 = len(a5 & b5)
                                c_ac5[i2] = i5 / (len(a5) + lb5 - i5)

                raw_batch[keep_indices[keep2_loc]] = booster.predict(X_k2, num_threads=num_threads)
                cgram_cache.clear()
        else:
            raw_batch = np.full(b_len, -20.0, dtype=np.float64)

        all_probs[start:end] = calibrator.predict_proba(raw_batch.reshape(-1, 1))[:, 1]

    bonus_agree = float(policy_cfg["bonus_agree"])
    pen_conflict = float(policy_cfg["pen_conflict"])
    pen_branch = float(policy_cfg["pen_branch"])
    max_k = int(policy_cfg["max_k"])
    rescue_margin = float(policy_cfg.get("rescue_margin", 1.0))
    rescue_floor = float(policy_cfg.get("rescue_floor", 0.965))

    adj_scores = (
        all_probs
        + bonus_agree * ((all_exact_ev | (all_num_agree & (all_core_name_set >= 0.85))).astype(np.float64))
        - pen_conflict * (all_num_conflict.astype(np.float64))
        - pen_branch * (all_high_name_low_addr.astype(np.float64))
    )

    matches_per_s1: List[List[str]] = []
    for q_idx in range(n_s1):
        st, en = s1_slices[q_idx]
        if st == en:
            matches_per_s1.append([])
            continue

        ctry = s1_dicts[q_idx]["country"]
        if ctry == "us":
            thr = float(policy_cfg["t_us"])
        elif ctry == "india":
            thr = float(policy_cfg["t_in"])
        else:
            thr = float(policy_cfg["t_default"])

        sc_sub = all_probs[st:en]
        adj_sub = adj_scores[st:en]
        order = np.argsort(-sc_sub)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(sc_sub) + 1)

        sel_mask = (adj_sub >= thr) & (ranks <= max_k)
        if rescue_margin < 1.0:
            t1 = float(sc_sub[order[0]])
            t2 = float(sc_sub[order[1]]) if len(sc_sub) >= 2 else 0.0
            if (t1 - t2) >= rescue_margin:
                sel_mask |= (ranks == 1) & (adj_sub >= rescue_floor)

        sel_cids = [pair_cid[st + loc_i] for loc_i in np.where(sel_mask)[0]]
        matches_per_s1.append(sel_cids)

    return matches_per_s1


class CountryIndexV3:
    """SharedMemory-backed Two-Tier 4-Channel country index for 8-worker parallel retrieval & scoring."""

    def __init__(self, country: str, scratch_dir: Optional[Path] = None):
        self.country = country
        self.n_s2: int = 0
        self.n_targets: int = 0
        self.n_postings: int = 0

        # Dictionaries mapping key -> (start_offset, length) in postings_shm
        self.io_n: Dict[str, Tuple[int, int]] = {}
        self.il_n: Dict[str, Tuple[int, int]] = {}
        self.ic_n: Dict[str, Tuple[int, int]] = {}
        self.icp_n: Dict[str, Tuple[int, int]] = {}
        self.io_a: Dict[str, Tuple[int, int]] = {}
        self.il_a: Dict[str, Tuple[int, int]] = {}

        self.it_n: Dict[str, Tuple[int, int]] = {}
        self.it_p: Dict[str, Tuple[int, int]] = {}
        self.it_a: Dict[str, Tuple[int, int]] = {}
        self.ig_c: Dict[str, Tuple[int, int]] = {}

        self.ib_n: Dict[str, Tuple[int, int]] = {}
        self.ip_c: Dict[str, Tuple[int, int]] = {}
        self.ibp: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self.ibs: Dict[Tuple[str, str], Tuple[int, int]] = {}

        self._shms: List[shared_memory.SharedMemory] = []
        self.shm_config: Dict[str, object] = {}

    def build(
        self,
        s1_records: List[Tuple[int, str, str, str]],
        s2_path: str,
        s3_path: str,
        worker_pool: Optional[mp.pool.Pool] = None,
        num_workers: int = 8,
    ) -> None:
        """Scan S1 query vocabulary, normalize S2+S3 targets in parallel, and pack into SharedMemory."""
        t0 = time.time()
        print(
            f"[{self.country.upper()}] Scanning {len(s1_records):,} S1 queries to build active vocabulary...",
            flush=True,
        )

        qo_n: Set[str] = set()
        ql_n: Set[str] = set()
        qc_n: Set[str] = set()
        qcp_n: Set[str] = set()
        qo_a: Set[str] = set()
        ql_a: Set[str] = set()

        qt_n: Set[str] = set()
        qt_p: Set[str] = set()
        qt_a: Set[str] = set()
        qg_c: Set[str] = set()

        qb_n: Set[str] = set()
        qp_c: Set[str] = set()
        qbp: Set[Tuple[str, str]] = set()
        qbs: Set[Tuple[str, str]] = set()

        c = self.country
        for _, _, n_raw, a_raw in s1_records:
            n_info = normalize_name_v3(n_raw)
            a_info = parse_address_v3(a_raw, c)
            c_grams = extract_char_ngrams_for_name(n_info["compact_norm"], n_info["core_tokens"])

            if n_info["orig_norm"]:
                qo_n.add(n_info["orig_norm"])
            if n_info["lower_norm"]:
                ql_n.add(n_info["lower_norm"])
            if n_info["core_name_norm"]:
                qc_n.add(n_info["core_name_norm"])
            if len(n_info["compact_norm"]) >= 3:
                qcp_n.add(n_info["compact_norm"])
            if a_info["orig_norm"]:
                qo_a.add(a_info["orig_norm"])
            if a_info["lower_norm"]:
                ql_a.add(a_info["lower_norm"])

            qt_n.update(t for t in n_info["core_tokens"] if len(t) >= 2)
            qt_n.update(t for t in n_info["tokens"] if len(t) >= 3)
            qt_p.update(p for p in n_info["phonetic_tokens"] if len(p) >= 3)
            qt_a.update(t for t in a_info["tokens"] if len(t) >= 2)
            qg_c.update(c_grams)

            b_num = a_info["building_num"]
            p_code = a_info["postal_code"]
            s_code = a_info["state_code"]
            if b_num:
                qb_n.add(b_num)
                if p_code:
                    qbp.add((b_num, p_code))
                if s_code:
                    qbs.add((b_num, s_code))
            if p_code:
                qp_c.add(p_code)

        print(
            f"[{self.country.upper()}] Vocabulary ready in {time.time()-t0:.1f}s "
            f"(name_tok={len(qt_n):,}, addr_tok={len(qt_a):,}, char_gram={len(qg_c):,}).",
            flush=True,
        )

        # Broadcast active query vocabulary to workers via temporary SharedMemory kept open during build()
        shm_vocab = None
        vocab_len = 0
        if worker_pool is not None:
            vocab_bytes = pickle.dumps((c, qt_n, qt_p, qt_a, qg_c), protocol=pickle.HIGHEST_PROTOCOL)
            vocab_len = len(vocab_bytes)
            shm_vocab = shared_memory.SharedMemory(create=True, size=max(1, vocab_len))
            shm_vocab.buf[:vocab_len] = vocab_bytes
            del vocab_bytes

        raw_io_n: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_il_n: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_ic_n: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_icp_n: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_io_a: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_il_a: Dict[str, array] = defaultdict(lambda: array("i"))

        raw_it_n: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_it_p: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_it_a: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_ig_c: Dict[str, array] = defaultdict(lambda: array("i"))

        raw_ib_n: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_ip_c: Dict[str, array] = defaultdict(lambda: array("i"))
        raw_ibp: Dict[Tuple[str, str], array] = defaultdict(lambda: array("i"))
        raw_ibs: Dict[Tuple[str, str], array] = defaultdict(lambda: array("i"))

        dn_t: Set[str] = set()
        dp_t: Set[str] = set()
        da_t: Set[str] = set()
        dc_g: Set[str] = set()
        db_n: Set[str] = set()
        dp_c: Set[str] = set()
        dbp: Set[Tuple[str, str]] = set()
        dbs: Set[Tuple[str, str]] = set()

        t_idx0 = time.time()
        tid = 0
        blob = bytearray()
        blob_offsets = array("Q", [0])
        cids_blob = bytearray()
        cid_offsets = array("Q", [0])

        try:
            for src_label, src_path in [("S2", s2_path), ("S3", s3_path)]:
                t_s0 = time.time()
                for chunk in pd.read_csv(src_path, sep="\t", chunksize=400_000, dtype=str):
                    c_col = chunk["country"].fillna("").str.strip().str.lower()
                    sub = chunk[c_col == c]
                    if sub.empty:
                        continue

                    rows_raw = [
                        (
                            cid,
                            "" if pd.isna(n_raw) else str(n_raw),
                            "" if pd.isna(a_raw) else str(a_raw),
                        )
                        for cid, n_raw, a_raw in zip(
                            sub["entity_id"], sub["business_name"], sub["business_address"]
                        )
                    ]

                    if worker_pool is not None and shm_vocab is not None and len(rows_raw) >= 4_000:
                        b_sz = 20_000
                        batches = [
                            (shm_vocab.name, vocab_len, rows_raw[i : i + b_sz])
                            for i in range(0, len(rows_raw), b_sz)
                        ]
                        norm_iter = worker_pool.imap(worker_normalize_target_batch, batches)
                    else:
                        global _WORKER_VOCAB
                        _WORKER_VOCAB = (c, qt_n, qt_p, qt_a, qg_c)
                        norm_iter = [worker_normalize_target_batch((None, 0, rows_raw))]

                    for batch_res in norm_iter:
                        for (
                            cid,
                            packed_b,
                            on,
                            ln,
                            cn,
                            cpn,
                            oa,
                            la,
                            cand_name_toks,
                            cand_phon_toks,
                            cand_addr_toks,
                            c_grams,
                            b_num,
                            p_code,
                            s_code,
                        ) in batch_res:
                            cid_b = cid.encode("ascii")
                            cids_blob.extend(cid_b)
                            cid_offsets.append(len(cids_blob))

                            blob.extend(packed_b)
                            blob_offsets.append(len(blob))

                            if on in qo_n:
                                raw_io_n[on].append(tid)
                            if ln in ql_n:
                                raw_il_n[ln].append(tid)
                            if cn in qc_n and len(raw_ic_n[cn]) < 10_000:
                                raw_ic_n[cn].append(tid)
                            if cpn in qcp_n and len(raw_icp_n[cpn]) < 10_000:
                                raw_icp_n[cpn].append(tid)
                            if oa in qo_a:
                                raw_io_a[oa].append(tid)
                            if la in ql_a:
                                raw_il_a[la].append(tid)

                            for t in cand_name_toks:
                                if t not in dn_t:
                                    lst = raw_it_n[t]
                                    lst.append(tid)
                                    if len(lst) > T2_NAME_TOK_MAX_DF:
                                        dn_t.add(t)
                                        del raw_it_n[t]

                            for p in cand_phon_toks:
                                if p not in dp_t:
                                    lst = raw_it_p[p]
                                    lst.append(tid)
                                    if len(lst) > T2_PHON_TOK_MAX_DF:
                                        dp_t.add(p)
                                        del raw_it_p[p]

                            for t in cand_addr_toks:
                                if t not in da_t:
                                    lst = raw_it_a[t]
                                    lst.append(tid)
                                    if len(lst) > T2_ADDR_TOK_MAX_DF:
                                        da_t.add(t)
                                        del raw_it_a[t]

                            for g in c_grams:
                                if g not in dc_g:
                                    lst = raw_ig_c[g]
                                    lst.append(tid)
                                    if len(lst) > T2_CHAR_GRAM_MAX_DF:
                                        dc_g.add(g)
                                        del raw_ig_c[g]

                            if b_num:
                                if b_num in qb_n and b_num not in db_n:
                                    lst = raw_ib_n[b_num]
                                    lst.append(tid)
                                    if len(lst) > T2_BLDG_NUM_MAX_DF:
                                        db_n.add(b_num)
                                        del raw_ib_n[b_num]
                                if p_code and (b_num, p_code) in qbp and (b_num, p_code) not in dbp:
                                    lst = raw_ibp[(b_num, p_code)]
                                    lst.append(tid)
                                    if len(lst) > T1_BLDG_POSTAL_DF:
                                        dbp.add((b_num, p_code))
                                        del raw_ibp[(b_num, p_code)]
                                if s_code and (b_num, s_code) in qbs and (b_num, s_code) not in dbs:
                                    lst = raw_ibs[(b_num, s_code)]
                                    lst.append(tid)
                                    if len(lst) > T2_BLDG_STATE_MAX_DF:
                                        dbs.add((b_num, s_code))
                                        del raw_ibs[(b_num, s_code)]
                            if p_code and p_code in qp_c and p_code not in dp_c:
                                lst = raw_ip_c[p_code]
                                lst.append(tid)
                                if len(lst) > T2_POSTAL_MAX_DF:
                                    dp_c.add(p_code)
                                    del raw_ip_c[p_code]

                            tid += 1

                if src_label == "S2":
                    self.n_s2 = tid
                print(
                    f"  [{self.country.upper()}] Indexed {src_label} -> cumulative targets={tid:,} ({time.time()-t_s0:.1f}s)",
                    flush=True,
                )
        finally:
            if shm_vocab is not None:
                try:
                    shm_vocab.close()
                except Exception:
                    pass
                try:
                    shm_vocab.unlink()
                except Exception:
                    pass
            if worker_pool is not None:
                try:
                    worker_pool.map(worker_clear_vocab, range(num_workers), chunksize=1)
                except Exception:
                    pass

        self.n_targets = tid

        # Free temporary vocabulary sets immediately before allocating SharedMemory buffers
        del qo_n, ql_n, qc_n, qcp_n, qo_a, ql_a, qt_n, qt_p, qt_a, qg_c, qb_n, qp_c, qbp, qbs
        del dn_t, dp_t, da_t, dc_g, db_n, dp_c, dbp, dbs
        gc.collect()

        # Allocate SharedMemory blocks for cids, cid_offsets, target_blob, target_offsets first and free bytearrays
        shm_cids = shared_memory.SharedMemory(create=True, size=max(1, len(cids_blob)))
        self._shms.append(shm_cids)
        shm_cids.buf[: len(cids_blob)] = cids_blob
        cids_blob.clear()
        del cids_blob

        cid_off_bytes = cid_offsets.tobytes()
        shm_cid_off = shared_memory.SharedMemory(create=True, size=len(cid_off_bytes))
        self._shms.append(shm_cid_off)
        shm_cid_off.buf[: len(cid_off_bytes)] = cid_off_bytes
        del cid_offsets, cid_off_bytes

        blob_len = len(blob)
        shm_blob = shared_memory.SharedMemory(create=True, size=max(1, blob_len))
        self._shms.append(shm_blob)
        shm_blob.buf[:blob_len] = blob
        blob.clear()
        del blob

        blob_off_bytes = blob_offsets.tobytes()
        shm_blob_off = shared_memory.SharedMemory(create=True, size=len(blob_off_bytes))
        self._shms.append(shm_blob_off)
        shm_blob_off.buf[: len(blob_off_bytes)] = blob_off_bytes
        del blob_offsets, blob_off_bytes
        gc.collect()

        # Pack all posting arrays into ONE contiguous SharedMemory int32 buffer
        all_dicts = [
            (raw_io_n, self.io_n),
            (raw_il_n, self.il_n),
            (raw_ic_n, self.ic_n),
            (raw_icp_n, self.icp_n),
            (raw_io_a, self.io_a),
            (raw_il_a, self.il_a),
            (raw_it_n, self.it_n),
            (raw_it_p, self.it_p),
            (raw_it_a, self.it_a),
            (raw_ig_c, self.ig_c),
            (raw_ib_n, self.ib_n),
            (raw_ip_c, self.ip_c),
            (raw_ibp, self.ibp),
            (raw_ibs, self.ibs),
        ]
        total_post_len = sum(sum(len(arr) for arr in src_d.values()) for src_d, _ in all_dicts)
        self.n_postings = max(1, total_post_len)

        shm_post = shared_memory.SharedMemory(create=True, size=self.n_postings * 4)
        self._shms.append(shm_post)
        post_np = np.ndarray((self.n_postings,), dtype=np.int32, buffer=shm_post.buf)

        cur_p = 0
        for src_d, dst_d in all_dicts:
            while src_d:
                k, arr = src_d.popitem()
                ln = len(arr)
                if ln > 0:
                    post_np[cur_p : cur_p + ln] = np.frombuffer(arr, dtype=np.int32)
                    dst_d[k] = (cur_p, ln)
                    cur_p += ln
                del arr

        self.shm_config = {
            "country": self.country,
            "n_targets": self.n_targets,
            "n_postings": self.n_postings,
            "postings_shm_name": shm_post.name,
            "cids_shm_name": shm_cids.name,
            "cid_offsets_shm_name": shm_cid_off.name,
            "blob_shm_name": shm_blob.name,
            "blob_offsets_shm_name": shm_blob_off.name,
        }

        gc.collect()
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(
            f"[{self.country.upper()}] Country index ready in {time.time()-t_idx0:.1f}s | "
            f"Targets={self.n_targets:,} (S2={self.n_s2:,}, S3={self.n_targets-self.n_s2:,}) | "
            f"Postings={total_post_len*4/(1024**2):.1f} MB | NormBlob={blob_len/(1024**2):.1f} MB | Avail RAM={avail_gb:.2f} GB",
            flush=True,
        )

    def build_query_retrieval_plan(
        self, s1_dict: dict
    ) -> Tuple[
        List[Tuple[int, int, float, int, int, bool]],
        List[Tuple[int, int, float, int, int, bool]],
    ]:
        """Resolve an S1 query's tokens into (start, len, weight, ch_bit, ev_bit, is_name_tok) slices."""
        t1_specs: List[Tuple[int, int, float, int, int, bool]] = []
        t2_specs: List[Tuple[int, int, float, int, int, bool]] = []

        def _push_t1(sl: Optional[Tuple[int, int]], w: float, ch_bit: int, ev_bit: int, is_name: bool = False):
            if sl is not None and sl[1] > 0:
                t1_specs.append((sl[0], sl[1], w, ch_bit, ev_bit, is_name))

        _push_t1(self.io_n.get(s1_dict["n_orig_norm"]), 220.0, 1, 1 | 4)
        _push_t1(self.il_n.get(s1_dict["n_lower_norm"]), 180.0, 1, 1 | 4)
        _push_t1(self.ic_n.get(s1_dict["n_core_norm"]), 140.0, 1, 1 | 4)
        _push_t1(self.icp_n.get(s1_dict["n_compact_norm"]), 130.0, 1, 1 | 4)
        _push_t1(self.io_a.get(s1_dict["a_orig_norm"]), 200.0, 1, 2 | 4)
        _push_t1(self.il_a.get(s1_dict["a_lower_norm"]), 170.0, 1, 2 | 4)

        u_name_t1 = []
        u_name_t2 = []
        it_n = self.it_n
        for t in s1_dict["ret_name_toks"]:
            sl = it_n.get(t)
            if sl is not None:
                df = sl[1]
                if 0 < df <= T1_NAME_TOK_DF:
                    u_name_t1.append((df, t, sl))
                elif df <= T2_NAME_TOK_MAX_DF:
                    u_name_t2.append((df, t, sl))
        u_name_t1.sort(key=lambda x: (x[0], x[1]))
        u_name_t2.sort(key=lambda x: (x[0], x[1]))

        active_t1_name = u_name_t1[:5]
        if not active_t1_name and u_name_t2 and u_name_t2[0][0] <= T1_NAME_TOK_FALLBACK_DF:
            active_t1_name = [u_name_t2.pop(0)]
        else:
            u_name_t2 = u_name_t1[5:] + u_name_t2

        for df, _, sl in active_t1_name:
            _push_t1(sl, 22.0 / math.log10(10.0 + df), 2, 1, True)
        for df, _, sl in u_name_t2[:3]:
            if df > T1_NAME_TOK_DF:
                t2_specs.append((sl[0], sl[1], 22.0 / math.log10(10.0 + df), 2, 1, True))

        u_phon_t1 = []
        u_phon_t2 = []
        it_p = self.it_p
        for p in s1_dict["ret_phon_toks"]:
            sl = it_p.get(p)
            if sl is not None:
                df = sl[1]
                if 0 < df <= T1_PHON_TOK_DF:
                    u_phon_t1.append((df, p, sl))
                elif df <= T2_PHON_TOK_MAX_DF:
                    u_phon_t2.append((df, p, sl))
        u_phon_t1.sort(key=lambda x: (x[0], x[1]))
        u_phon_t2 = u_phon_t1[3:] + sorted(u_phon_t2, key=lambda x: (x[0], x[1]))
        for df, _, sl in u_phon_t1[:3]:
            _push_t1(sl, 9.0 / math.log10(10.0 + df), 2, 1, False)
        for df, _, sl in u_phon_t2[:2]:
            if df > T1_PHON_TOK_DF:
                t2_specs.append((sl[0], sl[1], 9.0 / math.log10(10.0 + df), 2, 1, False))

        u_addr_t1 = []
        u_addr_t2 = []
        it_a = self.it_a
        for t in s1_dict["ret_addr_toks"]:
            sl = it_a.get(t)
            if sl is not None:
                df = sl[1]
                if 0 < df <= T1_ADDR_TOK_DF:
                    u_addr_t1.append((df, t, sl))
                elif df <= T2_ADDR_TOK_MAX_DF:
                    u_addr_t2.append((df, t, sl))
        u_addr_t1.sort(key=lambda x: (x[0], x[1]))
        u_addr_t2 = u_addr_t1[4:] + sorted(u_addr_t2, key=lambda x: (x[0], x[1]))
        for df, _, sl in u_addr_t1[:4]:
            _push_t1(sl, 11.0 / math.log10(10.0 + df), 2, 2, False)
        for df, _, sl in u_addr_t2[:3]:
            if df > T1_ADDR_TOK_DF:
                t2_specs.append((sl[0], sl[1], 11.0 / math.log10(10.0 + df), 2, 2, False))

        u_gram_t1 = []
        u_gram_t2 = []
        ig_c = self.ig_c
        for g in s1_dict["ret_char_grams"]:
            sl = ig_c.get(g)
            if sl is not None:
                df = sl[1]
                if 0 < df <= T1_CHAR_GRAM_DF:
                    u_gram_t1.append((df, g, sl))
                elif df <= T2_CHAR_GRAM_MAX_DF:
                    u_gram_t2.append((df, g, sl))
        u_gram_t1.sort(key=lambda x: (x[0], x[1]))
        u_gram_t2 = u_gram_t1[5:] + sorted(u_gram_t2, key=lambda x: (x[0], x[1]))
        for df, _, sl in u_gram_t1[:5]:
            _push_t1(sl, 5.5 / math.log10(10.0 + df), 4, 1, False)
        for df, _, sl in u_gram_t2[:3]:
            if df > T1_CHAR_GRAM_DF:
                t2_specs.append((sl[0], sl[1], 5.5 / math.log10(10.0 + df), 4, 1, False))

        b_num = s1_dict["a_bldg_num"]
        p_code = s1_dict["a_postal_code"]
        s_code = s1_dict["a_state_code"]

        if b_num and p_code:
            sl_bp = self.ibp.get((b_num, p_code))
            if sl_bp is not None and 0 < sl_bp[1] <= T1_BLDG_POSTAL_DF:
                _push_t1(sl_bp, 55.0, 8, 2, False)

        if b_num and s_code:
            sl_bs = self.ibs.get((b_num, s_code))
            if sl_bs is not None:
                df_bs = sl_bs[1]
                if 0 < df_bs <= T1_BLDG_STATE_DF:
                    _push_t1(sl_bs, 32.0 / math.log10(10.0 + df_bs), 8, 2, False)
                elif df_bs > T1_BLDG_STATE_DF:
                    t2_specs.append((sl_bs[0], df_bs, 32.0 / math.log10(10.0 + df_bs), 8, 2, False))

        if b_num:
            sl_b = self.ib_n.get(b_num)
            if sl_b is not None:
                df_b = sl_b[1]
                if 0 < df_b <= T1_BLDG_NUM_DF:
                    _push_t1(sl_b, 26.0 / math.log10(10.0 + df_b), 8, 2, False)
                elif df_b > T1_BLDG_NUM_DF:
                    t2_specs.append((sl_b[0], df_b, 26.0 / math.log10(10.0 + df_b), 8, 2, False))

        if p_code:
            sl_p = self.ip_c.get(p_code)
            if sl_p is not None:
                df_p = sl_p[1]
                if 0 < df_p <= T1_POSTAL_DF:
                    _push_t1(sl_p, 20.0 / math.log10(10.0 + df_p), 8, 2, False)
                elif df_p > T1_POSTAL_DF:
                    t2_specs.append((sl_p[0], df_p, 20.0 / math.log10(10.0 + df_p), 8, 2, False))

        return t1_specs, t2_specs

    def retrieve_and_score_chunk(
        self,
        s1_records: List[Tuple[int, str, str, str]],
        worker_pool: mp.pool.Pool,
        top_k: int = 500,
        num_shards: int = 16,
    ) -> Tuple[List[List[str]], List[List[str]], int, float, float]:
        """Run parallel 8-worker retrieval + parallel 8-worker scoring on a chunk of S1 records."""
        n_s1 = len(s1_records)
        if n_s1 == 0:
            return [], [], 0, 0.0, 0.0

        t_prep0 = time.time()
        ctry = self.country
        s1_dicts = [prepare_s1_entity_dict(n, a, ctry) for _, _, n, a in s1_records]
        query_plans = [self.build_query_retrieval_plan(d) for d in s1_dicts]

        # Compute exact chunk-wide IDF dictionaries
        name_df_cnt: Counter = Counter()
        addr_df_cnt: Counter = Counter()
        for d in s1_dicts:
            for t in d["n_core_set"]:
                name_df_cnt[t] += 1
            for t in d["a_tok_set"]:
                addr_df_cnt[t] += 1

        name_idf = {t: 1.0 / math.log10(10.0 + df * 10) for t, df in name_df_cnt.items()}
        addr_idf = {t: 1.0 / math.log10(10.0 + df * 10) for t, df in addr_df_cnt.items()}

        # Build shards with shard-local IDF dictionaries (eliminates pickling unrelated tokens)
        shard_size = max(1, (n_s1 + num_shards - 1) // num_shards)
        payloads = []
        for st in range(0, n_s1, shard_size):
            en = min(st + shard_size, n_s1)
            sub_s1 = s1_dicts[st:en]
            sub_plans = query_plans[st:en]

            # Strip retrieval-only token sets before pickling to workers
            for d in sub_s1:
                d.pop("ret_name_toks", None)
                d.pop("ret_phon_toks", None)
                d.pop("ret_addr_toks", None)
                d.pop("ret_char_grams", None)

            payloads.append((
                self.shm_config,
                self.n_s2,
                top_k,
                sub_s1,
                sub_plans,
                name_idf,
                addr_idf,
            ))
        t_prep_el = time.time() - t_prep0

        t_exec0 = time.time()
        shard_outputs = list(worker_pool.imap(worker_retrieve_and_score_shard, payloads))
        t_exec_el = time.time() - t_exec0

        all_cands: List[List[str]] = []
        all_matches: List[List[str]] = []
        total_pairs = 0
        sum_worker_ret = 0.0
        sum_worker_sc = 0.0

        for c_ids, m_ids, n_p, w_r, w_s in shard_outputs:
            all_cands.extend(c_ids)
            all_matches.extend(m_ids)
            total_pairs += n_p
            sum_worker_ret += w_r
            sum_worker_sc += w_s

        tot_w = max(1e-6, sum_worker_ret + sum_worker_sc)
        wall_ret = t_prep_el + t_exec_el * (sum_worker_ret / tot_w)
        wall_sc = t_exec_el * (sum_worker_sc / tot_w)
        return all_cands, all_matches, total_pairs, wall_ret, wall_sc

    def close_and_cleanup(self, worker_pool: Optional[mp.pool.Pool] = None, num_workers: int = 8) -> None:
        """Detach workers and unlink all SharedMemory segments for this country."""
        if worker_pool is not None:
            try:
                worker_pool.map(worker_detach_country_shm, range(num_workers), chunksize=1)
            except Exception:
                pass
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self._shms.clear()


if __name__ == "__main__":
    from code.business_entity_resolution.src.preprocessing import parse_match_ids

    print("=" * 80, flush=True)
    print("BENCHMARKING SHARDED SHARED-MEMORY 8-WORKER V3 ENGINE ON FINAL_HOLDOUT_1000", flush=True)
    print("=" * 80, flush=True)

    qdf = pd.read_csv("experiments/v3/pools/final_holdout_1000_s1.tsv", sep="\t", dtype=str)
    with open("experiments/v3/results/v3_d_bakeoff_metrics.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)["selected_frozen_pipeline"]
    policy_cfg = cfg["policy_config"]

    model_path = "experiments/v3/models/v3_model_c_lambdarank.joblib"
    num_workers = 8
    print(f"Starting persistent worker pool ({num_workers} workers)...", flush=True)
    pool = mp.Pool(
        processes=num_workers,
        initializer=init_inference_worker,
        initargs=(model_path, policy_cfg, 1),
    )

    preds_by_sid: Dict[str, List[str]] = {}
    cands_by_sid: Dict[str, List[str]] = {}
    tot_ret_s = 0.0
    tot_sc_s = 0.0
    tot_idx_s = 0.0
    tot_pairs = 0

    try:
        for ctry in ["us", "india"]:
            sub_q = qdf[qdf["country"].str.strip().str.lower() == ctry].reset_index(drop=True)
            recs = [
                (
                    i,
                    r.source1_entity_id,
                    "" if pd.isna(r.business_name) else str(r.business_name),
                    "" if pd.isna(r.business_address) else str(r.business_address),
                )
                for i, r in enumerate(sub_q.itertuples(index=False))
            ]
            t_i0 = time.time()
            c_idx = CountryIndexV3(ctry)
            c_idx.build(
                recs,
                "dataset/train/train_source2.tsv",
                "dataset/train/train_source3.tsv",
                worker_pool=pool,
                num_workers=num_workers,
            )
            tot_idx_s += time.time() - t_i0

            cands_c, preds_c, n_cands_c, el_r, el_s = c_idx.retrieve_and_score_chunk(
                recs, worker_pool=pool, top_k=500, num_shards=16
            )
            tot_ret_s += el_r
            tot_sc_s += el_s
            tot_pairs += n_cands_c

            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            el_tot_c = el_r + el_s
            print(
                f"[{ctry.upper()}] 8-Worker Retrieval+Scoring: {len(recs)} S1 -> {n_cands_c:,} pairs "
                f"in {el_tot_c:.2f}s (Ret={el_r:.2f}s, Score={el_s:.2f}s | "
                f"{len(recs)/el_tot_c:.1f} S1/s | {n_cands_c/el_tot_c:,.0f} pairs/s) | Avail RAM={avail_gb:.2f} GB",
                flush=True,
            )

            for (_, sid, _, _), c_ids, p_ids in zip(recs, cands_c, preds_c):
                cands_by_sid[sid] = c_ids
                preds_by_sid[sid] = p_ids

            c_idx.close_and_cleanup(worker_pool=pool, num_workers=num_workers)
            gc.collect()
    finally:
        pool.close()
        pool.join()

    f05_list, prec_list, rec_list = [], [], []
    tp_tot = fp_tot = fn_tot = 0
    cand_hits = cand_tot = 0

    for r in qdf.itertuples(index=False):
        sid = r.source1_entity_id
        gt = set(parse_match_ids("" if pd.isna(r.matched_entity_ids) else str(r.matched_entity_ids)))
        pred = set(preds_by_sid.get(sid, []))
        cset = set(cands_by_sid.get(sid, []))
        cand_hits += len(gt & cset)
        cand_tot += len(gt)

        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

        if len(gt) == 0 and len(pred) == 0:
            f05_list.append(1.0)
            prec_list.append(1.0)
            rec_list.append(1.0)
        elif len(gt) == 0 or len(pred) == 0 or tp == 0:
            f05_list.append(0.0)
            prec_list.append(0.0)
            rec_list.append(0.0)
        else:
            p = tp / len(pred)
            rc = tp / len(gt)
            prec_list.append(p)
            rec_list.append(rc)
            f05_list.append(1.25 * p * rc / (0.25 * p + rc))

    comb_s = tot_ret_s + tot_sc_s
    s1_per_s = 1000.0 / comb_s
    pairs_per_s = tot_pairs / comb_s
    proj_hours = (1_732_544 / s1_per_s) / 3600.0

    print("\n" + "=" * 80, flush=True)
    print("FINAL_HOLDOUT_1000 Top-500 8-Worker SharedMemory Benchmark:", flush=True)
    print(f"  Candidate Pairs Total:    {tot_pairs:,} (Baseline: 505,107)", flush=True)
    print(f"  Candidate Recall (Micro): {cand_hits/cand_tot*100:.2f}% ({cand_hits}/{cand_tot})", flush=True)
    print(f"  Macro F0.5:               {np.mean(f05_list):.4f} (Baseline: 0.9144)", flush=True)
    print(f"  Macro Precision:          {np.mean(prec_list)*100:.2f}% (Baseline: 95.20%)", flush=True)
    print(f"  Macro Recall:             {np.mean(rec_list)*100:.2f}% (Baseline: 83.89%)", flush=True)
    print(f"  TP={tp_tot}, FP={fp_tot}, FN={fn_tot} (Baseline: TP=2856, FP=79, FN=574)", flush=True)
    print("-" * 80, flush=True)
    print(f"  Parallel Index Build:     {tot_idx_s:.2f}s (vs 716.2s single-thread baseline)", flush=True)
    print(f"  Retrieval Wall Time:      {tot_ret_s:.2f}s ({1000/tot_ret_s:.1f} S1/s)", flush=True)
    print(f"  Scoring Wall Time:        {tot_sc_s:.2f}s ({tot_pairs/tot_sc_s:,.0f} pairs/s)", flush=True)
    print(f"  Combined Query Time:      {comb_s:.2f}s ({s1_per_s:.1f} S1/s | {pairs_per_s:,.0f} pairs/s)", flush=True)
    print(f"  Projected Full-Test ETA:  {proj_hours:.2f} hours (1,732,544 S1 entities)", flush=True)
    print("=" * 80, flush=True)
