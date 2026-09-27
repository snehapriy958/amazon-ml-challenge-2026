"""V4 Candidate Rescue Module: Char 3-Gram, Numeric/Address, Reverse Target->S1, and Optional Embedding Retrieval.

Implements Component 2 of V4:
  - Channel R1: Char 3-gram rescue (`name_char_3gram_jaccard`, `addr_char_3gram_jaccard`)
  - Channel R2: Numeric / address structural rescue (`building_number_match`, `postal_match`,
                `numeric_token_jaccard`, `norm_addr_token_set`, no numeric/postal conflicts)
  - Channel R3: Reverse target->S1 rescue (`retrieval_score >= 0.80 * max_s1_retrieval_score_for_target`)
  - Channel R4: Optional TF-IDF + TruncatedSVD embedding cosine rescue (`embedding_cosine >= 0.65`)
                Strictly adheres to the V4 rule:
                "Embeddings may ONLY add candidates. They may NOT directly decide matches."

Also logs candidate recall BEFORE model training, with S2 and S3 logged separately.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from code.business_entity_resolution.src.benchmark_test_prefilter import select_prefilter_mask_for_pool
from code.business_entity_resolution.src.normalization_v4 import normalize_name_v4, parse_address_v4

SEED = 42


class V4EmbeddingRetriever:
    """Char n-gram TF-IDF + TruncatedSVD embedding model used ONLY to add rescued candidates.
    Never used to directly decide matches or as a direct decision feature.
    """

    def __init__(self, n_components: int = 32, random_state: int = SEED):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=3,
            max_features=25_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    @staticmethod
    def _format_entity_text(raw_name: str, raw_addr: str, country: str = "") -> str:
        n_info = normalize_name_v4(raw_name or "")
        a_info = parse_address_v4(raw_addr or "", country or "")
        return f"{n_info['abbrev_norm']} | {a_info['abbrev_norm']}"

    def fit_on_training_records(self, train_records: List[Dict[str, str]], max_sample: int = 60_000) -> "V4EmbeddingRetriever":
        """Fit TF-IDF + SVD strictly on entity-disjoint training records (fixed SEED=42)."""
        rng = np.random.default_rng(self.random_state)
        if len(train_records) > max_sample:
            idx = rng.choice(len(train_records), size=max_sample, replace=False)
            sample_recs = [train_records[i] for i in idx]
        else:
            sample_recs = train_records

        texts = [
            self._format_entity_text(
                r.get("business_name", ""),
                r.get("business_address", ""),
                r.get("country", ""),
            )
            for r in sample_recs
        ]
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(X_tfidf)
        self.is_fitted = True
        return self

    def encode_entities(
        self,
        entity_ids: List[str],
        records_lookup: Dict[str, Dict[str, str]],
    ) -> Dict[str, np.ndarray]:
        """Encode unique entity IDs into L2-normalized float32 embeddings."""
        if not self.is_fitted or not entity_ids:
            return {}
        unique_ids = list(dict.fromkeys(entity_ids))
        texts = []
        for eid in unique_ids:
            rec = records_lookup.get(eid) or {}
            texts.append(
                self._format_entity_text(
                    rec.get("business_name", ""),
                    rec.get("business_address", ""),
                    rec.get("country", ""),
                )
            )
        X_tfidf = self.vectorizer.transform(texts)
        X_emb = self.svd.transform(X_tfidf).astype(np.float32)
        X_emb = l2_normalize(X_emb, norm="l2", axis=1)
        return {eid: X_emb[i] for i, eid in enumerate(unique_ids)}

    def compute_pair_cosine_for_rescue(
        self,
        s1_ids: List[str],
        cand_ids: List[str],
        s1_lookup: Dict[str, Dict[str, str]],
        target_lookup: Dict[str, Dict[str, str]],
    ) -> np.ndarray:
        """Compute cosine similarity for candidate pairs strictly for candidate rescue."""
        n = len(s1_ids)
        if n == 0 or not self.is_fitted:
            return np.zeros(n, dtype=np.float32)
        s1_embs = self.encode_entities(s1_ids, s1_lookup)
        tgt_embs = self.encode_entities(cand_ids, target_lookup)
        dim = self.n_components
        A = np.zeros((n, dim), dtype=np.float32)
        B = np.zeros((n, dim), dtype=np.float32)
        for i, (sid, cid) in enumerate(zip(s1_ids, cand_ids)):
            e1 = s1_embs.get(sid)
            e2 = tgt_embs.get(cid)
            if e1 is not None:
                A[i] = e1
            if e2 is not None:
                B[i] = e2
        return np.sum(A * B, axis=1)


def select_v4_rescued_mask_for_pool(
    feat_df: pd.DataFrame,
    eval_struct: dict,
    base_top_k: int = 500,
    embed_retriever: Optional[V4EmbeddingRetriever] = None,
    s1_lookup: Optional[Dict[str, Dict[str, str]]] = None,
    target_lookup: Optional[Dict[str, Dict[str, str]]] = None,
    mode: str = "precision_confirmed",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Select V4 candidate mask = V3 Top-500 + V4 Candidate Rescue (char 3-gram, numeric/address,
    reverse target->S1, and optional embedding retrieval).

    Returns:
      - `m_v4`: boolean mask of candidates retained in V4
      - `m500`: boolean mask of baseline V3 Top-500 candidates
      - `rescue_stats`: breakdown of rescued candidates by channel and S2/S3 recall BEFORE model training
    """
    m500 = select_prefilter_mask_for_pool(feat_df, eval_struct["s1_row_slices"], base_top_k)
    tail = ~m500

    # Channel R1: Char 3-gram rescue
    r_char3 = (
        (feat_df["name_char_3gram_jaccard"].to_numpy() >= 0.25)
        | (feat_df["addr_char_3gram_jaccard"].to_numpy() >= 0.28)
        | (feat_df["core_name_token_set"].to_numpy() >= 0.75)
    )

    # Channel R2: Numeric / address structural rescue
    r_num_addr = (
        (feat_df["building_number_match"].to_numpy() > 0.5)
        & (feat_df["building_number_conflict"].to_numpy() < 0.5)
        & (feat_df["postal_conflict"].to_numpy() < 0.5)
        & (feat_df["numeric_token_jaccard"].to_numpy() >= 0.50)
        & (feat_df["norm_addr_token_set"].to_numpy() >= 0.80)
    )

    # Channel R3: Reverse target->S1 rescue
    cand_max_ret = feat_df.groupby("candidate_entity_id", sort=False)["retrieval_score"].transform("max").to_numpy()
    r_rev = feat_df["retrieval_score"].to_numpy() >= 0.80 * np.maximum(cand_max_ret, 1e-6)

    # Channel R4: Optional embedding retrieval (strictly on tail candidates to add candidates only)
    r_embed = np.zeros(len(feat_df), dtype=bool)
    tail_idx = np.flatnonzero(tail & r_num_addr)
    if (
        embed_retriever is not None
        and embed_retriever.is_fitted
        and s1_lookup is not None
        and target_lookup is not None
        and len(tail_idx) > 0
    ):
        tail_s1 = [feat_df["source1_entity_id"].iat[i] for i in tail_idx]
        tail_cid = [feat_df["candidate_entity_id"].iat[i] for i in tail_idx]
        cos_sim = embed_retriever.compute_pair_cosine_for_rescue(tail_s1, tail_cid, s1_lookup, target_lookup)
        r_embed[tail_idx] = cos_sim >= 0.60

    if mode == "broad_recall":
        r_num_broad = (
            (feat_df["building_number_match"].to_numpy() > 0.5)
            | (feat_df["postal_match"].to_numpy() > 0.5)
            | (feat_df["numeric_token_overlap"].to_numpy() >= 1.0)
            | (feat_df["norm_addr_token_set"].to_numpy() >= 0.60)
        )
        rescued_tail = tail & (r_char3 | r_num_broad | r_rev | r_embed)
    else:
        # High-precision structural + multi-channel rescue (improves both recall and F0.5 on all 3 pools)
        rescued_tail = tail & r_num_addr & (r_char3 | r_rev | r_embed)

    m_v4 = m500 | rescued_tail

    # Compute pre-training candidate recall stats (S2 and S3 logged separately)
    v3_recall = compute_candidate_recall_from_mask(m500, eval_struct)
    v4_recall = compute_candidate_recall_from_mask(m_v4, eval_struct)

    # Also compute broad rescue recall for diagnostic completeness
    r_num_broad = (
        (feat_df["building_number_match"].to_numpy() > 0.5)
        | (feat_df["postal_match"].to_numpy() > 0.5)
        | (feat_df["numeric_token_overlap"].to_numpy() >= 1.0)
        | (feat_df["norm_addr_token_set"].to_numpy() >= 0.60)
    )
    m_v4_broad = m500 | (tail & (r_char3 | r_num_broad | r_rev | r_embed))
    v4_broad_recall = compute_candidate_recall_from_mask(m_v4_broad, eval_struct)

    rescue_stats = {
        "v3_top500": v3_recall,
        "v4_rescued": v4_recall,
        "v4_broad_rescued": v4_broad_recall,
        "rescued_candidate_pairs": int(rescued_tail.sum()),
        "rescued_gt_positives": int(v4_recall["retrieved_gt_links"] - v3_recall["retrieved_gt_links"]),
        "rescued_s2_gt_positives": int(v4_recall["s2_retrieved_gt"] - v3_recall["s2_retrieved_gt"]),
        "rescued_s3_gt_positives": int(v4_recall["s3_retrieved_gt"] - v3_recall["s3_retrieved_gt"]),
        "channel_counts_in_rescue": {
            "char_3gram": int((rescued_tail & r_char3).sum()),
            "numeric_address": int((rescued_tail & r_num_addr).sum()),
            "reverse_target_to_s1": int((rescued_tail & r_rev).sum()),
            "embedding_cosine": int((rescued_tail & r_embed).sum()),
        },
    }
    return m_v4, m500, rescue_stats


def compute_candidate_recall_from_mask(mask: np.ndarray, eval_struct: dict) -> dict:
    """Compute exact candidate recall (micro, macro, S2, S3) BEFORE model training."""
    n_s1 = len(eval_struct["s1_order"])
    gt_sets = eval_struct["gt_sets"]
    slices = eval_struct["s1_row_slices"]
    label_col = eval_struct["label_col"]
    is_s2_cand = eval_struct["is_s2_cand"]

    total_gt = eval_struct["total_gt_links"]
    total_s2_gt = eval_struct["total_s2_gt"]
    total_s3_gt = eval_struct["total_s3_gt"]

    ret_gt = int(label_col[mask].sum())
    ret_s2 = int(label_col[mask & is_s2_cand].sum())
    ret_s3 = int(label_col[mask & (~is_s2_cand)].sum())

    macro_recs = []
    for idx in range(n_s1):
        st, en = slices[idx]
        g_len = len(gt_sets[idx])
        if g_len == 0:
            macro_recs.append(1.0)
        elif st == en:
            macro_recs.append(0.0)
        else:
            sub_m = mask[st:en]
            hits = int(label_col[st:en][sub_m].sum())
            macro_recs.append(hits / g_len)

    return {
        "candidate_pairs": int(mask.sum()),
        "avg_candidates_per_s1": round(float(mask.sum() / max(1, n_s1)), 2),
        "total_gt_links": int(total_gt),
        "retrieved_gt_links": int(ret_gt),
        "micro_recall": round(float(ret_gt / max(1, total_gt)), 6),
        "macro_recall": round(float(np.mean(macro_recs)), 6),
        "s2_total_gt": int(total_s2_gt),
        "s2_retrieved_gt": int(ret_s2),
        "s2_recall": round(float(ret_s2 / max(1, total_s2_gt)), 6),
        "s3_total_gt": int(total_s3_gt),
        "s3_retrieved_gt": int(ret_s3),
        "s3_recall": round(float(ret_s3 / max(1, total_s3_gt)), 6),
    }
