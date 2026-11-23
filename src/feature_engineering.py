"""Phase 5: Leak-Free Feature Engineering & Reference WNC Baseline.

Constructs modular feature blocks fit strictly on training indices (zero test-label leakage):
- Block A (Centrality): 8 core centralities + weighted_degree + log_degree + harmonic_closeness
- Block B (Functional): Leak-scrubbed TF-IDF + SVD + annotation/size descriptors
- Block C (Centrality + Functional)
- Block D (Graph Topology Embeddings): Normalized-adjacency SVD spectral embeddings +
          1-hop unsupervised neighborhood centrality aggregations
- Block E (Combined Fusion): Centrality + Functional + Graph Topology Embeddings (+ optional GNN embeddings)

Also provides a clean, leak-free Python 3 reimplementation of the reference repository's
Weighted Neighbor Counting (WNC) heuristic (sevmardi/Protein-Protein-Networks) using ONLY
training-fold cancer labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

from .centrality import EXTENDED_CENTRALITY_COLUMNS


@dataclass
class FeatureBlocks:
    """Container for aligned, leak-free feature matrices and feature names."""

    protein_ids: List[str]
    X_centrality: np.ndarray
    centrality_names: List[str]
    X_functional: np.ndarray
    functional_names: List[str]
    X_graph_topo: np.ndarray
    graph_topo_names: List[str]
    # Combined convenience views
    X_centrality_functional: np.ndarray
    centrality_functional_names: List[str]
    X_combined_base: np.ndarray
    combined_base_names: List[str]
    # Fitted transformers (fit on train_idx only)
    artifacts: Dict[str, Any]


def compute_unsupervised_graph_topology_features(
    G: nx.Graph,
    centrality_df: pd.DataFrame,
    spectral_dims: int = 32,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Compute strictly label-free graph topology embeddings & 1-hop neighborhood centrality aggregations.

    1. Normalized adjacency spectral embeddings: top-k singular vectors of D^{-1/2} A D^{-1/2}.
    2. Unsupervised 1-hop neighborhood aggregations (mean, max, std of neighbor degree, PageRank,
       betweenness, k-core, clustering) computed purely from graph structure (zero labels).
    """
    nodelist = centrality_df["protein_id"].tolist()
    n = len(nodelist)

    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr", dtype=np.float64)
    deg_w = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg_w)
    nz = deg_w > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg_w[nz])
    D_inv_sqrt = sp.diags(inv_sqrt)
    A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

    # Truncated spectral SVD embeddings of normalized adjacency
    k_eff = min(spectral_dims, max(2, n - 2))
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    try:
        U, S, _ = svds(A_norm, k=k_eff, v0=v0, tol=1e-4)
        order = np.argsort(S)[::-1]
        spectral_emb = U[:, order] * np.sqrt(np.maximum(S[order], 0.0))[None, :]
    except Exception:
        svd = TruncatedSVD(n_components=k_eff, random_state=seed)
        spectral_emb = svd.fit_transform(A_norm)

    spectral_names = [f"spectral_emb_{i}" for i in range(spectral_emb.shape[1])]

    # Unsupervised 1-hop neighbor aggregations of core centrality signals
    A_bin = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None, format="csr", dtype=np.float64)
    deg = np.asarray(A_bin.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]
    P_mean = sp.diags(inv_deg) @ A_bin

    nbr_cols = [
        "log_degree",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "pagerank",
        "clustering_coefficient",
        "k_core",
    ]
    base_mat = centrality_df[nbr_cols].values.astype(np.float64)
    nbr_mean = P_mean @ base_mat
    nbr_sq_mean = P_mean @ (base_mat ** 2)
    nbr_std = np.sqrt(np.maximum(nbr_sq_mean - nbr_mean ** 2, 0.0))

    # 1-hop max of log_degree, pagerank, k_core
    max_cols_idx = [0, 4, 6]
    nbr_max = np.zeros((n, len(max_cols_idx)), dtype=np.float64)
    indptr = A_bin.indptr
    indices = A_bin.indices
    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if start < end:
            nbr_idx = indices[start:end]
            for c_pos, col_i in enumerate(max_cols_idx):
                nbr_max[i, c_pos] = float(base_mat[nbr_idx, col_i].max())

    # 2-hop average log-degree & PageRank
    two_hop_mean = P_mean @ nbr_mean[:, [0, 4]]

    agg_names = (
        [f"nbr_mean_{c}" for c in nbr_cols]
        + [f"nbr_std_{c}" for c in nbr_cols]
        + [f"nbr_max_{nbr_cols[c]}" for c in max_cols_idx]
        + ["two_hop_mean_log_degree", "two_hop_mean_pagerank"]
    )
    X_topo = np.hstack([spectral_emb, nbr_mean, nbr_std, nbr_max, two_hop_mean]).astype(np.float32)
    topo_names = spectral_names + agg_names
    return X_topo, topo_names


def build_leakfree_feature_blocks(
    G: nx.Graph,
    proteins_df: pd.DataFrame,
    centrality_df: pd.DataFrame,
    train_idx: np.ndarray,
    cfg: Dict[str, Any],
) -> FeatureBlocks:
    """Build all leak-free feature blocks by fitting scalers/TF-IDF/SVD strictly on train_idx."""
    prep_cfg = cfg.get("preprocessing", {})
    seed = int(cfg.get("training", {}).get("random_seed", 42))
    max_tfidf = int(prep_cfg.get("max_tfidf_features", 500))
    func_svd_dims = int(prep_cfg.get("functional_svd_dims", 32))
    spectral_dims = int(prep_cfg.get("spectral_embedding_dims", 32))

    # Align proteins_df to centrality_df ordering
    p_aligned = proteins_df.set_index("protein_id").loc[centrality_df["protein_id"]].reset_index()
    protein_ids = centrality_df["protein_id"].tolist()

    # ---- Block A: Centrality Features (fit RobustScaler strictly on train_idx) ----
    cent_names = list(EXTENDED_CENTRALITY_COLUMNS)
    X_cent_raw = centrality_df[cent_names].values.astype(np.float64)
    cent_scaler = RobustScaler()
    cent_scaler.fit(X_cent_raw[train_idx])
    X_centrality = cent_scaler.transform(X_cent_raw).astype(np.float32)

    # ---- Block B: Functional Annotation Features (fit TF-IDF + SVD strictly on train_idx) ----
    texts = p_aligned["clean_annotation"].fillna("").astype(str).tolist()
    train_texts = [texts[i] for i in train_idx]

    tfidf = TfidfVectorizer(
        max_features=max_tfidf,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    non_empty_train = [t for t in train_texts if t.strip()]
    if len(non_empty_train) >= 5:
        tfidf.fit(train_texts)
        X_tfidf_all = tfidf.transform(texts)
        n_svd = min(func_svd_dims, max(2, X_tfidf_all.shape[1] - 1))
        func_svd = TruncatedSVD(n_components=n_svd, random_state=seed)
        func_svd.fit(X_tfidf_all[train_idx])
        X_func_emb = func_svd.transform(X_tfidf_all)
    else:
        func_svd = None
        X_func_emb = np.zeros((len(protein_ids), 2), dtype=np.float64)

    # Add non-leaky structural annotation metadata (protein_size, has_annotation, token_count)
    meta_func_raw = np.column_stack(
        [
            np.log1p(p_aligned["protein_size"].values.astype(np.float64)),
            p_aligned["has_annotation"].values.astype(np.float64),
            np.log1p(p_aligned["annotation_token_count"].values.astype(np.float64)),
        ]
    )
    X_func_raw = np.hstack([X_func_emb, meta_func_raw])
    func_scaler = RobustScaler()
    func_scaler.fit(X_func_raw[train_idx])
    X_functional = func_scaler.transform(X_func_raw).astype(np.float32)
    functional_names = (
        [f"func_svd_{i}" for i in range(X_func_emb.shape[1])]
        + ["log_protein_size", "has_annotation", "log_annotation_tokens"]
    )

    # ---- Block D: Unsupervised Graph Topology Embeddings & 1-hop Aggregations ----
    X_topo_raw, topo_names = compute_unsupervised_graph_topology_features(
        G, centrality_df, spectral_dims=spectral_dims, seed=seed
    )
    topo_scaler = RobustScaler()
    topo_scaler.fit(X_topo_raw[train_idx])
    X_graph_topo = topo_scaler.transform(X_topo_raw).astype(np.float32)

    # ---- Combined Views ----
    X_cent_func = np.hstack([X_centrality, X_functional]).astype(np.float32)
    cent_func_names = cent_names + functional_names

    X_combined_base = np.hstack([X_centrality, X_functional, X_graph_topo]).astype(np.float32)
    combined_base_names = cent_names + functional_names + topo_names

    return FeatureBlocks(
        protein_ids=protein_ids,
        X_centrality=X_centrality,
        centrality_names=cent_names,
        X_functional=X_functional,
        functional_names=functional_names,
        X_graph_topo=X_graph_topo,
        graph_topo_names=topo_names,
        X_centrality_functional=X_cent_func,
        centrality_functional_names=cent_func_names,
        X_combined_base=X_combined_base,
        combined_base_names=combined_base_names,
        artifacts={
            "cent_scaler": cent_scaler,
            "tfidf": tfidf,
            "func_svd": func_svd,
            "func_scaler": func_scaler,
            "topo_scaler": topo_scaler,
        },
    )


def compute_leakfree_wnc_scores(
    G: nx.Graph,
    proteins_df: pd.DataFrame,
    protein_ids: List[str],
    train_idx: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Clean, leak-free Python 3 reimplementation of sevmardi/Protein-Protein-Networks WNC heuristic.

    Fixes all critical bugs found in Phase 1:
    - Uses ONLY training-fold cancer labels (zero test/val label leakage).
    - Excludes the target protein itself from its own neighbor/function counts.
    - Handles unannotated proteins and disconnected components cleanly.
    - Returns continuous scores normalized to [0, 1] via rank/sigmoid for ROC-AUC & PR-AUC evaluation.
    """
    train_cancer_ids = {protein_ids[i] for i in train_idx if y[i] == 1}

    # Build protein -> functional token sets (from clean_annotation keywords)
    p_aligned = proteins_df.set_index("protein_id").loc[protein_ids]
    prot_to_funcs: Dict[str, set] = {}
    func_to_train_cps: Dict[str, set] = {}

    for pid, text in zip(protein_ids, p_aligned["clean_annotation"].fillna("").astype(str)):
        tokens = {tok.lower() for tok in text.split() if len(tok) >= 4}
        prot_to_funcs[pid] = tokens
        if pid in train_cancer_ids:
            for tok in tokens:
                func_to_train_cps.setdefault(tok, set()).add(pid)

    # Precompute train-only cancer-protein degree for each train cancer protein
    cp_train_deg: Dict[str, int] = {}
    for cp in train_cancer_ids:
        cp_train_deg[cp] = sum(1 for nbr in G.neighbors(cp) if nbr in train_cancer_ids and nbr != cp)

    # Precompute function cancerweight (sum of cp_train_deg of train cancer proteins in that function)
    func_weight: Dict[str, float] = {
        fn: float(sum(cp_train_deg[c] for c in cps))
        for fn, cps in func_to_train_cps.items()
    }

    raw_scores = np.zeros(len(protein_ids), dtype=np.float64)
    for idx, pid in enumerate(protein_ids):
        cp_nbrs = [nbr for nbr in G.neighbors(pid) if nbr in train_cancer_ids and nbr != pid]
        if not cp_nbrs:
            raw_scores[idx] = 0.0
            continue
        # Union of functions of train cancer neighbors (excluding self contribution)
        shared_fns = set()
        for cp in cp_nbrs:
            shared_fns.update(prot_to_funcs.get(cp, set()))
        # Sum function weights minus self contribution if pid happens to be in train_cancer_ids
        self_deg = cp_train_deg.get(pid, 0) if pid in train_cancer_ids else 0
        w = 0.0
        for fn in shared_fns:
            fw = func_weight.get(fn, 0.0)
            if pid in func_to_train_cps.get(fn, set()):
                fw = max(0.0, fw - self_deg)
            w += fw
        # Combine direct cancer-neighbor count + function cancerweight smoothly
        raw_scores[idx] = np.log1p(w) + 0.5 * len(cp_nbrs)

    max_val = float(raw_scores.max())
    if max_val > 0:
        return (raw_scores / max_val).astype(np.float64)
    return raw_scores
