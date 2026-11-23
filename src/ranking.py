"""Phase 6: Validation-Optimized Centrality Composite Scoring & Protein Ranking System.

Explicitly distinguishes:
1. Network Importance (Centrality Score) — purely topological importance in the PPI network.
2. Cancer Association Probability — calibrated ML/Graph model probability of cancer association.

Weights (w1..w5) for the composite centrality score are optimized strictly on the VALIDATION split
(never using test labels), and benchmarked against equal weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score


RANKING_CENTRALITY_COLS = [
    "degree_centrality_norm",
    "betweenness_centrality_norm",
    "closeness_centrality_norm",
    "eigenvector_centrality_norm",
    "pagerank_norm",
]

RANKING_WEIGHT_LABELS = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
]


def compute_composite_centrality_score(
    centrality_df: pd.DataFrame,
    weights: List[float] | np.ndarray,
) -> np.ndarray:
    """Compute weighted linear combination of normalized centralities:
    Score = w1*Degree + w2*Betweenness + w3*Closeness + w4*Eigenvector + w5*PageRank.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if w.sum() <= 1e-12:
        w = np.ones(5, dtype=np.float64) / 5.0
    else:
        w = w / w.sum()
    mat = centrality_df[RANKING_CENTRALITY_COLS].values.astype(np.float64)
    return mat @ w


def optimize_centrality_weights_on_validation(
    centrality_df: pd.DataFrame,
    val_idx: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Find non-negative simplex weights maximizing validation PR-AUC (with L2 prior toward stability)."""
    X_val = centrality_df[RANKING_CENTRALITY_COLS].values.astype(np.float64)[val_idx]
    y_val = y[val_idx]

    equal_w = np.ones(5, dtype=np.float64) / 5.0
    equal_scores = X_val @ equal_w
    equal_pr_auc = float(average_precision_score(y_val, equal_scores))
    equal_roc_auc = float(roc_auc_score(y_val, equal_scores))

    # Multi-start Dirichlet search + Nelder-Mead refinement on softmax-parameterized simplex
    rng = np.random.default_rng(seed)
    best_w = equal_w.copy()
    best_ap = equal_pr_auc

    # Candidate random simplex points + single-feature vertices
    candidates = [equal_w] + [np.eye(5)[i] for i in range(5)] + list(rng.dirichlet(np.ones(5), size=300))
    for cand in candidates:
        ap = float(average_precision_score(y_val, X_val @ cand))
        if ap > best_ap:
            best_ap = ap
            best_w = cand.copy()

    def neg_objective(theta: np.ndarray) -> float:
        ex = np.exp(theta - np.max(theta))
        w = ex / np.sum(ex)
        scores = X_val @ w
        ap = float(average_precision_score(y_val, scores))
        # Mild entropy/L2 regularization to avoid collapsing onto a single noisy metric
        reg = 0.002 * float(np.sum((w - 0.2) ** 2))
        return -ap + reg

    init_theta = np.log(np.maximum(best_w, 1e-4))
    res = minimize(neg_objective, init_theta, method="Nelder-Mead", options={"maxiter": 400})
    ex = np.exp(res.x - np.max(res.x))
    opt_w = ex / np.sum(ex)
    opt_scores = X_val @ opt_w
    opt_pr_auc = float(average_precision_score(y_val, opt_scores))
    opt_roc_auc = float(roc_auc_score(y_val, opt_scores))

    if opt_pr_auc < best_ap:
        opt_w = best_w
        opt_scores = X_val @ opt_w
        opt_pr_auc = float(average_precision_score(y_val, opt_scores))
        opt_roc_auc = float(roc_auc_score(y_val, opt_scores))

    return {
        "weights": {lbl: round(float(val), 4) for lbl, val in zip(RANKING_WEIGHT_LABELS, opt_w)},
        "weights_vector": [float(val) for val in opt_w],
        "equal_weights_val_pr_auc": round(equal_pr_auc, 5),
        "equal_weights_val_roc_auc": round(equal_roc_auc, 5),
        "optimized_val_pr_auc": round(opt_pr_auc, 5),
        "optimized_val_roc_auc": round(opt_roc_auc, 5),
    }


def generate_protein_rankings(
    centrality_df: pd.DataFrame,
    model_probabilities: np.ndarray,
    weight_info: Dict[str, Any],
    output_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate full & Top-10/20/50 rankings separating network centrality from cancer probability."""
    opt_weights = weight_info["weights_vector"]
    composite_scores = compute_composite_centrality_score(centrality_df, opt_weights)

    rankings_df = centrality_df[
        [
            "protein_id",
            "gene_symbol",
            "cancer_label",
            "cosmic_tier",
            "role_in_cancer",
            "degree",
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "pagerank",
            "clustering_coefficient",
            "k_core",
            "network_constraint",
        ]
    ].copy()

    rankings_df["composite_centrality_score"] = composite_scores
    rankings_df["cancer_association_probability"] = model_probabilities
    rankings_df["centrality_rank"] = (
        rankings_df["composite_centrality_score"].rank(ascending=False, method="min").astype(int)
    )
    rankings_df["model_prediction_rank"] = (
        rankings_df["cancer_association_probability"].rank(ascending=False, method="min").astype(int)
    )
    rankings_df["is_known_cancer"] = rankings_df["cancer_label"] == 1
    rankings_df["candidate_status"] = np.where(
        rankings_df["is_known_cancer"],
        "Known COSMIC Cancer Protein",
        "Novel Candidate (Unannotated in COSMIC)",
    )

    # Sort primarily by calibrated model cancer association probability
    rankings_df = rankings_df.sort_values(
        ["cancer_association_probability", "composite_centrality_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rankings_df.to_csv(output_csv, index=False)

    # Extract Top 10 / 20 / 50 novel candidate proteins (non-COSMIC) & overall centrality hubs
    novel_df = rankings_df[~rankings_df["is_known_cancer"]].copy()
    centrality_sorted = rankings_df.sort_values("composite_centrality_score", ascending=False)

    summary = {
        "weight_optimization": weight_info,
        "top_10_novel_candidates": novel_df.head(10)[
            ["protein_id", "gene_symbol", "cancer_association_probability", "composite_centrality_score", "degree"]
        ].to_dict(orient="records"),
        "top_20_novel_candidates_count": int(min(20, len(novel_df))),
        "top_50_novel_candidates_count": int(min(50, len(novel_df))),
        "top_10_centrality_hubs": centrality_sorted.head(10)[
            ["protein_id", "gene_symbol", "is_known_cancer", "composite_centrality_score", "cancer_association_probability", "degree"]
        ].to_dict(orient="records"),
        "top_50_model_known_cancer_enrichment": int(rankings_df.head(50)["is_known_cancer"].sum()),
        "top_50_centrality_known_cancer_enrichment": int(centrality_sorted.head(50)["is_known_cancer"].sum()),
    }
    return rankings_df, summary
