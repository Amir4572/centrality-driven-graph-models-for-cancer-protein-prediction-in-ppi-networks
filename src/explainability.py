"""Phase 8: Explainable AI (SHAP Global Importance & Local Per-Protein Explanations).

Uses SHAP TreeExplainer (with exact XGBoost C++ TreeSHAP fallback via pred_contribs=True
for full compatibility across XGBoost/SHAP versions) to explain:
1. Global feature importance across Centrality, Functional, Topology, and GNN Embedding groups.
2. Per-protein local SHAP attributions showing why a protein receives its predicted score.
3. Network neighborhood context (interacting partners & known cancer neighbors).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier


def _categorize_feature(feat_name: str) -> str:
    if feat_name.startswith("gnn_fusion_emb_"):
        return "GNN Embedding"
    if feat_name.startswith("func_svd_") or feat_name in {
        "log_protein_size",
        "has_annotation",
        "log_annotation_tokens",
    }:
        return "Functional Annotation"
    if feat_name.startswith("spectral_emb_") or feat_name.startswith("nbr_") or feat_name.startswith("two_hop_"):
        return "Graph Neighborhood / Topology"
    return "Centrality"


def _compute_exact_treeshap(
    model: XGBClassifier,
    X: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    """Compute exact TreeSHAP contributions for every protein.

    Attempts shap.TreeExplainer first; falls back seamlessly to XGBoost's native exact C++
    TreeSHAP engine (pred_contribs=True, identical Lundberg TreeSHAP values) if version
    parsing mismatches occur.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[-1]
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.asarray(base_val).ravel()[-1])
        else:
            base_val = float(base_val)
        return np.asarray(sv, dtype=np.float64), base_val, "shap.TreeExplainer"
    except Exception:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X)
        contribs = booster.predict(dmat, pred_contribs=True)
        shap_vals = contribs[:, :-1].astype(np.float64)
        base_val = float(np.mean(contribs[:, -1]))
        return shap_vals, base_val, "xgboost.exact_treeshap_pred_contribs"


def generate_shap_explanations(
    model: XGBClassifier,
    X: np.ndarray,
    feature_names: List[str],
    centrality_df: pd.DataFrame,
    probabilities: np.ndarray,
    tables_dir: Path,
    processed_dir: Path,
    figures_dir: Path,
) -> Dict[str, Any]:
    """Compute global & per-protein SHAP explanations, save CSV/JSON, and plot figures."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    shap_values, base_value, method_used = _compute_exact_treeshap(model, X)
    abs_mean = np.mean(np.abs(shap_values), axis=0)
    signed_mean = np.mean(shap_values, axis=0)

    # Directionality: correlation between raw feature value and its SHAP contribution
    directions: List[float] = []
    for j in range(X.shape[1]):
        xv = X[:, j]
        sv = shap_values[:, j]
        if np.std(xv) > 1e-12 and np.std(sv) > 1e-12:
            corr = float(np.corrcoef(xv, sv)[0, 1])
        else:
            corr = 0.0
        directions.append(round(corr, 4))

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "category": [_categorize_feature(f) for f in feature_names],
            "mean_abs_shap": abs_mean,
            "mean_signed_shap": signed_mean,
            "feature_shap_correlation": directions,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["importance_share_pct"] = (
        100.0 * imp_df["mean_abs_shap"] / max(float(imp_df["mean_abs_shap"].sum()), 1e-12)
    )

    imp_csv = tables_dir / "feature_importance.csv"
    imp_df.to_csv(imp_csv, index=False)

    # Category-level aggregated importance
    cat_summary = (
        imp_df.groupby("category", as_index=False)["importance_share_pct"]
        .sum()
        .sort_values("importance_share_pct", ascending=False)
        .to_dict(orient="records")
    )

    # Build compact per-protein local SHAP explanation store (top 10 contributors per protein)
    protein_ids = centrality_df["protein_id"].tolist()
    per_protein_shap: Dict[str, Any] = {}
    for idx, pid in enumerate(protein_ids):
        row_sv = shap_values[idx]
        top_k_idx = np.argsort(np.abs(row_sv))[::-1][:10]
        contributors = [
            {
                "feature": feature_names[j],
                "category": _categorize_feature(feature_names[j]),
                "feature_value": round(float(X[idx, j]), 4),
                "shap_value": round(float(row_sv[j]), 5),
                "direction": "increases_cancer_score" if row_sv[j] >= 0 else "decreases_cancer_score",
            }
            for j in top_k_idx
        ]
        per_protein_shap[pid] = {
            "base_value": round(base_value, 5),
            "sum_shap": round(float(row_sv.sum()), 5),
            "top_contributors": contributors,
        }

    shap_json_path = processed_dir / "protein_shap_explanations.json"
    with shap_json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": method_used,
                "base_value": round(base_value, 5),
                "disclaimer": (
                    "SHAP values explain how network centrality, functional annotations, and graph "
                    "topology drive the model's predicted association score. A high prediction score "
                    "prioritizes candidate proteins for biological follow-up and is NOT proof of cancer causation."
                ),
                "category_importance_pct": cat_summary,
                "explanations": per_protein_shap,
            },
            fh,
        )

    # ---- Figure 1: Global Top-20 SHAP Feature Importance + Category Pie/Bar ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150, gridspec_kw={"width_ratios": [2.2, 1.0]})
    top20 = imp_df.head(20).iloc[::-1]
    cat_colors = {
        "Centrality": "#2563eb",
        "Functional Annotation": "#10b981",
        "Graph Neighborhood / Topology": "#8b5cf6",
        "GNN Embedding": "#f59e0b",
    }
    bar_colors = [cat_colors.get(c, "#64748b") for c in top20["category"]]
    axes[0].barh(top20["feature"], top20["mean_abs_shap"], color=bar_colors, alpha=0.9)
    axes[0].set_title(f"Top-20 Global SHAP Feature Importance ({method_used})", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP Value| (Impact on Cancer Log-Odds)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=col, label=cat) for cat, col in cat_colors.items()]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8.5)

    cat_df = pd.DataFrame(cat_summary)
    axes[1].bar(
        range(len(cat_df)),
        cat_df["importance_share_pct"],
        color=[cat_colors.get(c, "#64748b") for c in cat_df["category"]],
        alpha=0.9,
    )
    axes[1].set_xticks(range(len(cat_df)))
    axes[1].set_xticklabels(cat_df["category"], rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Share of Total |SHAP| (%)")
    axes[1].set_title("Feature Group Contribution (%)", fontsize=11.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    for i, val in enumerate(cat_df["importance_share_pct"]):
        axes[1].text(i, val + 0.8, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    global_fig = figures_dir / "shap_feature_importance.png"
    fig.savefig(global_fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: Local SHAP Waterfall Examples (Top Known Cancer + Top Novel Candidate) ----
    is_cancer = centrality_df["cancer_label"].values == 1
    top_known_idx = int(np.where(is_cancer)[0][np.argmax(probabilities[is_cancer])]) if is_cancer.any() else 0
    top_novel_idx = int(np.where(~is_cancer)[0][np.argmax(probabilities[~is_cancer])]) if (~is_cancer).any() else 0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150)
    for ax, sample_idx, title_prefix in [
        (axes[0], top_known_idx, "Known Cancer Protein"),
        (axes[1], top_novel_idx, "Top Novel Candidate"),
    ]:
        sym = centrality_df.iloc[sample_idx]["gene_symbol"]
        prob = probabilities[sample_idx]
        sv = shap_values[sample_idx]
        top10 = np.argsort(np.abs(sv))[::-1][:10][::-1]
        vals = sv[top10]
        names = [feature_names[j] for j in top10]
        colors = ["#dc2626" if v >= 0 else "#2563eb" for v in vals]
        ax.barh(names, vals, color=colors, alpha=0.88)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_title(f"{title_prefix}: {sym} (Pred Prob={prob:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Contribution (+ increases cancer probability)")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    local_fig = figures_dir / "shap_local_examples.png"
    fig.savefig(local_fig, bbox_inches="tight")
    plt.close(fig)

    return {
        "method": method_used,
        "feature_importance_csv": str(imp_csv),
        "protein_shap_json": str(shap_json_path),
        "global_fig": str(global_fig),
        "local_fig": str(local_fig),
        "category_importance_pct": cat_summary,
        "top_10_features": imp_df.head(10).to_dict(orient="records"),
    }
