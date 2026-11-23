"""Centrality Analysis Rankings & Configurable Composite Centrality Score API routes."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.state import get_app_state
from src.ranking import compute_composite_centrality_score

router = APIRouter(prefix="/api/centrality", tags=["centrality"])

ALLOWED_METRICS = {
    "composite": "composite_centrality_score",
    "model_probability": "cancer_probability",
    "degree": "degree_centrality",
    "betweenness": "betweenness_centrality",
    "closeness": "closeness_centrality",
    "eigenvector": "eigenvector_centrality",
    "pagerank": "pagerank",
    "clustering": "clustering_coefficient",
    "k_core": "k_core",
    "constraint": "network_constraint",
}


class CustomCentralityRequest(BaseModel):
    degree: float = Field(default=0.2, ge=0.0, le=10.0)
    betweenness: float = Field(default=0.2, ge=0.0, le=10.0)
    closeness: float = Field(default=0.2, ge=0.0, le=10.0)
    eigenvector: float = Field(default=0.2, ge=0.0, le=10.0)
    pagerank: float = Field(default=0.2, ge=0.0, le=10.0)
    limit: int = Field(default=30, ge=5, le=100)
    only_novel: bool = Field(default=False)


@router.get("/rankings")
def get_centrality_rankings(
    metric: str = Query(default="composite", description="Centrality metric key"),
    limit: int = Query(default=50, ge=5, le=200),
    filter_cancer: Literal["all", "known", "novel"] = Query(default="all"),
):
    state = get_app_state()
    col = ALLOWED_METRICS.get(metric, "composite_centrality_score")
    ascending = metric == "constraint"  # lower constraint = greater structural brokerage

    df = state.predictor.df.copy()
    if filter_cancer == "known":
        df = df[df["cancer_label"] == 1]
    elif filter_cancer == "novel":
        df = df[df["cancer_label"] == 0]

    top = df.sort_values(col, ascending=ascending).head(limit)
    rows = []
    for rank_i, r in enumerate(top.itertuples(index=False), start=1):
        rows.append(
            {
                "rank": rank_i,
                "protein_id": r.protein_id,
                "gene_symbol": r.gene_symbol,
                "is_known_cancer": bool(r.cancer_label == 1),
                "role_in_cancer": r.role_in_cancer,
                "degree": int(r.degree),
                "degree_centrality": round(float(r.degree_centrality), 6),
                "betweenness_centrality": round(float(r.betweenness_centrality), 6),
                "closeness_centrality": round(float(r.closeness_centrality), 6),
                "eigenvector_centrality": round(float(r.eigenvector_centrality), 6),
                "pagerank": round(float(r.pagerank), 6),
                "clustering_coefficient": round(float(r.clustering_coefficient), 4),
                "k_core": int(r.k_core),
                "network_constraint": round(float(r.network_constraint), 4),
                "composite_centrality_score": round(float(r.composite_centrality_score), 4),
                "cancer_probability": round(float(r.cancer_probability), 4),
            }
        )

    return {
        "metric": metric,
        "column": col,
        "filter_cancer": filter_cancer,
        "validated_weights": state.ranking_summary.get("weight_optimization", {}),
        "rows": rows,
    }


@router.post("/custom-score")
def compute_custom_composite_ranking(req: CustomCentralityRequest):
    state = get_app_state()
    weights = [req.degree, req.betweenness, req.closeness, req.eigenvector, req.pagerank]
    total = sum(weights)
    norm_w = [w / total for w in weights] if total > 1e-12 else [0.2] * 5

    scores = compute_composite_centrality_score(state.centrality_df, norm_w)
    merged = state.predictor.df[["protein_id", "gene_symbol", "cancer_label", "degree", "cancer_probability", "role_in_cancer"]].copy()
    score_map = dict(zip(state.centrality_df["protein_id"], scores))
    merged["custom_centrality_score"] = merged["protein_id"].map(score_map)

    if req.only_novel:
        merged = merged[merged["cancer_label"] == 0]

    top = merged.sort_values("custom_centrality_score", ascending=False).head(req.limit)
    rows = [
        {
            "rank": idx,
            "protein_id": r.protein_id,
            "gene_symbol": r.gene_symbol,
            "is_known_cancer": bool(r.cancer_label == 1),
            "role_in_cancer": r.role_in_cancer,
            "degree": int(r.degree),
            "custom_centrality_score": round(float(r.custom_centrality_score), 4),
            "cancer_probability": round(float(r.cancer_probability), 4),
        }
        for idx, r in enumerate(top.itertuples(index=False), start=1)
    ]

    weight_dict = {
        "degree": round(norm_w[0], 4),
        "betweenness": round(norm_w[1], 4),
        "closeness": round(norm_w[2], 4),
        "eigenvector": round(norm_w[3], 4),
        "pagerank": round(norm_w[4], 4),
    }
    if rows:
        state.db.log_centrality_experiment(weight_dict, rows[0]["gene_symbol"], rows[0]["custom_centrality_score"])

    return {
        "normalized_weights": weight_dict,
        "only_novel": req.only_novel,
        "rows": rows,
    }
