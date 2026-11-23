"""Explainable AI (SHAP Global Feature Importance & Local Protein Explanations) API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.state import get_app_state
from src.prediction import ProteinNotFoundError

router = APIRouter(prefix="/api/explainability", tags=["explainability"])


@router.get("/global")
def get_global_explainability(top_n: int = Query(default=25, ge=5, le=100)):
    state = get_app_state()
    df = state.feature_importance_df.head(top_n)
    return {
        "method": state.predictor.shap_meta.get("method", "TreeSHAP"),
        "base_value": state.predictor.shap_meta.get("base_value", 0.0),
        "disclaimer": state.predictor.shap_meta.get("disclaimer", ""),
        "category_importance_pct": state.predictor.shap_meta.get("category_importance_pct", []),
        "top_features": df.to_dict(orient="records"),
    }


@router.get("/protein/{protein_query}")
def get_protein_explainability(protein_query: str):
    state = get_app_state()
    try:
        res = state.predictor.predict_protein(protein_query, max_neighbors=12)
    except ProteinNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "protein_not_found", "query": protein_query, "message": str(exc)},
        ) from exc

    return {
        "protein_id": res["protein_id"],
        "gene_symbol": res["gene_symbol"],
        "known_label": res["known_label"],
        "prediction": res["prediction"],
        "network_importance": res["network_importance"],
        "shap_explanation": res["shap_explanation"],
        "neighbors": res["neighbors"],
        "scientific_disclaimer": res["scientific_disclaimer"],
    }
