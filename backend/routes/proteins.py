"""Protein Search, Individual Prediction, and Recent Query History API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.state import get_app_state
from src.prediction import ProteinNotFoundError

router = APIRouter(prefix="/api/proteins", tags=["proteins"])


@router.get("/search")
def search_proteins(
    q: str = Query(default="", description="Gene symbol or Ensembl protein ID prefix"),
    limit: int = Query(default=15, ge=1, le=100),
):
    state = get_app_state()
    return {"query": q, "results": state.predictor.search_proteins(q, limit=limit)}


@router.get("/history/recent")
def get_recent_history(limit: int = Query(default=15, ge=1, le=50)):
    state = get_app_state()
    return {"history": state.db.get_recent_queries(limit=limit)}


@router.get("/{protein_query}")
def predict_protein(
    protein_query: str,
    max_neighbors: int = Query(default=25, ge=1, le=100),
):
    state = get_app_state()
    try:
        result = state.predictor.predict_protein(protein_query, max_neighbors=max_neighbors)
    except ProteinNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "protein_not_found",
                "query": protein_query,
                "message": str(exc),
            },
        ) from exc

    state.db.log_prediction_query(
        query_input=protein_query,
        protein_id=result["protein_id"],
        gene_symbol=result["gene_symbol"],
        cancer_probability=result["prediction"]["cancer_association_probability"],
        composite_centrality=result["network_importance"]["composite_centrality_score"],
        is_known_cancer=result["known_label"]["is_known_cancer"],
    )
    return result
