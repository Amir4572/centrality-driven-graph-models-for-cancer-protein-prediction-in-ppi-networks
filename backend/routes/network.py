"""Interactive PPI Network Subgraph Visualization API routes."""
from __future__ import annotations

import networkx as nx
from fastapi import APIRouter, HTTPException, Query

from backend.state import get_app_state
from src.prediction import ProteinNotFoundError

router = APIRouter(prefix="/api/network", tags=["network"])


@router.get("/ego/{protein_query}")
def get_ego_network(
    protein_query: str,
    max_nodes: int = Query(default=35, ge=5, le=80),
):
    state = get_app_state()
    try:
        return state.predictor.get_ego_subgraph(protein_query, max_nodes=max_nodes)
    except ProteinNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "protein_not_found", "query": protein_query, "message": str(exc)},
        ) from exc


@router.get("/top-hubs")
def get_top_hubs_network(
    limit: int = Query(default=45, ge=10, le=80),
    sort_by: str = Query(default="composite_centrality_score"),
):
    state = get_app_state()
    df = state.predictor.df
    col = sort_by if sort_by in {"composite_centrality_score", "cancer_probability", "degree"} else "composite_centrality_score"
    top_ids = df.sort_values(col, ascending=False).head(limit)["protein_id"].tolist()

    G = state.predictor.G
    subG = G.subgraph(top_ids)
    pos = nx.spring_layout(subG, seed=42, k=0.55, iterations=60)

    nodes_out = []
    for u in subG.nodes():
        nd = subG.nodes[u]
        x, y = pos[u]
        nodes_out.append(
            {
                "id": u,
                "gene_symbol": nd.get("gene_symbol", u),
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "is_center": False,
                "is_known_cancer": bool(nd.get("cancer_label", 0) == 1),
                "cancer_probability": round(float(nd.get("cancer_probability", 0.0)), 4),
                "composite_centrality_score": round(float(nd.get("composite_centrality_score", 0.0)), 4),
                "degree": int(nd.get("degree", 0)),
            }
        )
    edges_out = [
        {
            "source": u,
            "target": v,
            "weight": round(float(d.get("weight", 1.0)), 4),
            "combined_score": int(d.get("combined_score", 700)),
        }
        for u, v, d in subG.edges(data=True)
    ]
    return {"sort_by": col, "nodes": nodes_out, "edges": edges_out}
