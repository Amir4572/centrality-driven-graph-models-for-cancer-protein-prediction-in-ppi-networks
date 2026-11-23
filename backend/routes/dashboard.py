"""Dashboard and Dataset Information API routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.state import get_app_state

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard")
def get_dashboard_summary():
    state = get_app_state()
    ns = state.network_stats
    ev = state.eval_metrics
    best_name = ev.get("best_model_by_test_pr_auc", "")
    best_metrics = ev.get("models", {}).get(best_name, {}).get("test_metrics", {})

    return {
        "kpis": {
            "total_proteins": ns.get("num_proteins", 0),
            "total_interactions": ns.get("num_interactions", 0),
            "cancer_proteins": ns.get("cancer_protein_count", 0),
            "non_cancer_proteins": ns.get("non_cancer_protein_count", 0),
            "best_model_name": best_name,
            "best_pr_auc": best_metrics.get("pr_auc", 0.0),
            "best_roc_auc": best_metrics.get("roc_auc", 0.0),
            "best_f1": best_metrics.get("f1_score", 0.0),
        },
        "network_statistics": ns,
        "top_novel_candidates": state.ranking_summary.get("top_10_novel_candidates", []),
        "top_centrality_hubs": state.ranking_summary.get("top_10_centrality_hubs", []),
        "hub_bias": ev.get("hub_bias_degree_buckets", []),
        "weight_optimization": state.ranking_summary.get("weight_optimization", {}),
    }


@router.get("/dataset")
def get_dataset_information():
    state = get_app_state()
    ns = state.network_stats
    prep = ns.get("preprocessing", {})
    prov = prep.get("provenance", {})
    return {
        "dataset_name": prep.get("dataset_name", "STRING_v12.5_COSMIC_v104"),
        "dataset_type": prep.get("dataset_type", "string_cosmic"),
        "sources": {
            "ppi_database": "STRING v12.5 Homo sapiens (CC BY 4.0)",
            "cancer_labels": "COSMIC Cancer Gene Census v104 GRCh38 (Wellcome Sanger Institute)",
            "functional_annotations": "STRING v12.5 curated protein annotations (cancer-leakage terms scrubbed)",
            "original_repo_note": (
                "The original sevmardi/Protein-Protein-Networks dataset is private and not publicly distributed. "
                "A separate adapter configuration (config/dataset_original_schema.yaml) is maintained for compatibility."
            ),
        },
        "statistics": {
            "num_proteins": ns.get("num_proteins", 0),
            "num_interactions": ns.get("num_interactions", 0),
            "cancer_proteins": ns.get("cancer_protein_count", 0),
            "non_cancer_proteins": ns.get("non_cancer_protein_count", 0),
            "cancer_prevalence_pct": round(prep.get("cancer_prevalence", 0.0) * 100.0, 2),
            "connected_components": ns.get("connected_components", 0),
            "giant_component_fraction_pct": round(ns.get("giant_component_fraction", 0.0) * 100.0, 2),
            "average_degree": ns.get("average_degree", 0.0),
            "network_density": ns.get("network_density", 0.0),
            "average_clustering": ns.get("average_clustering_coefficient", 0.0),
        },
        "preprocessing": prep,
        "provenance": prov,
        "centrality_computation": ns.get("centrality_computation", {}),
    }
