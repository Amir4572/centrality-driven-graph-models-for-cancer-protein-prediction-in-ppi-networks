"""Automated Phase 11 Verification Suite (Unit, Anti-Leakage, Prediction & FastAPI Integration Tests)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app
from src.centrality import CORE_CENTRALITY_COLUMNS, compute_centrality_features
from src.config import get_project_dirs
from src.data_loader import RawDatasetBundle, load_original_schema_dataset
from src.feature_engineering import compute_leakfree_wnc_scores
from src.graph_builder import build_ppi_graph
from src.prediction import CancerProteinPredictor, ProteinNotFoundError
from src.preprocessing import preprocess_dataset, sanitize_annotation_text


# =============================================================================
# 1. Unit Tests: Preprocessing, Self-Loop Removal, Deduplication & Leak Scrubbing
# =============================================================================
def test_preprocessing_deduplicates_reverse_edges_and_removes_self_loops():
    edges = pd.DataFrame(
        {
            "protein1": ["P1", "P2", "P1", "P2", "P3"],
            "protein2": ["P2", "P1", "P1", "P3", "P2"],
            "weight": [0.8, 0.9, 1.0, 0.75, 0.75],
            "combined_score": [800, 900, 1000, 750, 750],
        }
    )
    proteins = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P_ISO"],
            "gene_symbol": ["G1", "G2", "G3", "GISO"],
            "protein_size": [300, 400, 500, 100],
            "annotation": [
                "Tumor suppressor kinase signaling",
                "Annotation not available",
                "Cell cycle regulator oncogene",
                "",
            ],
            "is_cancer": [1, 0, 1, 0],
            "cosmic_tier": [1, 0, 2, 0],
            "role_in_cancer": ["TSG", "non_cancer", "oncogene", "non_cancer"],
        }
    )
    bundle = RawDatasetBundle("unit_test", "string_cosmic", edges, proteins)
    cleaned = preprocess_dataset(bundle, {"preprocessing": {"strip_cancer_leak_terms": True}})

    assert len(cleaned.edges_df) == 2
    assert cleaned.preprocessing_report["self_loops_removed"] == 1
    assert cleaned.preprocessing_report["duplicate_reverse_edges_removed"] == 2
    # Max weight kept when reverse duplicate had higher score
    p1_p2 = cleaned.edges_df[(cleaned.edges_df["protein1"] == "P1") & (cleaned.edges_df["protein2"] == "P2")]
    assert float(p1_p2["weight"].iloc[0]) == pytest.approx(0.9)

    # Leak terms scrubbed from functional annotations
    clean_p1 = cleaned.proteins_df.loc[cleaned.proteins_df["protein_id"] == "P1", "clean_annotation"].iloc[0]
    assert "tumor" not in clean_p1.lower()
    assert "kinase" in clean_p1.lower()
    assert sanitize_annotation_text("Proto-oncogene carcinoma factor", True) == "factor"


def test_original_schema_adapter_handles_missing_annotations(tmp_path: Path):
    (tmp_path / "HumanPPI.txt").write_text("Prot_1,Prot_2\nProt_2,Prot_1\nProt_2,Prot_3\n", encoding="utf-8")
    (tmp_path / "Functions.txt").write_text("Prot_1,Func_A\nProt_1,Func_B\n", encoding="utf-8")
    (tmp_path / "Cancers.txt").write_text("Prot_1\n", encoding="utf-8")
    (tmp_path / "Test1.txt").write_text("Prot_1,cancer\nProt_3,nonCancer\n", encoding="utf-8")
    (tmp_path / "Test2.txt").write_text("Prot_2\n", encoding="utf-8")

    cfg = {
        "dataset_name": "orig_unit",
        "dataset_type": "original_schema",
        "_project_root": str(tmp_path),
        "paths": {"data_dir": str(tmp_path)},
    }
    bundle = load_original_schema_dataset(cfg)
    assert len(bundle.proteins_df) == 3
    assert int(bundle.proteins_df["is_cancer"].sum()) == 1
    assert bundle.test2_ids == ["Prot_2"]


def test_centrality_and_leakfree_wnc_on_disconnected_graph():
    proteins = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4", "P5"],
            "gene_symbol": ["G1", "G2", "G3", "G4", "G5"],
            "protein_size": [100] * 5,
            "annotation": ["kinase signaling"] * 5,
            "clean_annotation": ["kinase signaling"] * 5,
            "is_cancer": [1, 0, 1, 0, 0],
            "cosmic_tier": [1, 0, 1, 0, 0],
            "role_in_cancer": ["TSG", "non_cancer", "oncogene", "non_cancer", "non_cancer"],
            "has_annotation": [1] * 5,
        }
    )
    # Two disconnected components: (P1-P2-P3) and (P4-P5)
    edges = pd.DataFrame(
        {
            "protein1": ["P1", "P2", "P4"],
            "protein2": ["P2", "P3", "P5"],
            "weight": [0.9, 0.85, 0.8],
            "combined_score": [900, 850, 800],
        }
    )
    G = build_ppi_graph(proteins, edges)
    cdf, _ = compute_centrality_features(G, proteins, {"centrality": {}, "training": {"random_seed": 42}})
    assert len(cdf) == 5
    for col in CORE_CENTRALITY_COLUMNS:
        assert col in cdf.columns
        assert not cdf[col].isna().any()

    # Leak-free WNC: hold out P3 as test (only P1 in train_idx) -> P3's own label must NOT leak
    y = np.array([1, 0, 1, 0, 0])
    scores = compute_leakfree_wnc_scores(G, proteins, ["P1", "P2", "P3", "P4", "P5"], np.array([0, 1, 3, 4]), y)
    assert scores[1] > 0.0   # P2 neighbors train cancer P1
    assert scores[3] == 0.0  # P4 has no cancer neighbors


# =============================================================================
# 2. End-to-End Artifact Integrity Tests (Generated by main.py on Real Data)
# =============================================================================
def test_all_required_phase10_artifacts_exist_and_non_empty():
    dirs = get_project_dirs(PROJECT_ROOT)
    required_files = [
        dirs["data_processed"] / "centrality_features.csv",
        dirs["results_predictions"] / "protein_predictions.csv",
        dirs["results_predictions"] / "cancer_protein_rankings.csv",
        dirs["results_tables"] / "model_comparison.csv",
        dirs["results_tables"] / "ablation_comparison.csv",
        dirs["results_tables"] / "feature_importance.csv",
        dirs["results_predictions"] / "evaluation_metrics.json",
        dirs["results_predictions"] / "network_statistics.json",
        dirs["models_xgb"] / "xgboost_centrality_graph_fusion.joblib",
        dirs["models_rf"] / "random_forest_centrality.joblib",
        dirs["models_gcn"] / "gcn_model.pt",
        dirs["models_gcn"] / "gat_model.pt",
        dirs["models_gcn"] / "graphsage_model.pt",
        dirs["models_gcn"] / "centrality_fusion_gnn.pt",
        dirs["models_saved"] / "preprocessing_pipeline.joblib",
    ]
    for path in required_files:
        assert path.exists(), f"Missing expected artifact: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"

    cent_df = pd.read_csv(dirs["data_processed"] / "centrality_features.csv")
    assert len(cent_df) > 10000
    assert not cent_df[CORE_CENTRALITY_COLUMNS].isna().any().any()

    with (dirs["results_predictions"] / "evaluation_metrics.json").open("r", encoding="utf-8") as fh:
        ev = json.load(fh)
    assert len(ev["models"]) >= 9
    assert len(ev["ablation_study"]) == 5


# =============================================================================
# 3. Prediction Engine & Invalid Input Edge-Case Tests
# =============================================================================
def test_predictor_valid_and_invalid_protein_ids():
    predictor = CancerProteinPredictor(PROJECT_ROOT)
    tp53 = predictor.predict_protein("TP53")
    assert tp53["gene_symbol"] == "TP53"
    assert 0.0 <= tp53["prediction"]["cancer_association_probability"] <= 1.0
    assert tp53["known_label"]["is_known_cancer"] is True
    assert len(tp53["neighbors"]) > 0
    assert len(tp53["shap_explanation"]["top_contributors"]) > 0

    # Also test querying by Ensembl protein ID directly
    by_id = predictor.predict_protein(tp53["protein_id"])
    assert by_id["gene_symbol"] == "TP53"

    # Invalid & empty protein IDs must raise ProteinNotFoundError cleanly
    with pytest.raises(ProteinNotFoundError):
        predictor.predict_protein("NOT_A_REAL_PROTEIN_999")
    with pytest.raises(ProteinNotFoundError):
        predictor.predict_protein("   ")


# =============================================================================
# 4. Full-Stack FastAPI Backend + Frontend Integration Tests
# =============================================================================
def test_fastapi_endpoints_and_error_handling():
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200

    dash = client.get("/api/dashboard")
    assert dash.status_code == 200
    assert dash.json()["kpis"]["total_proteins"] > 10000

    ds = client.get("/api/dataset")
    assert ds.status_code == 200
    assert ds.json()["statistics"]["num_interactions"] > 100000

    search = client.get("/api/proteins/search?q=BRCA&limit=5")
    assert search.status_code == 200
    assert len(search.json()["results"]) >= 1

    pred_ok = client.get("/api/proteins/TP53")
    assert pred_ok.status_code == 200
    assert pred_ok.json()["gene_symbol"] == "TP53"

    # Invalid protein returns clean HTTP 404 JSON
    pred_404 = client.get("/api/proteins/DOES_NOT_EXIST_XYZ")
    assert pred_404.status_code == 404
    assert pred_404.json()["detail"]["error"] == "protein_not_found"

    # Centrality rankings & custom score
    rk = client.get("/api/centrality/rankings?metric=pagerank&limit=10")
    assert rk.status_code == 200
    assert len(rk.json()["rows"]) == 10

    custom = client.post(
        "/api/centrality/custom-score",
        json={"degree": 0.4, "betweenness": 0.3, "closeness": 0.1, "eigenvector": 0.1, "pagerank": 0.1, "limit": 10},
    )
    assert custom.status_code == 200
    assert len(custom.json()["rows"]) == 10

    # Network ego & top-hubs
    ego = client.get("/api/network/ego/TP53?max_nodes=20")
    assert ego.status_code == 200
    assert len(ego.json()["nodes"]) <= 20

    hubs = client.get("/api/network/top-hubs?limit=25")
    assert hubs.status_code == 200
    assert len(hubs.json()["nodes"]) == 25

    # Model comparison & Explainability
    comp = client.get("/api/models/comparison")
    assert comp.status_code == 200
    assert len(comp.json()["ablation_table"]) == 5

    exp_g = client.get("/api/explainability/global?top_n=15")
    assert exp_g.status_code == 200
    assert len(exp_g.json()["top_features"]) == 15

    exp_p = client.get("/api/explainability/protein/BRCA1")
    assert exp_p.status_code == 200
    assert exp_p.json()["gene_symbol"] == "BRCA1"
