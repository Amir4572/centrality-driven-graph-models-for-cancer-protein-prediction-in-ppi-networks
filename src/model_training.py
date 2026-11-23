"""Phase 5 & 7: Model Training, 5-Way Feature Ablation, 5-Fold CV, Calibration & Model Persistence.

Trains and compares:
- Reference WNC Heuristic (Leak-Free reimplementation of sevmardi/Protein-Protein-Networks)
- Centrality Baselines: Logistic Regression, Random Forest, XGBoost, SVM (RBF)
- Graph Neural Networks (PyTorch): GCN, GAT, GraphSAGE, Centrality-Driven Fusion GNN
- Combined Centrality-Driven Graph Fusion Model (Centrality + Functional + Graph Topology + GNN Embeddings)
- 5-Way Ablation Study (A: Centrality, B: Functional, C: Centrality+Functional, D: Graph Embeddings, E: Combined)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .evaluation import evaluate_degree_buckets, evaluate_predictions, find_optimal_f1_threshold
from .feature_engineering import FeatureBlocks, compute_leakfree_wnc_scores
from .graph_models import (
    CentralityDrivenFusionGNN,
    GATNet,
    GCNNet,
    GraphSAGENet,
    build_graph_tensors,
    train_gnn_model,
)


@dataclass
class TrainingBundle:
    """All outputs from training, ablation, CV, and calibration."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    model_results: Dict[str, Dict[str, Any]]
    ablation_results: Dict[str, Dict[str, Any]]
    cv_summary: Dict[str, Any]
    hub_bias: List[Dict[str, Any]]
    best_model_name: str
    best_model_probabilities: np.ndarray
    best_model_threshold: float
    # For SHAP explainability & saved inference bundle
    explain_xgb_model: XGBClassifier
    explain_X_matrix: np.ndarray
    explain_feature_names: List[str]
    calibrator: IsotonicRegression
    gnn_embeddings: np.ndarray
    gnn_embedding_names: List[str]
    all_model_probabilities: Dict[str, np.ndarray]


def create_stratified_splits(
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible, leak-free stratified Train / Validation / Test index splits."""
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=rel_val,
        stratify=y[train_val_idx],
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _fit_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    n_estimators: int = 350,
) -> XGBClassifier:
    """Train regularized XGBoost with class imbalance weighting and validation early stopping."""
    n_pos = max(float((y_train == 1).sum()), 1.0)
    n_neg = float((y_train == 0).sum())
    scale_pos = float(np.sqrt(n_neg / n_pos))

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _calibrate_on_validation(
    raw_probs: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, IsotonicRegression]:
    """Calibrate raw model probabilities using Isotonic Regression fit strictly on val_idx."""
    iso = IsotonicRegression(y_min=0.0005, y_max=0.9995, out_of_bounds="clip")
    iso.fit(raw_probs[val_idx], y[val_idx])
    cal_probs = iso.transform(raw_probs).astype(np.float64)
    return cal_probs, iso


def run_all_models_and_ablation(
    G: nx.Graph,
    proteins_df: pd.DataFrame,
    centrality_df: pd.DataFrame,
    features: FeatureBlocks,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
) -> TrainingBundle:
    """Execute full benchmark: baselines, GNNs, feature fusion, 5-way ablation (A–E), and 5-fold CV."""
    t_cfg = cfg.get("training", {})
    seed = int(t_cfg.get("random_seed", 42))
    cv_folds = int(t_cfg.get("cv_folds", 5))
    gnn_epochs = int(t_cfg.get("gnn_epochs", 120))
    gnn_hidden = int(t_cfg.get("gnn_hidden_dim", 64))
    gnn_lr = float(t_cfg.get("gnn_lr", 0.01))
    gnn_wd = float(t_cfg.get("gnn_weight_decay", 5e-4))
    gnn_patience = int(t_cfg.get("gnn_patience", 20))

    y = centrality_df["cancer_label"].values.astype(int)
    degrees = centrality_df["degree"].values.astype(float)

    model_results: Dict[str, Dict[str, Any]] = {}
    all_probs: Dict[str, np.ndarray] = {}

    def _record_model(
        name: str,
        probs: np.ndarray,
        category: str,
        feature_set: str,
    ) -> None:
        thr = find_optimal_f1_threshold(y[val_idx], probs[val_idx])
        val_ap = float(average_precision_score(y[val_idx], probs[val_idx]))
        val_roc = float(roc_auc_score(y[val_idx], probs[val_idx]))
        test_m = evaluate_predictions(y[test_idx], probs[test_idx], threshold=thr)
        model_results[name] = {
            "category": category,
            "feature_set": feature_set,
            "val_pr_auc": round(val_ap, 4),
            "val_roc_auc": round(val_roc, 4),
            "test_metrics": test_m,
        }
        all_probs[name] = probs

    # =========================================================================
    # 1. Reference Baseline: Leak-Free WNC (sevmardi/Protein-Protein-Networks)
    # =========================================================================
    print("  [1/9] Evaluating Leak-Free Reference WNC Heuristic...")
    wnc_probs = compute_leakfree_wnc_scores(G, proteins_df, features.protein_ids, train_idx, y)
    _record_model(
        "Reference WNC Heuristic (Leak-Free)",
        wnc_probs,
        category="reference_heuristic",
        feature_set="Train-only Cancer Neighbors + Shared Functions",
    )

    # =========================================================================
    # 2. Centrality Baselines: Logistic Regression, Random Forest, XGBoost, SVM
    # =========================================================================
    X_cent = features.X_centrality

    print("  [2/9] Training Logistic Regression (Centrality)...")
    lr_cent = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    lr_cent.fit(X_cent[train_idx], y[train_idx])
    lr_probs = lr_cent.predict_proba(X_cent)[:, 1]
    joblib.dump(lr_cent, dirs["models_saved"] / "logistic_regression_centrality.joblib")
    _record_model("Logistic Regression (Centrality)", lr_probs, "baseline_ml", "Centrality (11)")

    print("  [3/9] Training Random Forest (Centrality)...")
    rf_cent = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf_cent.fit(X_cent[train_idx], y[train_idx])
    rf_probs = rf_cent.predict_proba(X_cent)[:, 1]
    joblib.dump(rf_cent, dirs["models_rf"] / "random_forest_centrality.joblib")
    _record_model("Random Forest (Centrality)", rf_probs, "baseline_ml", "Centrality (11)")

    print("  [4/9] Training XGBoost (Centrality)...")
    xgb_cent = _fit_xgb_classifier(
        X_cent[train_idx], y[train_idx], X_cent[val_idx], y[val_idx], seed=seed
    )
    xgb_cent_probs = xgb_cent.predict_proba(X_cent)[:, 1]
    joblib.dump(xgb_cent, dirs["models_xgb"] / "xgboost_centrality.joblib")
    _record_model("XGBoost (Centrality)", xgb_cent_probs, "baseline_ml", "Centrality (11)")

    print("  [5/9] Training Support Vector Machine RBF (Centrality)...")
    svm_cent = CalibratedClassifierCV(
        SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        ),
        cv=3,
        ensemble=False,
    )
    svm_cent.fit(X_cent[train_idx], y[train_idx])
    svm_probs = svm_cent.predict_proba(X_cent)[:, 1]
    joblib.dump(svm_cent, dirs["models_saved"] / "svm_rbf_centrality.joblib")
    _record_model("SVM-RBF (Centrality)", svm_probs, "baseline_ml", "Centrality (11)")

    # =========================================================================
    # 3. Graph Neural Networks (PyTorch): GCN, GAT, GraphSAGE, Fusion-GNN
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [6/9] Training PyTorch GNNs (GCN, GAT, GraphSAGE, Fusion-GNN) on {device}...")
    graph_tensors = build_graph_tensors(G, features.protein_ids, device=device)

    X_gnn_in = features.X_combined_base
    in_dim = X_gnn_in.shape[1]

    gcn_model = GCNNet(in_dim=in_dim, hidden_dim=gnn_hidden, dropout=0.35).to(device)
    gcn_res = train_gnn_model(
        gcn_model, "GCN", X_gnn_in, y, graph_tensors, train_idx, val_idx,
        checkpoint_path=dirs["models_gcn"] / "gcn_model.pt",
        epochs=gnn_epochs, lr=gnn_lr, weight_decay=gnn_wd, patience=gnn_patience, seed=seed,
    )
    _record_model("GCN (PyTorch)", gcn_res.probabilities, "gnn", "Centrality + Functional + Topology")

    gat_model = GATNet(in_dim=in_dim, hidden_dim=gnn_hidden, heads=2, dropout=0.35).to(device)
    gat_res = train_gnn_model(
        gat_model, "GAT", X_gnn_in, y, graph_tensors, train_idx, val_idx,
        checkpoint_path=dirs["models_gcn"] / "gat_model.pt",
        epochs=gnn_epochs, lr=gnn_lr, weight_decay=gnn_wd, patience=gnn_patience, seed=seed + 1,
    )
    _record_model("GAT (PyTorch)", gat_res.probabilities, "gnn", "Centrality + Functional + Topology")

    sage_model = GraphSAGENet(in_dim=in_dim, hidden_dim=gnn_hidden, dropout=0.35).to(device)
    sage_res = train_gnn_model(
        sage_model, "GraphSAGE", X_gnn_in, y, graph_tensors, train_idx, val_idx,
        checkpoint_path=dirs["models_gcn"] / "graphsage_model.pt",
        epochs=gnn_epochs, lr=gnn_lr, weight_decay=gnn_wd, patience=gnn_patience, seed=seed + 2,
    )
    _record_model("GraphSAGE (PyTorch)", sage_res.probabilities, "gnn", "Centrality + Functional + Topology")

    fusion_gnn = CentralityDrivenFusionGNN(
        cent_dim=features.X_centrality.shape[1],
        func_dim=features.X_functional.shape[1],
        topo_dim=features.X_graph_topo.shape[1],
        hidden_dim=gnn_hidden,
        dropout=0.35,
    ).to(device)
    fusion_gnn_res = train_gnn_model(
        fusion_gnn, "Centrality-Driven Fusion GNN", X_gnn_in, y, graph_tensors, train_idx, val_idx,
        checkpoint_path=dirs["models_gcn"] / "centrality_fusion_gnn.pt",
        epochs=gnn_epochs, lr=gnn_lr, weight_decay=gnn_wd, patience=gnn_patience, seed=seed + 3,
    )
    _record_model(
        "Centrality-Driven Fusion GNN",
        fusion_gnn_res.probabilities,
        "proposed_gnn",
        "Centrality + Functional + Dual GraphSAGE/GCN",
    )

    # =========================================================================
    # 4. Proposed Hybrid Feature-Fusion Model (Centrality + Func + Topo + GNN Emb)
    #    Note: Use compact top-16 PCA/SVD projection of Fusion-GNN embeddings fit on train_idx
    # =========================================================================
    print("  [7/9] Training Centrality-Driven Graph Fusion XGBoost & Random Forest...")
    from sklearn.decomposition import PCA

    gnn_pca = PCA(n_components=min(16, fusion_gnn_res.embeddings.shape[1]), random_state=seed)
    gnn_pca.fit(fusion_gnn_res.embeddings[train_idx])
    gnn_emb_compact = gnn_pca.transform(fusion_gnn_res.embeddings).astype(np.float32)
    gnn_emb_names = [f"gnn_fusion_emb_{i}" for i in range(gnn_emb_compact.shape[1])]

    X_full_fusion = np.hstack([features.X_combined_base, gnn_emb_compact]).astype(np.float32)
    full_fusion_names = features.combined_base_names + gnn_emb_names

    xgb_fusion_raw = _fit_xgb_classifier(
        X_full_fusion[train_idx], y[train_idx], X_full_fusion[val_idx], y[val_idx], seed=seed, n_estimators=450
    )
    xgb_fusion_raw_probs = xgb_fusion_raw.predict_proba(X_full_fusion)[:, 1]
    xgb_fusion_cal_probs, fusion_calibrator = _calibrate_on_validation(xgb_fusion_raw_probs, val_idx, y)

    # Train interpretable combined XGBoost (Centrality + Functional + Unsupervised Graph Topology)
    # for transparent SHAP attribution without 2-stage GNN embedding absorption
    xgb_interpretable = _fit_xgb_classifier(
        features.X_combined_base[train_idx],
        y[train_idx],
        features.X_combined_base[val_idx],
        y[val_idx],
        seed=seed,
        n_estimators=400,
    )
    joblib.dump(xgb_interpretable, dirs["models_xgb"] / "xgboost_interpretable_combined.joblib")

    joblib.dump(xgb_fusion_raw, dirs["models_xgb"] / "xgboost_centrality_graph_fusion.joblib")
    joblib.dump(fusion_calibrator, dirs["models_saved"] / "isotonic_calibrator.joblib")
    joblib.dump(gnn_pca, dirs["models_saved"] / "gnn_embedding_pca.joblib")

    _record_model(
        "Centrality-Driven Graph Fusion (XGBoost)",
        xgb_fusion_cal_probs,
        "proposed_fusion",
        "Centrality + Functional + Topology + GNN Embeddings",
    )

    # Also train a Random Forest on the combined feature set for comparison
    rf_fusion = RandomForestClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf_fusion.fit(X_full_fusion[train_idx], y[train_idx])
    rf_fusion_probs = rf_fusion.predict_proba(X_full_fusion)[:, 1]
    joblib.dump(rf_fusion, dirs["models_rf"] / "random_forest_centrality_graph_fusion.joblib")
    _record_model(
        "Random Forest (Combined Graph Fusion)",
        rf_fusion_probs,
        "proposed_fusion",
        "Centrality + Functional + Topology + GNN Embeddings",
    )

    # Also evaluate an ensemble blending calibrated Fusion XGBoost + Fusion GNN (weights chosen on val)
    best_alpha = 0.7
    best_ens_ap = -1.0
    for alpha in np.linspace(0.1, 0.9, 17):
        blend_val = alpha * xgb_fusion_raw_probs[val_idx] + (1.0 - alpha) * fusion_gnn_res.probabilities[val_idx]
        ap = float(average_precision_score(y[val_idx], blend_val))
        if ap > best_ens_ap:
            best_ens_ap = ap
            best_alpha = float(alpha)
    ens_raw = best_alpha * xgb_fusion_raw_probs + (1.0 - best_alpha) * fusion_gnn_res.probabilities
    ens_cal, ens_calibrator = _calibrate_on_validation(ens_raw, val_idx, y)
    joblib.dump({"alpha": best_alpha, "calibrator": ens_calibrator}, dirs["models_saved"] / "ensemble_blender.joblib")
    _record_model(
        f"Centrality-Driven Ensemble (XGB+GNN, a={best_alpha:.2f})",
        ens_cal,
        "proposed_ensemble",
        "Centrality + Functional + Topology + GNN Fusion",
    )

    # =========================================================================
    # 5. Systematic 5-Way Ablation Study (A through E) using unified XGBoost
    # =========================================================================
    print("  [8/9] Running 5-Way Feature Ablation Study (A–E)...")
    ablation_specs = [
        ("A_Centrality_Only", "A. Centrality-Only Features", features.X_centrality),
        ("B_Functional_Only", "B. Functional Annotation-Only Features", features.X_functional),
        ("C_Centrality_Plus_Functional", "C. Centrality + Functional Annotations", features.X_centrality_functional),
        ("D_Graph_Embeddings_Only", "D. Graph Topology & Spectral Embeddings Only", features.X_graph_topo),
        ("E_Combined_Centrality_Graph", "E. Combined Centrality-Driven Graph Model (All)", X_full_fusion),
    ]
    ablation_results: Dict[str, Dict[str, Any]] = {}
    for ab_key, ab_desc, X_ab in ablation_specs:
        m_ab = _fit_xgb_classifier(X_ab[train_idx], y[train_idx], X_ab[val_idx], y[val_idx], seed=seed)
        probs_ab = m_ab.predict_proba(X_ab)[:, 1]
        thr_ab = find_optimal_f1_threshold(y[val_idx], probs_ab[val_idx])
        val_ap = float(average_precision_score(y[val_idx], probs_ab[val_idx]))
        ablation_results[ab_key] = {
            "description": ab_desc,
            "num_features": int(X_ab.shape[1]),
            "val_pr_auc": round(val_ap, 4),
            "test_metrics": evaluate_predictions(y[test_idx], probs_ab[test_idx], threshold=thr_ab),
        }

    # =========================================================================
    # 6. 5-Fold Stratified Cross-Validation (Train+Val Pool Only — Test Strictly Held Out)
    # =========================================================================
    print(f"  [9/9] Running {cv_folds}-Fold Stratified CV & Hub-Bias Analysis...")
    tv_idx = np.concatenate([train_idx, val_idx])
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_records: Dict[str, Dict[str, List[float]]] = {
        "Logistic Regression (Centrality)": {"pr_auc": [], "roc_auc": []},
        "Random Forest (Centrality)": {"pr_auc": [], "roc_auc": []},
        "XGBoost (Centrality)": {"pr_auc": [], "roc_auc": []},
        "XGBoost (Combined Base)": {"pr_auc": [], "roc_auc": []},
    }
    for tr_rel, va_rel in skf.split(tv_idx, y[tv_idx]):
        f_tr = tv_idx[tr_rel]
        f_va = tv_idx[va_rel]

        lr_f = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=seed)
        lr_f.fit(X_cent[f_tr], y[f_tr])
        p_lr = lr_f.predict_proba(X_cent[f_va])[:, 1]
        cv_records["Logistic Regression (Centrality)"]["pr_auc"].append(float(average_precision_score(y[f_va], p_lr)))
        cv_records["Logistic Regression (Centrality)"]["roc_auc"].append(float(roc_auc_score(y[f_va], p_lr)))

        rf_f = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
        rf_f.fit(X_cent[f_tr], y[f_tr])
        p_rf = rf_f.predict_proba(X_cent[f_va])[:, 1]
        cv_records["Random Forest (Centrality)"]["pr_auc"].append(float(average_precision_score(y[f_va], p_rf)))
        cv_records["Random Forest (Centrality)"]["roc_auc"].append(float(roc_auc_score(y[f_va], p_rf)))

        x_c = _fit_xgb_classifier(X_cent[f_tr], y[f_tr], X_cent[f_va], y[f_va], seed=seed, n_estimators=200)
        p_xc = x_c.predict_proba(X_cent[f_va])[:, 1]
        cv_records["XGBoost (Centrality)"]["pr_auc"].append(float(average_precision_score(y[f_va], p_xc)))
        cv_records["XGBoost (Centrality)"]["roc_auc"].append(float(roc_auc_score(y[f_va], p_xc)))

        x_b = _fit_xgb_classifier(
            features.X_combined_base[f_tr], y[f_tr], features.X_combined_base[f_va], y[f_va], seed=seed, n_estimators=250
        )
        p_xb = x_b.predict_proba(features.X_combined_base[f_va])[:, 1]
        cv_records["XGBoost (Combined Base)"]["pr_auc"].append(float(average_precision_score(y[f_va], p_xb)))
        cv_records["XGBoost (Combined Base)"]["roc_auc"].append(float(roc_auc_score(y[f_va], p_xb)))

    cv_summary = {
        mname: {
            "pr_auc_mean": round(float(np.mean(vals["pr_auc"])), 4),
            "pr_auc_std": round(float(np.std(vals["pr_auc"])), 4),
            "roc_auc_mean": round(float(np.mean(vals["roc_auc"])), 4),
            "roc_auc_std": round(float(np.std(vals["roc_auc"])), 4),
        }
        for mname, vals in cv_records.items()
    }

    # Select best overall model strictly by VALIDATION PR-AUC (never peeking at test!)
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]["val_pr_auc"])
    best_probs = all_probs[best_model_name]
    best_threshold = float(model_results[best_model_name]["test_metrics"]["threshold"])

    # Hub-bias evaluation on the held-out test split using the selected best model
    hub_bias = evaluate_degree_buckets(y[test_idx], best_probs[test_idx], degrees[test_idx])

    # Save preprocessing & feature metadata bundle for reproducible inference
    joblib.dump(
        {
            "feature_artifacts": features.artifacts,
            "full_fusion_names": full_fusion_names,
            "best_model_name": best_model_name,
            "best_threshold": best_threshold,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
        dirs["models_saved"] / "preprocessing_pipeline.joblib",
    )

    return TrainingBundle(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_results=model_results,
        ablation_results=ablation_results,
        cv_summary=cv_summary,
        hub_bias=hub_bias,
        best_model_name=best_model_name,
        best_model_probabilities=best_probs,
        best_model_threshold=best_threshold,
        explain_xgb_model=xgb_interpretable,
        explain_X_matrix=features.X_combined_base,
        explain_feature_names=features.combined_base_names,
        calibrator=fusion_calibrator,
        gnn_embeddings=gnn_emb_compact,
        gnn_embedding_names=gnn_emb_names,
        all_model_probabilities=all_probs,
    )
