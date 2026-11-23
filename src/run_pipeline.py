"""End-to-end reproducible pipeline orchestrator (Phases 2 through 10)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .centrality import compute_centrality_features, plot_centrality_distributions
from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_project_dirs, load_config, set_global_seed
from .data_loader import load_dataset
from .evaluation import save_evaluation_artifacts
from .explainability import generate_shap_explanations
from .feature_engineering import build_leakfree_feature_blocks
from .graph_builder import build_ppi_graph, compute_network_statistics, generate_network_visualizations
from .model_training import create_stratified_splits, run_all_models_and_ablation
from .prediction import export_protein_predictions
from .ranking import generate_protein_rankings, optimize_centrality_weights_on_validation


def run_full_pipeline(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Execute the complete research pipeline and persist all models, tables, figures, and predictions."""
    t_start = time.perf_counter()
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    seed = int(cfg.get("training", {}).get("random_seed", 42))
    set_global_seed(seed)
    dirs = get_project_dirs(PROJECT_ROOT)

    print("=" * 78)
    print("CENTRALITY-DRIVEN GRAPH MODELS FOR CANCER PROTEIN PREDICTION IN PPI NETWORKS")
    print("=" * 78)
    print(f"Dataset Config : {cfg.get('dataset_name')} ({cfg.get('dataset_type')})")
    print(f"Random Seed    : {seed}")

    # ------------------------------------------------------------------
    # PHASE 2: Dataset Extraction & Preprocessing
    # ------------------------------------------------------------------
    print("\n[Phase 2] Loading and preprocessing biological dataset...")
    t0 = time.perf_counter()
    raw_bundle = load_dataset(cfg)
    from .preprocessing import preprocess_dataset

    cleaned = preprocess_dataset(raw_bundle, cfg, output_dir=dirs["data_processed"])
    rep = cleaned.preprocessing_report
    print(
        f"  Loaded {rep['final_ppi_proteins']:,} PPI proteins, "
        f"{rep['final_undirected_edges']:,} deduplicated undirected edges "
        f"(removed {rep['self_loops_removed']:,} self-loops, {rep['duplicate_reverse_edges_removed']:,} reverse duplicates)."
    )
    print(
        f"  Labels: {rep['cancer_proteins']:,} known cancer proteins "
        f"({rep['cancer_prevalence']*100:.2f}%) vs {rep['non_cancer_proteins']:,} non-cancer. "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------
    # PHASE 3: PPI Graph Construction & Statistics
    # ------------------------------------------------------------------
    print("\n[Phase 3] Constructing weighted NetworkX PPI graph & statistics...")
    t0 = time.perf_counter()
    G = build_ppi_graph(cleaned.proteins_df, cleaned.edges_df)
    net_stats_path = dirs["results_predictions"] / "network_statistics.json"
    net_stats = compute_network_statistics(G, rep, output_path=net_stats_path)
    generate_network_visualizations(G, dirs["results_figures"])
    print(
        f"  Graph G=(V={net_stats['num_proteins']:,}, E={net_stats['num_interactions']:,}) | "
        f"Avg Degree={net_stats['average_degree']:.2f} | Density={net_stats['network_density']:.6f} | "
        f"Components={net_stats['connected_components']} (GCC={net_stats['giant_component_fraction']*100:.2f}%) | "
        f"Avg Clustering={net_stats['average_clustering_coefficient']:.4f} ({time.perf_counter() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------
    # PHASE 4: Centrality Feature Engineering
    # ------------------------------------------------------------------
    print("\n[Phase 4] Computing 8 centrality measures for all proteins...")
    t0 = time.perf_counter()
    cent_csv = dirs["data_processed"] / "centrality_features.csv"
    centrality_df, cent_meta = compute_centrality_features(G, cleaned.proteins_df, cfg, output_csv=cent_csv)
    plot_centrality_distributions(centrality_df, dirs["results_figures"])
    print(f"  Saved centrality feature matrix -> {cent_csv}")
    print(f"  Method details: betweenness={cent_meta['betweenness_method']}, closeness={cent_meta['closeness_method']}")
    print(f"  Timings (s): {cent_meta['timings_seconds']} (Total: {time.perf_counter() - t0:.1f}s)")

    # Save centrality timing metadata into network_statistics.json as well
    net_stats["centrality_computation"] = cent_meta
    with net_stats_path.open("w", encoding="utf-8") as fh:
        json.dump(net_stats, fh, indent=2)

    # ------------------------------------------------------------------
    # PHASE 5 & 7: Leak-Free Splits, Feature Engineering, Baselines, GNNs & Ablation
    # ------------------------------------------------------------------
    print("\n[Phase 5 & 7] Building leak-free splits, features, baselines, GNNs, and 5-way ablation...")
    t0 = time.perf_counter()
    y = centrality_df["cancer_label"].values.astype(int)
    t_cfg = cfg.get("training", {})
    train_idx, val_idx, test_idx = create_stratified_splits(
        y,
        val_size=float(t_cfg.get("val_size", 0.15)),
        test_size=float(t_cfg.get("test_size", 0.15)),
        seed=seed,
    )
    print(
        f"  Stratified Split: Train={len(train_idx):,} (pos={int(y[train_idx].sum())}), "
        f"Val={len(val_idx):,} (pos={int(y[val_idx].sum())}), "
        f"Test={len(test_idx):,} (pos={int(y[test_idx].sum())})"
    )

    features = build_leakfree_feature_blocks(G, cleaned.proteins_df, centrality_df, train_idx, cfg)
    train_bundle = run_all_models_and_ablation(
        G=G,
        proteins_df=cleaned.proteins_df,
        centrality_df=centrality_df,
        features=features,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cfg=cfg,
        dirs=dirs,
    )
    eval_paths = save_evaluation_artifacts(
        model_results=train_bundle.model_results,
        ablation_results=train_bundle.ablation_results,
        cv_summary=train_bundle.cv_summary,
        hub_bias=train_bundle.hub_bias,
        tables_dir=dirs["results_tables"],
        predictions_dir=dirs["results_predictions"],
        figures_dir=dirs["results_figures"],
    )
    print(f"  Best Model (selected on Validation PR-AUC): {train_bundle.best_model_name}")
    best_test = train_bundle.model_results[train_bundle.best_model_name]["test_metrics"]
    print(
        f"  Held-Out Test Metrics: PR-AUC={best_test['pr_auc']:.4f} | ROC-AUC={best_test['roc_auc']:.4f} | "
        f"F1={best_test['f1_score']:.4f} | Prec={best_test['precision']:.4f} | Rec={best_test['recall']:.4f} "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------
    # PHASE 6: Validation-Driven Centrality Ranking
    # ------------------------------------------------------------------
    print("\n[Phase 6] Optimizing centrality composite weights on validation split & ranking proteins...")
    weight_info = optimize_centrality_weights_on_validation(centrality_df, val_idx, y, seed=seed)
    rankings_csv = dirs["results_predictions"] / "cancer_protein_rankings.csv"
    rankings_df, ranking_summary = generate_protein_rankings(
        centrality_df=centrality_df,
        model_probabilities=train_bundle.best_model_probabilities,
        weight_info=weight_info,
        output_csv=rankings_csv,
    )
    with (dirs["results_predictions"] / "ranking_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(ranking_summary, fh, indent=2)
    print(f"  Optimized Centrality Weights (Val PR-AUC {weight_info['equal_weights_val_pr_auc']:.4f} -> {weight_info['optimized_val_pr_auc']:.4f}):")
    print(f"    {weight_info['weights']}")

    # ------------------------------------------------------------------
    # PHASE 8: Explainable AI (SHAP)
    # ------------------------------------------------------------------
    print("\n[Phase 8] Computing SHAP global & per-protein local explanations...")
    t0 = time.perf_counter()
    shap_summary = generate_shap_explanations(
        model=train_bundle.explain_xgb_model,
        X=train_bundle.explain_X_matrix,
        feature_names=train_bundle.explain_feature_names,
        centrality_df=centrality_df,
        probabilities=train_bundle.best_model_probabilities,
        tables_dir=dirs["results_tables"],
        processed_dir=dirs["data_processed"],
        figures_dir=dirs["results_figures"],
    )
    print(f"  SHAP Engine: {shap_summary['method']} ({time.perf_counter() - t0:.1f}s)")
    print(f"  Category Importance (%): {shap_summary['category_importance_pct']}")

    # ------------------------------------------------------------------
    # PHASE 10: Export Unified Protein Predictions Table
    # ------------------------------------------------------------------
    split_labels = ["train"] * len(centrality_df)
    for i in val_idx:
        split_labels[int(i)] = "val"
    for i in test_idx:
        split_labels[int(i)] = "test"

    pred_csv = dirs["results_predictions"] / "protein_predictions.csv"
    export_protein_predictions(
        proteins_df=cleaned.proteins_df,
        centrality_df=centrality_df,
        rankings_df=rankings_df,
        all_model_probabilities=train_bundle.all_model_probabilities,
        best_model_name=train_bundle.best_model_name,
        best_threshold=train_bundle.best_model_threshold,
        split_labels=split_labels,
        output_csv=pred_csv,
    )

    total_sec = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_sec:.1f} seconds!")
    print("=" * 78)

    comp_df = pd.read_csv(eval_paths["model_comparison_csv"])
    print("\nModel Comparison (sorted by Held-Out Test PR-AUC):")
    print(comp_df[["model", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_f1_score", "test_precision", "test_recall"]].to_string(index=False))

    ab_df = pd.read_csv(eval_paths["ablation_comparison_csv"])
    print("\n5-Way Feature Ablation Study (Held-Out Test Split):")
    print(ab_df[["ablation_id", "num_features", "test_pr_auc", "test_roc_auc", "test_f1_score"]].to_string(index=False))

    return {
        "network_statistics": net_stats,
        "best_model_name": train_bundle.best_model_name,
        "ranking_summary": ranking_summary,
        "shap_summary": shap_summary,
        "elapsed_seconds": round(total_sec, 2),
    }
