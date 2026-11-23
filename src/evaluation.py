"""Phase 7: Leak-Free Evaluation Suite, Hub-Bias Analysis, Comparison Tables & Figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select classification threshold maximizing F1 strictly on the validation set."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1s = (2.0 * prec[:-1] * rec[:-1]) / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best_i = int(np.argmax(f1s))
    return float(np.clip(thresholds[best_i], 0.05, 0.95))


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute complete classification and ranking evaluation metrics."""
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)

    # Downsample curves to <= 120 points for lightweight JSON serving in the web app
    def _downsample(a: np.ndarray, b: np.ndarray, max_pts: int = 120) -> Tuple[List[float], List[float]]:
        if len(a) <= max_pts:
            return [round(float(x), 4) for x in a], [round(float(y), 4) for y in b]
        idx = np.linspace(0, len(a) - 1, max_pts).astype(int)
        return [round(float(a[i]), 4) for i in idx], [round(float(b[i]), 4) for i in idx]

    roc_fpr_s, roc_tpr_s = _downsample(fpr, tpr)
    pr_rec_s, pr_prec_s = _downsample(pr_rec, pr_prec)

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": roc_fpr_s, "tpr": roc_tpr_s},
        "pr_curve": {"recall": pr_rec_s, "precision": pr_prec_s},
    }


def evaluate_degree_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    degrees: np.ndarray,
) -> List[Dict[str, Any]]:
    """Hub-bias diagnostic: evaluate ROC-AUC and PR-AUC within Low / Medium / High degree tiers."""
    q33, q67 = np.percentile(degrees, [33.3, 66.7])
    buckets = [
        ("Low Degree", degrees <= q33),
        ("Medium Degree", (degrees > q33) & (degrees <= q67)),
        ("High Degree (Hubs)", degrees > q67),
    ]
    rows: List[Dict[str, Any]] = []
    for label, mask in buckets:
        yt = y_true[mask]
        yp = y_prob[mask]
        n_pos = int(yt.sum())
        if len(yt) > 0 and n_pos > 0 and n_pos < len(yt):
            roc = round(float(roc_auc_score(yt, yp)), 4)
            pr = round(float(average_precision_score(yt, yp)), 4)
        else:
            roc, pr = 0.0, 0.0
        rows.append(
            {
                "bucket": label,
                "degree_min": int(degrees[mask].min()) if mask.any() else 0,
                "degree_max": int(degrees[mask].max()) if mask.any() else 0,
                "count": int(mask.sum()),
                "cancer_count": n_pos,
                "cancer_prevalence": round(n_pos / max(int(mask.sum()), 1), 4),
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )
    return rows


def save_evaluation_artifacts(
    model_results: Dict[str, Dict[str, Any]],
    ablation_results: Dict[str, Dict[str, Any]],
    cv_summary: Dict[str, Any],
    hub_bias: List[Dict[str, Any]],
    tables_dir: Path,
    predictions_dir: Path,
    figures_dir: Path,
) -> Dict[str, Path]:
    """Export model_comparison.csv, ablation_comparison.csv, evaluation_metrics.json, and plots."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. model_comparison.csv
    rows = []
    for name, res in model_results.items():
        m = res["test_metrics"]
        cm = m["confusion_matrix"]
        rows.append(
            {
                "model": name,
                "category": res.get("category", "model"),
                "feature_set": res.get("feature_set", ""),
                "val_pr_auc": res.get("val_pr_auc", 0.0),
                "test_accuracy": m["accuracy"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "test_f1_score": m["f1_score"],
                "test_roc_auc": m["roc_auc"],
                "test_pr_auc": m["pr_auc"],
                "test_brier_score": m["brier_score"],
                "tp": cm["tp"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tn": cm["tn"],
            }
        )
    comp_df = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
    comp_csv = tables_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    # 2. ablation_comparison.csv (A through E)
    ab_rows = []
    for key, res in ablation_results.items():
        m = res["test_metrics"]
        ab_rows.append(
            {
                "ablation_id": key,
                "description": res["description"],
                "num_features": res["num_features"],
                "val_pr_auc": res["val_pr_auc"],
                "test_roc_auc": m["roc_auc"],
                "test_pr_auc": m["pr_auc"],
                "test_f1_score": m["f1_score"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
            }
        )
    ab_df = pd.DataFrame(ab_rows)
    ab_csv = tables_dir / "ablation_comparison.csv"
    ab_df.to_csv(ab_csv, index=False)

    # 3. evaluation_metrics.json
    eval_json_path = predictions_dir / "evaluation_metrics.json"
    payload = {
        "models": model_results,
        "ablation_study": ablation_results,
        "cross_validation_5fold": cv_summary,
        "hub_bias_degree_buckets": hub_bias,
        "best_model_by_test_pr_auc": comp_df.iloc[0]["model"] if not comp_df.empty else "",
    }
    with eval_json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # 4. ROC & Precision-Recall Curves Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(model_results), 1)))
    for (name, res), color in zip(model_results.items(), palette):
        m = res["test_metrics"]
        axes[0].plot(
            m["roc_curve"]["fpr"],
            m["roc_curve"]["tpr"],
            lw=2,
            color=color,
            label=f"{name} (AUC={m['roc_auc']:.3f})",
        )
        axes[1].plot(
            m["pr_curve"]["recall"],
            m["pr_curve"]["precision"],
            lw=2,
            color=color,
            label=f"{name} (AP={m['pr_auc']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_title("Test ROC Curves (All Models)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].set_title("Test Precision-Recall Curves (Primary Minority-Class Metric)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=7.5, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    curves_fig = figures_dir / "roc_pr_curves.png"
    fig.savefig(curves_fig, bbox_inches="tight")
    plt.close(fig)

    # 5. 5-Way Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(ab_df))
    w = 0.28
    ax.bar(x - w, ab_df["test_pr_auc"], width=w, color="#2563eb", label="PR-AUC")
    ax.bar(x, ab_df["test_roc_auc"], width=w, color="#10b981", label="ROC-AUC")
    ax.bar(x + w, ab_df["test_f1_score"], width=w, color="#f59e0b", label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_df["ablation_id"], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("5-Way Feature Ablation Study (A–E) on Held-Out Test Split", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for idx_b, row in ab_df.iterrows():
        ax.text(idx_b - w, row["test_pr_auc"] + 0.015, f"{row['test_pr_auc']:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    ab_fig = figures_dir / "ablation_comparison.png"
    fig.savefig(ab_fig, bbox_inches="tight")
    plt.close(fig)

    # 6. Confusion Matrices of Top 4 Models
    top4 = comp_df.head(4)["model"].tolist()
    fig, axes = plt.subplots(1, len(top4), figsize=(4 * len(top4), 3.8), dpi=150)
    if len(top4) == 1:
        axes = [axes]
    for ax, mname in zip(axes, top4):
        cm = model_results[mname]["test_metrics"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{mat[r, c]:,}", ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if mat[r, c] > mat.max() * 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{mname}\nF1={model_results[mname]['test_metrics']['f1_score']:.3f}", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    cm_fig = figures_dir / "confusion_matrices.png"
    fig.savefig(cm_fig, bbox_inches="tight")
    plt.close(fig)

    return {
        "model_comparison_csv": comp_csv,
        "ablation_comparison_csv": ab_csv,
        "evaluation_metrics_json": eval_json_path,
        "roc_pr_curves_fig": curves_fig,
        "ablation_fig": ab_fig,
        "confusion_matrices_fig": cm_fig,
    }
