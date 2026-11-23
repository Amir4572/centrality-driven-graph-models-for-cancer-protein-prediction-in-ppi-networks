"""Model Comparison, 5-Way Ablation, Confusion Matrices, and ROC/PR Curves API routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.state import get_app_state

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/comparison")
def get_model_comparison():
    state = get_app_state()
    ev = state.eval_metrics
    return {
        "best_model": ev.get("best_model_by_test_pr_auc", ""),
        "comparison_table": state.model_comparison_df.to_dict(orient="records"),
        "ablation_table": state.ablation_df.to_dict(orient="records"),
        "cross_validation_5fold": ev.get("cross_validation_5fold", {}),
        "hub_bias_degree_buckets": ev.get("hub_bias_degree_buckets", []),
        "models_detail": ev.get("models", {}),
    }
