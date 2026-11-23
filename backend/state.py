"""Shared application state caching trained predictions, metrics, SHAP, and PPI graph."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backend.database import AppDatabase
from src.config import PROJECT_ROOT, get_project_dirs
from src.prediction import CancerProteinPredictor


class AppState:
    """Loads and caches all generated ML outputs and graph artifacts for low-latency API serving."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.dirs = get_project_dirs(project_root)
        self.db = AppDatabase(self.dirs["data_processed"] / "app_history.db")
        self.predictor = CancerProteinPredictor(project_root)

        self.network_stats = self._load_json(self.dirs["results_predictions"] / "network_statistics.json")
        self.eval_metrics = self._load_json(self.dirs["results_predictions"] / "evaluation_metrics.json")
        self.ranking_summary = self._load_json(self.dirs["results_predictions"] / "ranking_summary.json")

        self.centrality_df = pd.read_csv(self.dirs["data_processed"] / "centrality_features.csv")
        self.feature_importance_df = pd.read_csv(self.dirs["results_tables"] / "feature_importance.csv")
        self.model_comparison_df = pd.read_csv(self.dirs["results_tables"] / "model_comparison.csv")
        self.ablation_df = pd.read_csv(self.dirs["results_tables"] / "ablation_comparison.csv")
        self.rankings_df = pd.read_csv(self.dirs["results_predictions"] / "cancer_protein_rankings.csv")

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)


@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    return AppState(PROJECT_ROOT)
