"""Configuration loader, deterministic seeding, and project path management."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_string_cosmic.yaml"


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_project_dirs(root: Path = PROJECT_ROOT) -> Dict[str, Path]:
    """Return and ensure all required project directories exist."""
    dirs = {
        "root": root,
        "config": root / "config",
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "data_external": root / "data" / "external",
        "models_root": root / "models",
        "models_rf": root / "models" / "random_forest",
        "models_xgb": root / "models" / "xgboost",
        "models_gcn": root / "models" / "gcn",
        "models_saved": root / "models" / "saved_models",
        "results_figures": root / "results" / "figures",
        "results_tables": root / "results" / "tables",
        "results_predictions": root / "results" / "predictions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML dataset and training configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(rel_or_abs: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (base_dir / p).resolve()
