"""Main CLI entry point for Centrality-Driven Cancer Protein Prediction in PPI Networks.

Usage:
    python main.py                                    # Run full training, evaluation, ranking & SHAP pipeline
    python main.py --config config/dataset_string_cosmic.yaml
    python main.py --predict TP53 BRCA1 EGFR          # Query trained predictions for specific proteins
    python main.py --serve --port 8000                # Launch the full-stack FastAPI + React web application
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import CancerProteinPredictor, ProteinNotFoundError
from src.run_pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centrality-Driven Graph Models for Cancer Protein Prediction in PPI Networks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_string_cosmic.yaml",
        help="Path to dataset YAML configuration file.",
    )
    parser.add_argument(
        "--predict",
        nargs="+",
        default=None,
        help="Predict cancer association for one or more protein IDs or gene symbols (e.g., TP53 BRCA1).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the FastAPI + React web application server.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.serve:
        import uvicorn

        print(f"Starting web application at http://{args.host}:{args.port} ...")
        uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.predict:
        predictor = CancerProteinPredictor(PROJECT_ROOT)
        outputs = []
        exit_code = 0
        for q in args.predict:
            try:
                res = predictor.predict_protein(q, max_neighbors=10)
                outputs.append(res)
            except ProteinNotFoundError as exc:
                outputs.append({"query": q, "error": str(exc)})
                exit_code = 1
        print(json.dumps(outputs, indent=2))
        return exit_code

    run_full_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
