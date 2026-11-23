"""SQLite persistence for prediction history and custom centrality weight experiments."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.config import PROJECT_ROOT, get_project_dirs


class AppDatabase:
    """Lightweight SQLite store for user queries and custom centrality ranking experiments."""

    def __init__(self, db_path: Path | None = None):
        dirs = get_project_dirs(PROJECT_ROOT)
        self.db_path = db_path or (dirs["data_processed"] / "app_history.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queried_at TEXT NOT NULL,
                    query_input TEXT NOT NULL,
                    protein_id TEXT NOT NULL,
                    gene_symbol TEXT NOT NULL,
                    cancer_probability REAL NOT NULL,
                    composite_centrality REAL NOT NULL,
                    is_known_cancer INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS centrality_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    top_protein TEXT NOT NULL,
                    top_score REAL NOT NULL
                )
                """
            )
            conn.commit()

    def log_prediction_query(
        self,
        query_input: str,
        protein_id: str,
        gene_symbol: str,
        cancer_probability: float,
        composite_centrality: float,
        is_known_cancer: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_queries
                (queried_at, query_input, protein_id, gene_symbol, cancer_probability, composite_centrality, is_known_cancer)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    query_input,
                    protein_id,
                    gene_symbol,
                    float(cancer_probability),
                    float(composite_centrality),
                    1 if is_known_cancer else 0,
                ),
            )
            conn.commit()

    def get_recent_queries(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT queried_at, query_input, protein_id, gene_symbol,
                       cancer_probability, composite_centrality, is_known_cancer
                FROM prediction_queries
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def log_centrality_experiment(self, weights: Dict[str, float], top_protein: str, top_score: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO centrality_experiments (created_at, weights_json, top_protein, top_score)
                VALUES (?, ?, ?, ?)
                """,
                (now, json.dumps(weights), top_protein, float(top_score)),
            )
            conn.commit()
