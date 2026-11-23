"""Phase 5 & 10: Unified Protein Prediction Engine & Predictions CSV Export.

Supports querying any protein by Ensembl Protein ID (9606.ENSP... or ENSP...)
or HGNC Gene Symbol (e.g., TP53, BRCA1, EGFR), returning:
- Calibrated cancer association probability & predicted class
- Network importance (composite centrality score & rank) — clearly separated from probability
- All 8 centrality metrics + percentiles
- Local SHAP feature attributions
- Top 1-hop interacting neighbors in the PPI graph
- Clean error handling for invalid/missing protein IDs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .centrality import CORE_CENTRALITY_COLUMNS
from .config import PROJECT_ROOT, get_project_dirs


class ProteinNotFoundError(KeyError):
    """Raised when a queried protein ID or gene symbol is invalid or absent from the PPI network."""

    def __str__(self) -> str:
        # KeyError.__str__ wraps the message in extra quotes; return the plain message instead.
        return str(self.args[0]) if self.args else "Protein not found."


def export_protein_predictions(
    proteins_df: pd.DataFrame,
    centrality_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    all_model_probabilities: Dict[str, np.ndarray],
    best_model_name: str,
    best_threshold: float,
    split_labels: List[str],
    output_csv: Path,
) -> pd.DataFrame:
    """Export complete per-protein predictions table to results/predictions/protein_predictions.csv."""
    # Align all tables on protein_id
    base = centrality_df[
        [
            "protein_id",
            "gene_symbol",
            "cancer_label",
            "cosmic_tier",
            "role_in_cancer",
            "degree",
            "weighted_degree",
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "pagerank",
            "clustering_coefficient",
            "k_core",
            "network_constraint",
        ]
    ].copy()

    p_meta = proteins_df.set_index("protein_id").loc[base["protein_id"]]
    base["protein_size"] = p_meta["protein_size"].values
    base["annotation"] = p_meta["raw_annotation"].values
    base["split"] = split_labels

    rk = rankings_df.set_index("protein_id").loc[base["protein_id"]]
    base["composite_centrality_score"] = rk["composite_centrality_score"].values
    base["centrality_rank"] = rk["centrality_rank"].values
    base["cancer_probability"] = rk["cancer_association_probability"].values
    base["model_prediction_rank"] = rk["model_prediction_rank"].values
    base["predicted_label"] = (base["cancer_probability"] >= best_threshold).astype(int)
    base["predicted_class"] = np.where(base["predicted_label"] == 1, "Cancer-Associated", "Non-Cancer")
    base["confidence_tier"] = pd.cut(
        base["cancer_probability"],
        bins=[-0.01, 0.2, 0.5, 0.75, 1.01],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)
    base["best_model"] = best_model_name

    # Include key individual model probabilities for transparency
    for mname, probs in all_model_probabilities.items():
        safe_col = "prob_" + "".join(ch if ch.isalnum() else "_" for ch in mname.lower()).strip("_")
        base[safe_col] = np.round(probs, 5)

    # Precompute centrality percentiles (0-100) for clear UI interpretation
    for col in CORE_CENTRALITY_COLUMNS:
        base[f"{col}_pct"] = np.round(base[col].rank(pct=True) * 100.0, 2)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_csv, index=False)
    return base


class CancerProteinPredictor:
    """Fast production predictor serving cached model outputs, PPI subgraph, and SHAP explanations."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.dirs = get_project_dirs(project_root)
        pred_path = self.dirs["results_predictions"] / "protein_predictions.csv"
        edges_path = self.dirs["data_processed"] / "ppi_edges.csv"
        shap_path = self.dirs["data_processed"] / "protein_shap_explanations.json"

        if not pred_path.exists():
            raise FileNotFoundError(
                f"Predictions file not found at {pred_path}. Run `python main.py` first."
            )

        self.df = pd.read_csv(pred_path, dtype={"protein_id": str, "gene_symbol": str})
        self.df["annotation"] = self.df["annotation"].fillna("")
        self.df["role_in_cancer"] = self.df["role_in_cancer"].fillna("non_cancer")

        # Fast lookup indices: exact protein_id, stripped ENSP ID, and uppercase gene_symbol
        self._by_id: Dict[str, int] = {}
        self._by_symbol: Dict[str, int] = {}
        for idx, (pid, sym) in enumerate(zip(self.df["protein_id"], self.df["gene_symbol"])):
            self._by_id[pid.upper()] = idx
            if "." in pid:
                self._by_id[pid.split(".", 1)[1].upper()] = idx
            if isinstance(sym, str) and sym.strip():
                self._by_symbol.setdefault(sym.strip().upper(), idx)

        # Load PPI adjacency list for instant neighbor lookups & ego-network serving
        self.edges_df = pd.read_csv(edges_path, dtype={"protein1": str, "protein2": str})
        self.G = nx.Graph()
        for row in self.df.itertuples(index=False):
            self.G.add_node(
                row.protein_id,
                gene_symbol=row.gene_symbol,
                cancer_label=int(row.cancer_label),
                cancer_probability=float(row.cancer_probability),
                composite_centrality_score=float(row.composite_centrality_score),
                degree=int(row.degree),
            )
        self.G.add_edges_from(
            (r.protein1, r.protein2, {"weight": float(r.weight), "combined_score": int(r.combined_score)})
            for r in self.edges_df.itertuples(index=False)
        )

        # Load local SHAP explanations
        self.shap_store: Dict[str, Any] = {}
        self.shap_meta: Dict[str, Any] = {}
        if shap_path.exists():
            with shap_path.open("r", encoding="utf-8") as fh:
                raw_shap = json.load(fh)
            self.shap_store = raw_shap.get("explanations", {})
            self.shap_meta = {
                "method": raw_shap.get("method", "TreeSHAP"),
                "base_value": raw_shap.get("base_value", 0.0),
                "disclaimer": raw_shap.get("disclaimer", ""),
                "category_importance_pct": raw_shap.get("category_importance_pct", []),
            }

    def resolve_query(self, query: str) -> int:
        """Resolve protein ID, ENSP ID, or HGNC gene symbol to row index."""
        if not isinstance(query, str) or not query.strip():
            raise ProteinNotFoundError("Protein query must be a non-empty string.")
        q = query.strip().upper()
        if q in self._by_id:
            return self._by_id[q]
        if q in self._by_symbol:
            return self._by_symbol[q]
        raise ProteinNotFoundError(
            f"Protein '{query}' was not found in the PPI network. "
            "Try a valid HGNC gene symbol (e.g., TP53, BRCA1, EGFR) or STRING ID (e.g., 9606.ENSP00000269305)."
        )

    def search_proteins(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Case-insensitive prefix/substring search across gene_symbol and protein_id."""
        q = (query or "").strip().upper()
        if not q:
            sub = self.df.sort_values("cancer_probability", ascending=False).head(limit)
        else:
            sym_u = self.df["gene_symbol"].fillna("").str.upper()
            id_u = self.df["protein_id"].fillna("").str.upper()
            exact = (sym_u == q) | (id_u == q)
            starts = sym_u.str.startswith(q) | id_u.str.contains(q, regex=False)
            sub = self.df[exact | starts].copy()
            sub["_rank"] = np.where(exact[exact | starts], 0, 1)
            sub = sub.sort_values(["_rank", "cancer_probability"], ascending=[True, False]).head(limit)

        return [
            {
                "protein_id": r.protein_id,
                "gene_symbol": r.gene_symbol,
                "cancer_probability": round(float(r.cancer_probability), 4),
                "composite_centrality_score": round(float(r.composite_centrality_score), 4),
                "is_known_cancer": bool(r.cancer_label == 1),
                "degree": int(r.degree),
            }
            for r in sub.itertuples(index=False)
        ]

    def predict_protein(self, query: str, max_neighbors: int = 25) -> Dict[str, Any]:
        """Return full prediction, centrality profile, SHAP explanation, and top PPI neighbors."""
        idx = self.resolve_query(query)
        row = self.df.iloc[idx]
        pid = str(row["protein_id"])

        # Top interacting partners sorted by interaction weight & cancer probability
        neighbors: List[Dict[str, Any]] = []
        if pid in self.G:
            for nbr in self.G.neighbors(pid):
                edata = self.G.get_edge_data(pid, nbr) or {}
                ndata = self.G.nodes[nbr]
                neighbors.append(
                    {
                        "protein_id": nbr,
                        "gene_symbol": ndata.get("gene_symbol", nbr),
                        "interaction_weight": round(float(edata.get("weight", 1.0)), 4),
                        "combined_score": int(edata.get("combined_score", 700)),
                        "is_known_cancer": bool(ndata.get("cancer_label", 0) == 1),
                        "cancer_probability": round(float(ndata.get("cancer_probability", 0.0)), 4),
                        "degree": int(ndata.get("degree", 0)),
                    }
                )
            neighbors.sort(key=lambda x: (x["interaction_weight"], x["cancer_probability"]), reverse=True)

        known_cancer_neighbors = sum(1 for n in neighbors if n["is_known_cancer"])
        local_shap = self.shap_store.get(pid, {"base_value": 0.0, "sum_shap": 0.0, "top_contributors": []})

        centralities = {
            col: {
                "value": round(float(row[col]), 6),
                "percentile": round(float(row[f"{col}_pct"]), 2),
            }
            for col in CORE_CENTRALITY_COLUMNS
        }

        return {
            "protein_id": pid,
            "gene_symbol": str(row["gene_symbol"]),
            "annotation": str(row["annotation"]) or "No functional annotation available.",
            "protein_size": int(row["protein_size"]),
            "known_label": {
                "is_known_cancer": bool(int(row["cancer_label"]) == 1),
                "cosmic_tier": int(row["cosmic_tier"]),
                "role_in_cancer": str(row["role_in_cancer"]),
                "dataset_split": str(row["split"]),
            },
            "prediction": {
                "cancer_association_probability": round(float(row["cancer_probability"]), 4),
                "predicted_class": str(row["predicted_class"]),
                "confidence_tier": str(row["confidence_tier"]),
                "model_prediction_rank": int(row["model_prediction_rank"]),
                "total_proteins": int(len(self.df)),
                "best_model": str(row["best_model"]),
            },
            "network_importance": {
                "composite_centrality_score": round(float(row["composite_centrality_score"]), 4),
                "centrality_rank": int(row["centrality_rank"]),
                "degree": int(row["degree"]),
                "weighted_degree": round(float(row["weighted_degree"]), 3),
                "known_cancer_neighbors_count": int(known_cancer_neighbors),
                "total_neighbors_count": int(len(neighbors)),
            },
            "centralities": centralities,
            "shap_explanation": local_shap,
            "neighbors": neighbors[:max_neighbors],
            "scientific_disclaimer": (
                "Predicted probability reflects network-structural and functional association with known "
                "COSMIC cancer genes. It is intended for candidate prioritization and is NOT proof of cancer causation."
            ),
        }

    def get_ego_subgraph(self, query: str, max_nodes: int = 35) -> Dict[str, Any]:
        """Return an interactive 1-hop ego-network subgraph centered on a protein."""
        idx = self.resolve_query(query)
        center_id = str(self.df.iloc[idx]["protein_id"])

        nbrs = list(self.G.neighbors(center_id))
        nbrs.sort(
            key=lambda u: (
                self.G.get_edge_data(center_id, u).get("weight", 0.0),
                self.G.nodes[u].get("cancer_probability", 0.0),
            ),
            reverse=True,
        )
        selected = [center_id] + nbrs[: max(1, max_nodes - 1)]
        subG = self.G.subgraph(selected)
        pos = nx.spring_layout(subG, seed=42, k=0.6, iterations=50)

        nodes_out = []
        for u in subG.nodes():
            nd = subG.nodes[u]
            x, y = pos[u]
            nodes_out.append(
                {
                    "id": u,
                    "gene_symbol": nd.get("gene_symbol", u),
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "is_center": u == center_id,
                    "is_known_cancer": bool(nd.get("cancer_label", 0) == 1),
                    "cancer_probability": round(float(nd.get("cancer_probability", 0.0)), 4),
                    "composite_centrality_score": round(float(nd.get("composite_centrality_score", 0.0)), 4),
                    "degree": int(nd.get("degree", 0)),
                }
            )
        edges_out = [
            {
                "source": u,
                "target": v,
                "weight": round(float(d.get("weight", 1.0)), 4),
                "combined_score": int(d.get("combined_score", 700)),
            }
            for u, v, d in subG.edges(data=True)
        ]
        return {"center_id": center_id, "nodes": nodes_out, "edges": edges_out}
