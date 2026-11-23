"""Data cleaning, undirected edge deduplication, self-loop removal, and leak-free text sanitization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data_loader import RawDatasetBundle


# Patterns removed from functional annotation strings to prevent label leakage
CANCER_LEAK_PATTERNS = [
    r"\bcancer\w*\b",
    r"\btumou?r\w*\b",
    r"\boncogen\w*\b",
    r"\bproto-?oncogen\w*\b",
    r"\bcarcinom\w*\b",
    r"\bleukem\w*\b",
    r"\bleukaem\w*\b",
    r"\blymphom\w*\b",
    r"\bmelanom\w*\b",
    r"\bsarcom\w*\b",
    r"\bneoplas\w*\b",
    r"\bmalignan\w*\b",
    r"\bmetasta\w*\b",
    r"\bgliom\w*\b",
    r"\bmyelom\w*\b",
    r"\bblastom\w*\b",
    r"\bmesotheliom\w*\b",
    r"\bteratom\w*\b",
    r"\badenom\w*\b",
    r"\bpapillom\w*\b",
    r"\bchemotherapy\b",
]
_LEAK_REGEX = re.compile("|".join(CANCER_LEAK_PATTERNS), flags=re.IGNORECASE)
_SPACE_REGEX = re.compile(r"\s+")


@dataclass
class CleanedDataset:
    """Cleaned protein table and deduplicated undirected PPI edge table."""

    proteins_df: pd.DataFrame
    edges_df: pd.DataFrame
    preprocessing_report: Dict[str, Any]


def sanitize_annotation_text(text: str, strip_cancer_terms: bool = True) -> str:
    """Clean functional annotation text and optionally strip cancer-related leak words."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = text.strip()
    if cleaned.lower() in {"annotation not available", "uncharacterized protein", "none", "nan"}:
        return ""
    if strip_cancer_terms:
        cleaned = _LEAK_REGEX.sub(" ", cleaned)
    cleaned = _SPACE_REGEX.sub(" ", cleaned).strip()
    return cleaned


def preprocess_dataset(
    bundle: RawDatasetBundle,
    cfg: Dict[str, Any],
    output_dir: Path | None = None,
) -> CleanedDataset:
    """Clean edges (remove self-loops, deduplicate reverse undirected edges) and align protein metadata."""
    prep_cfg = cfg.get("preprocessing", {})
    remove_self_loops = bool(prep_cfg.get("remove_self_loops", True))
    deduplicate_edges = bool(prep_cfg.get("deduplicate_edges", True))
    strip_leak = bool(prep_cfg.get("strip_cancer_leak_terms", True))

    edges = bundle.edges_df.copy()
    initial_edge_rows = len(edges)

    # Drop missing endpoints
    edges = edges.dropna(subset=["protein1", "protein2"]).copy()
    edges["protein1"] = edges["protein1"].astype(str).str.strip()
    edges["protein2"] = edges["protein2"].astype(str).str.strip()

    # Remove self-loops
    self_loops_removed = 0
    if remove_self_loops:
        self_mask = edges["protein1"] == edges["protein2"]
        self_loops_removed = int(self_mask.sum())
        edges = edges[~self_mask].copy()

    # Canonicalize undirected edge ordering (u < v) and deduplicate reverse edges (retaining max weight)
    duplicate_edges_removed = 0
    if deduplicate_edges and not edges.empty:
        u = np.minimum(edges["protein1"].values, edges["protein2"].values)
        v = np.maximum(edges["protein1"].values, edges["protein2"].values)
        edges["u"] = u
        edges["v"] = v
        before_dedup = len(edges)
        edges = (
            edges.groupby(["u", "v"], as_index=False)
            .agg(
                weight=("weight", "max"),
                combined_score=("combined_score", "max"),
            )
            .rename(columns={"u": "protein1", "v": "protein2"})
        )
        duplicate_edges_removed = int(before_dedup - len(edges))

    # Keep proteins that participate in the PPI network (plus record isolated proteins from info)
    edge_proteins = set(edges["protein1"]).union(set(edges["protein2"]))
    proteins = bundle.proteins_df.copy()
    proteins["in_ppi_network"] = proteins["protein_id"].isin(edge_proteins)
    isolated_in_info = int((~proteins["in_ppi_network"]).sum())

    # Retain active PPI proteins (proteins with at least 1 high-confidence interaction)
    # Note: isolated proteins from original schema (if any) can also be tracked; for STRING we keep active PPI proteins
    # to focus graph centrality analysis on the interacting proteome while logging isolated count cleanly.
    proteins = proteins[proteins["in_ppi_network"]].copy().reset_index(drop=True)

    # Handle missing functional annotations & sanitize leak terms
    proteins["raw_annotation"] = proteins["annotation"].fillna("").astype(str)
    proteins["clean_annotation"] = [
        sanitize_annotation_text(t, strip_cancer_terms=strip_leak) for t in proteins["raw_annotation"]
    ]
    proteins["has_annotation"] = (proteins["clean_annotation"].str.len() > 0).astype(int)
    proteins["annotation_token_count"] = proteins["clean_annotation"].apply(
        lambda s: len(s.split()) if s else 0
    )
    missing_annotations = int((proteins["has_annotation"] == 0).sum())

    # Ensure deterministic node ordering
    proteins = proteins.sort_values("protein_id").reset_index(drop=True)
    edges = edges.sort_values(["protein1", "protein2"]).reset_index(drop=True)

    n_cancer = int((proteins["is_cancer"] == 1).sum())
    n_non_cancer = int((proteins["is_cancer"] == 0).sum())

    report: Dict[str, Any] = {
        "dataset_name": bundle.dataset_name,
        "dataset_type": bundle.dataset_type,
        "initial_filtered_edge_rows": int(initial_edge_rows),
        "self_loops_removed": int(self_loops_removed),
        "duplicate_reverse_edges_removed": int(duplicate_edges_removed),
        "final_undirected_edges": int(len(edges)),
        "isolated_proteins_excluded_from_info": int(isolated_in_info),
        "final_ppi_proteins": int(len(proteins)),
        "cancer_proteins": n_cancer,
        "non_cancer_proteins": n_non_cancer,
        "cancer_prevalence": round(n_cancer / max(len(proteins), 1), 6),
        "missing_functional_annotations": missing_annotations,
        "annotated_protein_fraction": round(1.0 - missing_annotations / max(len(proteins), 1), 4),
        "cancer_leak_terms_stripped": strip_leak,
        "provenance": bundle.provenance,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        proteins.to_csv(output_dir / "proteins_metadata.csv", index=False)
        edges.to_csv(output_dir / "ppi_edges.csv", index=False)

    return CleanedDataset(
        proteins_df=proteins,
        edges_df=edges,
        preprocessing_report=report,
    )
