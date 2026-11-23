"""Dual-schema biological dataset loader.

Supports:
1. Primary public biological dataset: STRING v12.5 Homo sapiens PPI + COSMIC Cancer Gene Census v104.
2. Original reference repository schema: HumanPPI.txt, Functions.txt, Cancers.txt, Test1.txt, Test2.txt
   (from sevmardi/Protein-Protein-Networks, cleanly reimplemented in Python 3).
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .config import PROJECT_ROOT, resolve_path


@dataclass
class RawDatasetBundle:
    """Standardized container for raw loaded PPI, metadata, annotations, and cancer labels."""

    dataset_name: str
    dataset_type: str
    edges_df: pd.DataFrame          # columns: [protein1, protein2, weight]
    proteins_df: pd.DataFrame       # columns: [protein_id, gene_symbol, protein_size, annotation, is_cancer, cosmic_tier, role_in_cancer]
    functions_map: Dict[str, List[str]] = field(default_factory=dict)  # protein_id -> list of function tokens/IDs
    test1_ids: Optional[List[str]] = None
    test2_ids: Optional[List[str]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def _open_text_maybe_gz(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _find_case_insensitive(directory: Path, filename: str) -> Optional[Path]:
    direct = directory / filename
    if direct.exists():
        return direct
    target_lower = filename.lower()
    if directory.exists():
        for entry in directory.iterdir():
            if entry.name.lower() == target_lower:
                return entry
    return None


def load_string_cosmic_dataset(cfg: Dict[str, Any]) -> RawDatasetBundle:
    """Load STRING v12.5 Homo sapiens PPI & protein info paired with COSMIC Cancer Gene Census v104."""
    base_dir = Path(cfg.get("_project_root", PROJECT_ROOT))
    links_path = resolve_path(cfg["paths"]["string_links"], base_dir)
    info_path = resolve_path(cfg["paths"]["string_info"], base_dir)
    cosmic_path = resolve_path(cfg["paths"]["cosmic_census"], base_dir)

    for label, p in [("STRING links", links_path), ("STRING info", info_path), ("COSMIC census", cosmic_path)]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} file not found at {p}. "
                "See data/external/DATASET_PROVENANCE.md for download instructions."
            )

    score_threshold = int(cfg["preprocessing"].get("score_threshold", 700))
    include_tiers = set(int(t) for t in cfg["preprocessing"].get("include_cosmic_tiers", [1, 2]))

    # 1. Load STRING protein info (ENSP ID -> gene symbol, size, functional annotation)
    info_df = pd.read_csv(
        info_path,
        sep="\t",
        compression="infer",
        dtype=str,
        keep_default_na=False,
    )
    info_df.columns = [c.lstrip("#").strip() for c in info_df.columns]
    # Standard STRING columns: string_protein_id, preferred_name, protein_size, annotation
    id_col = info_df.columns[0]
    info_df = info_df.rename(
        columns={
            id_col: "protein_id",
            "preferred_name": "gene_symbol",
            "protein_size": "protein_size",
            "annotation": "annotation",
        }
    )
    info_df["protein_id"] = info_df["protein_id"].str.strip()
    info_df["gene_symbol"] = info_df["gene_symbol"].str.strip()
    info_df["gene_symbol_upper"] = info_df["gene_symbol"].str.upper()
    info_df["protein_size"] = pd.to_numeric(info_df["protein_size"], errors="coerce").fillna(0).astype(int)
    info_df["annotation"] = info_df["annotation"].fillna("").str.strip()

    # 2. Load COSMIC Cancer Gene Census (gold-standard cancer driver genes)
    cosmic_df = pd.read_csv(
        cosmic_path,
        sep="\t",
        compression="infer",
        dtype=str,
        keep_default_na=False,
    )
    cosmic_df.columns = [c.strip().upper() for c in cosmic_df.columns]
    if "TIER" in cosmic_df.columns:
        cosmic_df["TIER_INT"] = pd.to_numeric(cosmic_df["TIER"], errors="coerce").fillna(0).astype(int)
        cosmic_df = cosmic_df[cosmic_df["TIER_INT"].isin(include_tiers)].copy()
    else:
        cosmic_df["TIER_INT"] = 1

    primary_cosmic: Dict[str, Tuple[int, str]] = {}
    synonym_cosmic: Dict[str, Tuple[int, str]] = {}
    for _, row in cosmic_df.iterrows():
        sym = str(row.get("GENE_SYMBOL", "")).strip().upper()
        tier = int(row.get("TIER_INT", 1))
        role = str(row.get("ROLE_IN_CANCER", "")).strip() or "unspecified"
        if sym:
            primary_cosmic[sym] = (tier, role)
        syns_raw = str(row.get("SYNONYMS", "")).strip()
        if syns_raw:
            for syn in syns_raw.split(","):
                s = syn.strip().upper()
                if s and s not in primary_cosmic and s not in synonym_cosmic:
                    synonym_cosmic[s] = (tier, role)

    # Map COSMIC labels to STRING proteins (primary symbol first, unambiguous synonym fallback)
    is_cancer_list: List[int] = []
    tier_list: List[int] = []
    role_list: List[str] = []
    matched_by_primary = 0
    matched_by_synonym = 0
    for sym_u in info_df["gene_symbol_upper"]:
        if sym_u in primary_cosmic:
            tier, role = primary_cosmic[sym_u]
            is_cancer_list.append(1)
            tier_list.append(tier)
            role_list.append(role)
            matched_by_primary += 1
        elif sym_u in synonym_cosmic:
            tier, role = synonym_cosmic[sym_u]
            is_cancer_list.append(1)
            tier_list.append(tier)
            role_list.append(role)
            matched_by_synonym += 1
        else:
            is_cancer_list.append(0)
            tier_list.append(0)
            role_list.append("non_cancer")

    info_df["is_cancer"] = is_cancer_list
    info_df["cosmic_tier"] = tier_list
    info_df["role_in_cancer"] = role_list

    # 3. Stream & filter STRING protein links by combined_score >= score_threshold
    valid_proteins: Set[str] = set(info_df["protein_id"])
    edge_chunks: List[pd.DataFrame] = []
    raw_edge_count = 0

    for chunk in pd.read_csv(
        links_path,
        sep=r"\s+",
        compression="infer",
        chunksize=500_000,
        dtype={"protein1": str, "protein2": str, "combined_score": int},
    ):
        raw_edge_count += len(chunk)
        filtered = chunk[chunk["combined_score"] >= score_threshold]
        if not filtered.empty:
            filtered = filtered[
                filtered["protein1"].isin(valid_proteins) & filtered["protein2"].isin(valid_proteins)
            ]
            if not filtered.empty:
                edge_chunks.append(filtered)

    if not edge_chunks:
        raise ValueError(
            f"No STRING edges remained after filtering combined_score >= {score_threshold}."
        )
    edges_df = pd.concat(edge_chunks, ignore_index=True)
    edges_df["weight"] = edges_df["combined_score"].astype(float) / 1000.0
    edges_df = edges_df[["protein1", "protein2", "weight", "combined_score"]]

    proteins_df = info_df[
        ["protein_id", "gene_symbol", "protein_size", "annotation", "is_cancer", "cosmic_tier", "role_in_cancer"]
    ].drop_duplicates(subset=["protein_id"]).reset_index(drop=True)

    provenance = {
        "dataset_name": cfg.get("dataset_name", "STRING_v12.5_COSMIC_v104"),
        "dataset_type": "string_cosmic",
        "string_links_file": links_path.name,
        "string_info_file": info_path.name,
        "cosmic_file": cosmic_path.name,
        "score_threshold": score_threshold,
        "raw_string_edge_rows": int(raw_edge_count),
        "filtered_edge_rows_before_dedup": int(len(edges_df)),
        "total_string_proteins_in_info": int(len(proteins_df)),
        "cosmic_census_genes_loaded": int(len(cosmic_df)),
        "cosmic_matched_primary": int(matched_by_primary),
        "cosmic_matched_synonym": int(matched_by_synonym),
        "attribution": cfg.get("attribution", {}),
    }

    return RawDatasetBundle(
        dataset_name=provenance["dataset_name"],
        dataset_type="string_cosmic",
        edges_df=edges_df,
        proteins_df=proteins_df,
        provenance=provenance,
    )


def load_original_schema_dataset(cfg: Dict[str, Any]) -> RawDatasetBundle:
    """Clean Python 3 loader for the original sevmardi/Protein-Protein-Networks schema.

    Expects: HumanPPI.txt (or humanPPI.txt), Functions.txt, Cancers.txt (or Cancer.txt),
    and optional Test1.txt / Test2.txt inside cfg['paths']['data_dir'].
    """
    base_dir = Path(cfg.get("_project_root", PROJECT_ROOT))
    data_dir = resolve_path(cfg["paths"]["data_dir"], base_dir)

    ppi_path = _find_case_insensitive(data_dir, cfg["paths"].get("human_ppi", "HumanPPI.txt"))
    fn_path = _find_case_insensitive(data_dir, cfg["paths"].get("functions", "Functions.txt"))
    cancer_path = (
        _find_case_insensitive(data_dir, cfg["paths"].get("cancers", "Cancers.txt"))
        or _find_case_insensitive(data_dir, "Cancer.txt")
    )
    test1_path = _find_case_insensitive(data_dir, cfg["paths"].get("test1", "Test1.txt"))
    test2_path = _find_case_insensitive(data_dir, cfg["paths"].get("test2", "Test2.txt"))

    if not ppi_path or not cancer_path:
        raise FileNotFoundError(
            f"Original schema files not found in {data_dir}. "
            "Note: The sevmardi/Protein-Protein-Networks dataset is private and not publicly distributed. "
            "Use config/dataset_string_cosmic.yaml for the public STRING v12.5 + COSMIC v104 dataset."
        )

    # Load cancer protein IDs
    cancer_set: Set[str] = set()
    with _open_text_maybe_gz(cancer_path) as fh:
        for line in fh:
            token = line.strip()
            if token:
                cancer_set.add(token)

    # Load PPI edges
    edge_rows: List[Tuple[str, str, float, int]] = []
    all_proteins: Set[str] = set(cancer_set)
    with _open_text_maybe_gz(ppi_path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if len(parts) >= 2:
                p1, p2 = parts[0], parts[1]
                w = float(parts[2]) if len(parts) >= 3 else 1.0
                edge_rows.append((p1, p2, w, int(round(w * 1000))))
                all_proteins.add(p1)
                all_proteins.add(p2)

    edges_df = pd.DataFrame(edge_rows, columns=["protein1", "protein2", "weight", "combined_score"])

    # Load functional annotations (optional proteins may have no annotations)
    functions_map: Dict[str, List[str]] = {}
    if fn_path and fn_path.exists():
        with _open_text_maybe_gz(fn_path) as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(",") if p.strip()]
                if len(parts) >= 2:
                    p, fn = parts[0], parts[1]
                    all_proteins.add(p)
                    functions_map.setdefault(p, []).append(fn)

    # Load optional Test1 & Test2
    test1_ids: Optional[List[str]] = None
    if test1_path and test1_path.exists():
        test1_ids = []
        with _open_text_maybe_gz(test1_path) as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(",") if p.strip()]
                if len(parts) >= 2:
                    p, lbl = parts[0], parts[1].lower()
                    all_proteins.add(p)
                    test1_ids.append(p)
                    if lbl == "cancer":
                        cancer_set.add(p)

    test2_ids: Optional[List[str]] = None
    if test2_path and test2_path.exists():
        test2_ids = []
        with _open_text_maybe_gz(test2_path) as fh:
            for line in fh:
                token = line.strip().split(",")[0].strip()
                if token:
                    all_proteins.add(token)
                    test2_ids.append(token)

    protein_rows = []
    for pid in sorted(all_proteins):
        fns = functions_map.get(pid, [])
        ann_str = " ".join(fns) if fns else ""
        is_c = 1 if pid in cancer_set else 0
        protein_rows.append(
            {
                "protein_id": pid,
                "gene_symbol": pid,
                "protein_size": 0,
                "annotation": ann_str,
                "is_cancer": is_c,
                "cosmic_tier": 1 if is_c else 0,
                "role_in_cancer": "cancer" if is_c else "non_cancer",
            }
        )

    proteins_df = pd.DataFrame(protein_rows)
    provenance = {
        "dataset_name": cfg.get("dataset_name", "Original_PPI_Schema"),
        "dataset_type": "original_schema",
        "data_dir": str(data_dir),
        "raw_edge_rows": int(len(edges_df)),
        "total_proteins": int(len(proteins_df)),
        "annotated_proteins": int(len(functions_map)),
        "attribution": cfg.get("attribution", {}),
    }

    return RawDatasetBundle(
        dataset_name=provenance["dataset_name"],
        dataset_type="original_schema",
        edges_df=edges_df,
        proteins_df=proteins_df,
        functions_map=functions_map,
        test1_ids=test1_ids,
        test2_ids=test2_ids,
        provenance=provenance,
    )


def load_dataset(cfg: Dict[str, Any]) -> RawDatasetBundle:
    """Dispatch to the configured dataset loader."""
    dtype = cfg.get("dataset_type", "string_cosmic")
    if dtype == "string_cosmic":
        return load_string_cosmic_dataset(cfg)
    if dtype == "original_schema":
        return load_original_schema_dataset(cfg)
    raise ValueError(f"Unsupported dataset_type: {dtype}")
