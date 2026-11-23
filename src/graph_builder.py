"""Phase 3: PPI Graph Construction, Connected Component Analysis, Statistics & Visualizations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def build_ppi_graph(
    proteins_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> nx.Graph:
    """Construct an undirected weighted NetworkX PPI graph G = (V, E, w)."""
    G = nx.Graph(name="Human_PPI_Network")

    # Add nodes with biological attributes
    for row in proteins_df.itertuples(index=False):
        G.add_node(
            row.protein_id,
            gene_symbol=row.gene_symbol,
            is_cancer=int(row.is_cancer),
            cosmic_tier=int(getattr(row, "cosmic_tier", 0)),
            role_in_cancer=str(getattr(row, "role_in_cancer", "non_cancer")),
            protein_size=int(getattr(row, "protein_size", 0)),
            has_annotation=int(getattr(row, "has_annotation", 0)),
        )

    # Add weighted undirected edges (guaranteeing no self-loops)
    edge_tuples = [
        (row.protein1, row.protein2, {"weight": float(row.weight), "combined_score": int(row.combined_score)})
        for row in edges_df.itertuples(index=False)
        if row.protein1 != row.protein2
    ]
    G.add_edges_from(edge_tuples)

    # Extra safety: strip any self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def compute_network_statistics(
    G: nx.Graph,
    preprocessing_report: Dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Compute comprehensive biological PPI network topology statistics."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    weighted_degrees = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)

    isolated_nodes = list(nx.isolates(G))
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc_size = len(components[0]) if components else 0
    comp_sizes = [len(c) for c in components]

    cancer_degrees = np.array(
        [G.degree(n) for n, attr in G.nodes(data=True) if attr.get("is_cancer", 0) == 1],
        dtype=float,
    )
    non_cancer_degrees = np.array(
        [G.degree(n) for n, attr in G.nodes(data=True) if attr.get("is_cancer", 0) == 0],
        dtype=float,
    )

    avg_clustering = float(nx.average_clustering(G)) if n_nodes > 0 else 0.0
    transitivity = float(nx.transitivity(G)) if n_nodes > 0 else 0.0

    # Degree distribution histogram bins (for dashboard plots)
    hist_counts, hist_edges = np.histogram(degrees, bins=[1, 2, 5, 10, 20, 50, 100, 250, 500, int(max(degrees.max() + 1, 501))])
    degree_bins = [
        {"range": f"{int(hist_edges[i])}-{int(hist_edges[i+1])-1}", "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    stats: Dict[str, Any] = {
        "num_proteins": int(n_nodes),
        "num_interactions": int(n_edges),
        "isolated_proteins_in_graph": int(len(isolated_nodes)),
        "network_density": round(float(nx.density(G)), 6),
        "average_degree": round(float(degrees.mean()), 4) if n_nodes else 0.0,
        "median_degree": round(float(np.median(degrees)), 4) if n_nodes else 0.0,
        "std_degree": round(float(degrees.std()), 4) if n_nodes else 0.0,
        "max_degree": int(degrees.max()) if n_nodes else 0,
        "p95_degree": round(float(np.percentile(degrees, 95)), 2) if n_nodes else 0.0,
        "average_weighted_degree": round(float(weighted_degrees.mean()), 4) if n_nodes else 0.0,
        "average_edge_weight": round(float(weights.mean()), 4) if len(weights) else 0.0,
        "connected_components": int(len(components)),
        "giant_component_size": int(gcc_size),
        "giant_component_fraction": round(gcc_size / max(n_nodes, 1), 6),
        "top_5_component_sizes": comp_sizes[:5],
        "average_clustering_coefficient": round(avg_clustering, 6),
        "global_transitivity": round(transitivity, 6),
        "cancer_protein_count": int(len(cancer_degrees)),
        "non_cancer_protein_count": int(len(non_cancer_degrees)),
        "cancer_avg_degree": round(float(cancer_degrees.mean()), 4) if len(cancer_degrees) else 0.0,
        "cancer_median_degree": round(float(np.median(cancer_degrees)), 4) if len(cancer_degrees) else 0.0,
        "non_cancer_avg_degree": round(float(non_cancer_degrees.mean()), 4) if len(non_cancer_degrees) else 0.0,
        "non_cancer_median_degree": round(float(np.median(non_cancer_degrees)), 4) if len(non_cancer_degrees) else 0.0,
        "degree_distribution_bins": degree_bins,
    }

    if preprocessing_report:
        stats["preprocessing"] = preprocessing_report

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    return stats


def generate_network_visualizations(
    G: nx.Graph,
    figures_dir: Path,
) -> List[Path]:
    """Generate publication-quality degree distribution and top-hub PPI subgraph figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # 1. Log-Log Degree Distribution + Cancer vs Non-Cancer Degree Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    all_deg = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(all_deg, return_counts=True)
    axes[0].loglog(vals, counts, "o", markersize=4, alpha=0.75, color="#2563eb")
    axes[0].set_title("PPI Network Degree Distribution (Log-Log)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Degree (k)")
    axes[0].set_ylabel("Number of Proteins P(k)")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.3)

    c_deg = [G.degree(n) for n, a in G.nodes(data=True) if a.get("is_cancer", 0) == 1]
    nc_deg = [G.degree(n) for n, a in G.nodes(data=True) if a.get("is_cancer", 0) == 0]
    bins = np.logspace(0, np.log10(max(all_deg.max(), 10)), 30)
    axes[1].hist(nc_deg, bins=bins, density=True, alpha=0.6, color="#64748b", label=f"Non-Cancer (n={len(nc_deg):,})")
    axes[1].hist(c_deg, bins=bins, density=True, alpha=0.65, color="#dc2626", label=f"COSMIC Cancer (n={len(c_deg):,})")
    axes[1].set_xscale("log")
    axes[1].set_title("Degree Distribution: Cancer vs Non-Cancer Proteins", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Degree (log scale)")
    axes[1].set_ylabel("Normalized Density")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    deg_fig_path = figures_dir / "degree_distribution.png"
    fig.savefig(deg_fig_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(deg_fig_path)

    # 2. Top-60 Hub Protein Subgraph Visualization
    top_nodes = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:60]]
    subG = G.subgraph(top_nodes).copy()

    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    pos = nx.spring_layout(subG, seed=42, k=0.45, iterations=60)
    node_colors = ["#dc2626" if subG.nodes[n].get("is_cancer", 0) == 1 else "#3b82f6" for n in subG.nodes()]
    node_sizes = [80 + 1.2 * G.degree(n) for n in subG.nodes()]
    labels = {n: subG.nodes[n].get("gene_symbol", n) for n in subG.nodes()}

    nx.draw_networkx_edges(subG, pos, alpha=0.18, width=0.6, edge_color="#94a3b8", ax=ax)
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, linewidths=0.8, edgecolors="white", ax=ax)
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=7, font_weight="bold", ax=ax)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Known Cancer Protein (COSMIC)", markerfacecolor="#dc2626", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Non-Cancer / Unlabeled Protein", markerfacecolor="#3b82f6", markersize=9),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)
    ax.set_title("Top-60 Highest-Degree PPI Hub Subgraph", fontsize=13, fontweight="bold")
    ax.axis("off")

    hub_fig_path = figures_dir / "ppi_hub_subgraph.png"
    fig.savefig(hub_fig_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(hub_fig_path)

    return saved_paths
