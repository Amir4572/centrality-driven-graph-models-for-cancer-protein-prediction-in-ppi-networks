"""Phase 5: PyTorch Graph Neural Networks (GCN, GAT, GraphSAGE) & Centrality-Driven Fusion GNN.

Implemented natively in PyTorch using sparse adjacency tensors & vectorized edge attention
(zero dependency on fragile C++ PyTorch Geometric wheels on Windows / Python 3.14).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


@dataclass
class GNNTrainingResult:
    """Trained GNN outputs: probabilities, learned node embeddings, history, and checkpoint path."""

    model_name: str
    probabilities: np.ndarray        # (N,) calibrated-range sigmoid probabilities
    embeddings: np.ndarray           # (N, hidden_dim) learned node representations
    best_val_pr_auc: float
    best_epoch: int
    checkpoint_path: str
    history: Dict[str, List[float]]


def build_graph_tensors(
    G: nx.Graph,
    nodelist: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Construct normalized sparse adjacency matrices and edge index tensors for GCN, GAT, and GraphSAGE."""
    n = len(nodelist)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="coo", dtype=np.float32)

    # Add self-loops A_tilde = A + I
    eye = sp.eye(n, format="coo", dtype=np.float32)
    A_tilde = (A + eye).tocsr()

    # Symmetric normalization for GCN: D^{-1/2} A_tilde D^{-1/2}
    deg_tilde = np.asarray(A_tilde.sum(axis=1)).ravel()
    inv_sqrt = np.power(np.maximum(deg_tilde, 1e-8), -0.5)
    D_inv_sqrt = sp.diags(inv_sqrt)
    A_sym = (D_inv_sqrt @ A_tilde @ D_inv_sqrt).tocoo()

    sym_indices = torch.tensor(np.vstack([A_sym.row, A_sym.col]), dtype=torch.long, device=device)
    sym_values = torch.tensor(A_sym.data, dtype=torch.float32, device=device)
    adj_sym = torch.sparse_coo_tensor(sym_indices, sym_values, size=(n, n), device=device).coalesce()

    # Row-normalized neighbor-only adjacency for GraphSAGE: D^{-1} A
    A_csr = A.tocsr()
    deg_raw = np.asarray(A_csr.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg_raw)
    nz = deg_raw > 0
    inv_deg[nz] = 1.0 / deg_raw[nz]
    A_row = (sp.diags(inv_deg) @ A_csr).tocoo()

    row_indices = torch.tensor(np.vstack([A_row.row, A_row.col]), dtype=torch.long, device=device)
    row_values = torch.tensor(A_row.data, dtype=torch.float32, device=device)
    adj_mean = torch.sparse_coo_tensor(row_indices, row_values, size=(n, n), device=device).coalesce()

    # Edge index with self-loops for GAT
    A_t_coo = A_tilde.tocoo()
    edge_src = torch.tensor(A_t_coo.row, dtype=torch.long, device=device)
    edge_dst = torch.tensor(A_t_coo.col, dtype=torch.long, device=device)

    return {
        "adj_sym": adj_sym,
        "adj_mean": adj_mean,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "num_nodes": n,
    }


class GCNNet(nn.Module):
    """2-Layer Graph Convolutional Network (Kipf & Welling) with LayerNorm and residual skip."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, g: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        adj = g["adj_sym"]
        h = torch.sparse.mm(adj, self.lin1(x))
        h = F.relu(self.norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h2 = torch.sparse.mm(adj, self.lin2(h))
        emb = F.relu(self.norm2(h2 + h))
        logits = self.head(F.dropout(emb, p=self.dropout * 0.5, training=self.training)).squeeze(-1)
        return logits, emb


class GraphSAGENet(nn.Module):
    """2-Layer GraphSAGE (Hamilton et al.) with ego + mean-neighborhood concatenation."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.nbr1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.nbr2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, g: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        adj = g["adj_mean"]
        nbr = torch.sparse.mm(adj, x)
        h = F.relu(self.norm1(self.self1(x) + self.nbr1(nbr)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        nbr2 = torch.sparse.mm(adj, h)
        emb = F.relu(self.norm2(self.self2(h) + self.nbr2(nbr2) + h))
        logits = self.head(F.dropout(emb, p=self.dropout * 0.5, training=self.training)).squeeze(-1)
        return logits, emb


class SparseGATLayer(nn.Module):
    """Single-head sparse Graph Attention Layer using numerically stable segment softmax."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        h = self.W(x)
        # Edge attention score e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        s_src = (h * self.a_src).sum(dim=-1)
        s_dst = (h * self.a_dst).sum(dim=-1)
        e = F.leaky_relu(s_src[edge_src] + s_dst[edge_dst], negative_slope=0.2)

        # Numerically stable softmax per destination node edge_dst
        # Subtract per-node max using scatter_reduce
        max_per_dst = torch.full((num_nodes,), -1e9, device=x.device, dtype=x.dtype)
        max_per_dst.scatter_reduce_(0, edge_dst, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_dst[edge_dst])

        denom = torch.zeros((num_nodes,), device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, edge_dst, exp_e)
        alpha = exp_e / (denom[edge_dst] + 1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Sparse attention-weighted aggregation: out_j = sum_{i->j} alpha_ij * h_i
        att_adj = torch.sparse_coo_tensor(
            torch.stack([edge_dst, edge_src], dim=0),
            alpha,
            size=(num_nodes, num_nodes),
            device=x.device,
        ).coalesce()
        return torch.sparse.mm(att_adj, h)


class GATNet(nn.Module):
    """2-Layer Multi-Head Sparse Graph Attention Network (Velickovic et al.)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, heads: int = 2, dropout: float = 0.35):
        super().__init__()
        head_dim = max(16, hidden_dim // heads)
        self.heads1 = nn.ModuleList([SparseGATLayer(in_dim, head_dim, dropout=dropout * 0.5) for _ in range(heads)])
        self.norm1 = nn.LayerNorm(head_dim * heads)
        self.gat2 = SparseGATLayer(head_dim * heads, hidden_dim, dropout=dropout * 0.5)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, g: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst, n = g["edge_src"], g["edge_dst"], g["num_nodes"]
        h = torch.cat([layer(x, src, dst, n) for layer in self.heads1], dim=-1)
        h = F.elu(self.norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h2 = self.gat2(h, src, dst, n)
        emb = F.elu(self.norm2(h2 + self.skip(x)))
        logits = self.head(F.dropout(emb, p=self.dropout * 0.5, training=self.training)).squeeze(-1)
        return logits, emb


class CentralityDrivenFusionGNN(nn.Module):
    """Proposed Centrality-Driven Graph Feature-Fusion Architecture.

    Combines three complementary branches via learned gating:
    1. Centrality Encoder (multi-layer nonlinear projection of the 11 centrality features)
    2. Functional Annotation Encoder (projection of leak-free functional embeddings)
    3. GraphSAGE + GCN Dual Message-Passing Branch over PPI topology
    """

    def __init__(
        self,
        cent_dim: int,
        func_dim: int,
        topo_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.cent_dim = cent_dim
        self.func_dim = func_dim
        self.topo_dim = topo_dim

        self.cent_enc = nn.Sequential(
            nn.Linear(cent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.func_enc = nn.Sequential(
            nn.Linear(func_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        total_in = cent_dim + func_dim + topo_dim
        self.sage_self1 = nn.Linear(total_in, hidden_dim)
        self.sage_nbr1 = nn.Linear(total_in, hidden_dim)
        self.gnn_norm1 = nn.LayerNorm(hidden_dim)

        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_norm2 = nn.LayerNorm(hidden_dim)

        # Learned modality attention gate over [cent, func, gnn]
        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, g: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cent = x[:, : self.cent_dim]
        x_func = x[:, self.cent_dim : self.cent_dim + self.func_dim]

        h_cent = self.cent_enc(x_cent)
        h_func = self.func_enc(x_func)

        # Dual neighborhood message passing (GraphSAGE layer 1 + GCN layer 2)
        nbr1 = torch.sparse.mm(g["adj_mean"], x)
        h_gnn = F.relu(self.gnn_norm1(self.sage_self1(x) + self.sage_nbr1(nbr1)))
        h_gnn = F.dropout(h_gnn, p=self.dropout, training=self.training)
        h_gnn = F.relu(self.gnn_norm2(torch.sparse.mm(g["adj_sym"], self.gcn2(h_gnn)) + h_gnn))

        concat = torch.cat([h_cent, h_func, h_gnn], dim=-1)
        weights = F.softmax(self.gate(concat), dim=-1)
        weighted = torch.cat(
            [
                h_cent * weights[:, 0:1],
                h_func * weights[:, 1:2],
                h_gnn * weights[:, 2:3],
            ],
            dim=-1,
        )
        fused_emb = self.fusion_proj(weighted + concat * 0.5)
        logits = self.classifier(fused_emb).squeeze(-1)
        return logits, fused_emb


def train_gnn_model(
    model: nn.Module,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    graph_tensors: Dict[str, torch.Tensor],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    checkpoint_path: Path,
    epochs: int = 120,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 20,
    seed: int = 42,
) -> GNNTrainingResult:
    """Train a PyTorch GNN transductively on graph topology with strict label masking.

    Loss is computed ONLY on train_idx; early stopping uses validation PR-AUC on val_idx.
    Test labels are NEVER exposed during training or model selection.
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)

    n_pos = float((y[train_idx] == 1).sum())
    n_neg = float((y[train_idx] == 0).sum())
    # Moderated pos_weight to improve PR-AUC & calibration without extreme over-prediction
    pos_weight = torch.tensor([np.sqrt(n_neg / max(n_pos, 1.0))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_ap = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_pr_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x_t, graph_tensors)
        loss = criterion(logits[train_idx_t], y_t[train_idx_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(x_t, graph_tensors)
            probs = torch.sigmoid(eval_logits).cpu().numpy()
            val_ap = float(average_precision_score(y[val_idx], probs[val_idx]))

        history["train_loss"].append(round(float(loss.item()), 5))
        history["val_pr_auc"].append(round(val_ap, 5))

        if val_ap > best_val_ap + 1e-5:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "best_val_pr_auc": best_val_ap,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    model.eval()
    with torch.no_grad():
        final_logits, final_emb = model(x_t, graph_tensors)
        final_probs = torch.sigmoid(final_logits).cpu().numpy().astype(np.float64)
        final_emb_np = final_emb.cpu().numpy().astype(np.float32)

    return GNNTrainingResult(
        model_name=model_name,
        probabilities=final_probs,
        embeddings=final_emb_np,
        best_val_pr_auc=best_val_ap,
        best_epoch=best_epoch,
        checkpoint_path=str(checkpoint_path),
        history=history,
    )
