"""Edge-feature explainers for edge-level GNN classifiers.

E-PGExplainer : a shared MLP maps the edge embeddings of the explained
                computation subgraphs to ONE feature-mask vector (class-level
                / multi-instance explanation, as in TrustGuard).
E-GNNExplainer: one learnable feature mask per explained edge; the class-wise
                importance is the mean of the per-edge masks. All edges are
                optimised jointly on a disjoint batch (equivalent to separate
                runs, but much faster).

Both return an importance vector of length F (higher = more important).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import batch_graphs


def true_positive_edges(model_pred: torch.Tensor, g, cls: int) -> np.ndarray:
    """Forward-direction test edges of class `cls` that the baseline got right."""
    m = (g.label == cls) & (model_pred == cls) & g.is_forward
    return m.nonzero().view(-1).cpu().numpy()


def _batched_subgraphs(g, edge_ids, num_hops, max_in, seed):
    rng = np.random.default_rng(seed)
    subs, tg = [], []
    for e in edge_ids:
        s, t = g.computation_subgraph(int(e), num_hops, max_in, rng)
        subs.append(s); tg.append(t)
    return subs, tg


# --------------------------------------------------------------------------
class EPGExplainer(nn.Module):
    def __init__(self, emb_dim, num_feats, coff_size=0.01, coff_ent=5e-4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, num_feats))
        self.coff_size, self.coff_ent = coff_size, coff_ent

    def feature_logits(self, z):
        return self.mlp(z).mean(0)

    @staticmethod
    def concrete(w, temp, training):
        if not training:
            return torch.sigmoid(w)
        u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
        return torch.sigmoid((torch.log(u) - torch.log(1 - u) + w) / temp)


def epg_importance(model, g, edge_ids, cfg, device, seed=0):
    """Train E-PGExplainer on the given edges and return feature importance (F,)."""
    x = cfg["explain"]
    torch.manual_seed(seed)
    subs, tg = _batched_subgraphs(g, edge_ids, x["num_hops"], x["max_in"], seed)
    B, tgt = batch_graphs(subs, tg)
    B = B.to(device); tgt = tgt.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        logits, z = model(B, return_emb=True)
        orig = logits[tgt].argmax(1)
    exp = EPGExplainer(z.shape[1], B.num_feats, x["pg_coff_size"], x["pg_coff_ent"]).to(device)
    opt = torch.optim.Adam(exp.parameters(), lr=x["pg_lr"])
    E = x["pg_epochs"]; t0, t1 = 5.0, 1.0
    for ep in range(E):
        temp = t0 * (t1 / t0) ** (ep / max(1, E))
        w = exp.feature_logits(z)
        mask = exp.concrete(w, temp, True)
        out = model(B, B.efeat * mask)[tgt]
        pred_loss = F.cross_entropy(out, orig)
        size_loss = exp.coff_size * mask.sum()
        m = mask * 0.98 + 0.01
        ent = -(m * torch.log(m) + (1 - m) * torch.log(1 - m)).mean()
        loss = pred_loss + size_loss + exp.coff_ent * ent
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        imp = torch.sigmoid(exp.feature_logits(z)).cpu().numpy()
    for p in model.parameters():
        p.requires_grad_(True)
    return imp


# --------------------------------------------------------------------------
def egnn_importance(model, g, edge_ids, cfg, device, seed=0):
    """Per-edge feature masks (GNNExplainer objective), averaged over edges."""
    x = cfg["explain"]
    torch.manual_seed(seed)
    subs, tg = _batched_subgraphs(g, edge_ids, x["num_hops"], x["max_in"], seed)
    B, tgt = batch_graphs(subs, tg)
    B = B.to(device); tgt = tgt.to(device)
    # which explained instance each edge belongs to
    owner = torch.cat([torch.full((s.num_edges,), i, dtype=torch.long) for i, s in enumerate(subs)]).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        orig = model(B)[tgt].argmax(1)
    n = len(subs)
    W = nn.Parameter(torch.randn(n, B.num_feats, device=device) * 0.1)
    opt = torch.optim.Adam([W], lr=x["gnn_lr"])
    for _ in range(x["gnn_epochs"]):
        mask = torch.sigmoid(W)
        out = model(B, B.efeat * mask[owner])[tgt]
        loss = F.cross_entropy(out, orig, reduction="sum")
        ent = -(mask * torch.log(mask + 1e-15) + (1 - mask) * torch.log(1 - mask + 1e-15))
        loss = loss + n * (x["gnn_coff_size"] * mask.mean() + x["gnn_coff_ent"] * ent.mean())
        opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return torch.sigmoid(W).detach().mean(0).cpu().numpy()


EXPLAINERS = {"pg": epg_importance, "gnn": egnn_importance}
