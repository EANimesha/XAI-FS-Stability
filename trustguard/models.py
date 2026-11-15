"""E-GraphSAGE (Lo et al., 2022) and the TrustGuard One-vs-All variant, pure PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import EdgeGraph

# ---------------------------------------------------------------- backend
# "dgl"  : message passing with DGL update_all (original TrustGuard library)
# "torch": pure PyTorch index_add_ (identical maths, no DGL needed)
_BACKEND = "torch"


def set_backend(name: str) -> str:
    """Select the message-passing backend; falls back to torch if DGL is missing."""
    global _BACKEND
    if name == "dgl":
        try:
            import dgl  # noqa: F401
        except Exception as e:  # pragma: no cover
            import logging
            logging.getLogger(__name__).warning("DGL not importable (%s); using torch backend", e)
            name = "torch"
    _BACKEND = name
    return name


def get_backend() -> str:
    return _BACKEND


class SAGELayer(nn.Module):
    """h_v' = ReLU(W_apply [h_v || mean_{(u->v)} W_msg [h_u || e_uv]])"""

    def __init__(self, ndim_in, edim, ndim_out):
        super().__init__()
        self.W_msg = nn.Linear(ndim_in + edim, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)

    def forward(self, g: EdgeGraph, h, efeat):
        m = self.W_msg(torch.cat([h[g.src], efeat], 1))
        if _BACKEND == "dgl":
            import dgl.function as fn
            dg = g.dgl()
            with dg.local_scope():
                dg.edata["m"] = m
                dg.update_all(fn.copy_e("m", "m"), fn.mean("m", "h_neigh"))
                neigh = dg.ndata["h_neigh"]
        else:
            agg = torch.zeros(g.num_nodes, m.shape[1], device=m.device, dtype=m.dtype)
            agg.index_add_(0, g.dst, m)
            deg = torch.bincount(g.dst, minlength=g.num_nodes).clamp(min=1).unsqueeze(1).to(m.dtype)
            neigh = agg / deg
        return F.relu(self.W_apply(torch.cat([h, neigh], 1)))


class Encoder(nn.Module):
    """Two SAGE layers -> edge embedding z_uv = [h_u || h_v]."""

    def __init__(self, edim, hidden=128, dropout=0.2):
        super().__init__()
        self.edim = edim
        self.l1 = SAGELayer(edim, edim, hidden)
        self.l2 = SAGELayer(hidden, edim, hidden)
        self.drop = nn.Dropout(dropout)
        self.out_dim = 2 * hidden

    def node_emb(self, g, efeat):
        h = torch.ones(g.num_nodes, self.edim, device=efeat.device)   # node features = 1
        h = self.l1(g, h, efeat)
        h = self.l2(g, self.drop(h), efeat)
        return h

    def forward(self, g, efeat=None):
        efeat = g.efeat if efeat is None else efeat
        h = self.node_emb(g, efeat)
        return torch.cat([h[g.src], h[g.dst]], 1)


class EGraphSAGE(nn.Module):
    """Baseline multi-class edge classifier (softmax head)."""

    def __init__(self, edim, num_classes, hidden=128, dropout=0.2):
        super().__init__()
        self.enc = Encoder(edim, hidden, dropout)
        self.head = nn.Linear(self.enc.out_dim, num_classes)

    def forward(self, g, efeat=None, return_emb=False):
        z = self.enc(g, efeat)
        out = self.head(z)
        return (out, z) if return_emb else out


class OvAEGraphSAGE(nn.Module):
    """TrustGuard Stage 2: one binary classifier per class, each fed its own
    feature subset `feat_idx[c]` (all subsets have the same size k).

    shared_encoder=True reproduces the original TrustGuard code (one encoder,
    re-used for every class subset). False gives each class its own encoder.
    """

    def __init__(self, feat_idx: dict, num_classes, hidden=128, dropout=0.2, shared_encoder=True):
        super().__init__()
        self.num_classes = num_classes
        self.feat_idx = {int(c): list(map(int, v)) for c, v in feat_idx.items()}
        k = len(self.feat_idx[0])
        assert all(len(v) == k for v in self.feat_idx.values()), "all classes need k features"
        self.shared = shared_encoder
        n_enc = 1 if shared_encoder else num_classes
        self.encs = nn.ModuleList([Encoder(k, hidden, dropout) for _ in range(n_enc)])
        self.heads = nn.ModuleList([nn.Linear(2 * hidden, 1) for _ in range(num_classes)])

    def logit(self, g, c):
        enc = self.encs[0 if self.shared else c]
        idx = torch.as_tensor(self.feat_idx[c], device=g.efeat.device)
        return self.heads[c](enc(g, g.efeat[:, idx])).squeeze(1)

    def forward(self, g):
        return torch.stack([self.logit(g, c) for c in range(self.num_classes)], 1)
