"""Lightweight edge-attributed directed graph in pure PyTorch (no DGL).

A flow dataset becomes a graph whose nodes are endpoints ("IP:port") and whose
edges are flows. Following E-GraphSAGE, each flow is inserted in both
directions; `flow_id` records which original row an edge came from and
`is_forward` marks the original direction (used for evaluation, so every flow
is scored exactly once).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class EdgeGraph:
    src: torch.Tensor          # (E,) long
    dst: torch.Tensor          # (E,) long
    efeat: torch.Tensor        # (E, F) float
    label: torch.Tensor        # (E,) long
    flow_id: torch.Tensor      # (E,) long, original row id
    is_forward: torch.Tensor   # (E,) bool
    num_nodes: int
    _in_csr: Optional[tuple] = field(default=None, repr=False)
    _dgl: Optional[object] = field(default=None, repr=False)

    def dgl(self):
        """Cached DGL view of this graph (same edge order, same device)."""
        if self._dgl is None or self._dgl.device != self.src.device:
            import dgl
            self._dgl = dgl.graph((self.src, self.dst), num_nodes=self.num_nodes)
        return self._dgl

    # ------------------------------------------------------------------
    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    @property
    def num_feats(self) -> int:
        return int(self.efeat.shape[1])

    @property
    def device(self):
        return self.src.device

    def to(self, device) -> "EdgeGraph":
        return EdgeGraph(self.src.to(device), self.dst.to(device), self.efeat.to(device),
                         self.label.to(device), self.flow_id.to(device),
                         self.is_forward.to(device), self.num_nodes)

    # ------------------------------------------------------------------
    @staticmethod
    def from_flows(src_names, dst_names, X: np.ndarray, y: np.ndarray, bidirectional=True) -> "EdgeGraph":
        names = np.concatenate([np.asarray(src_names), np.asarray(dst_names)])
        _, inv = np.unique(names, return_inverse=True)
        n = len(src_names)
        s, d = inv[:n], inv[n:]
        num_nodes = int(inv.max()) + 1 if len(inv) else 0
        fid = np.arange(n)
        if bidirectional:
            src = np.concatenate([s, d]); dst = np.concatenate([d, s])
            X2 = np.concatenate([X, X]); y2 = np.concatenate([y, y])
            fid2 = np.concatenate([fid, fid])
            fwd = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
        else:
            src, dst, X2, y2, fid2, fwd = s, d, X, y, fid, np.ones(n, bool)
        return EdgeGraph(torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long),
                         torch.as_tensor(X2, dtype=torch.float32), torch.as_tensor(y2, dtype=torch.long),
                         torch.as_tensor(fid2, dtype=torch.long), torch.as_tensor(fwd), num_nodes)

    # ------------------------------------------------------------------
    def _csr_in(self):
        """Incoming-edge CSR on CPU: for node v, edges order[ptr[v]:ptr[v+1]]."""
        if self._in_csr is None:
            dst = self.dst.cpu().numpy()
            order = np.argsort(dst, kind="stable")
            counts = np.bincount(dst, minlength=self.num_nodes)
            ptr = np.concatenate([[0], np.cumsum(counts)])
            self._in_csr = (order, ptr, self.src.cpu().numpy())
        return self._in_csr

    def computation_subgraph(self, edge_id: int, num_hops: int = 2, max_in: int = 50,
                             rng: Optional[np.random.Generator] = None):
        """Edges needed to compute a `num_hops`-layer embedding of both endpoints
        of `edge_id`. Incoming edges per node are sampled to at most `max_in`
        (GraphSAGE-style) so hub nodes (e.g. scanners) stay tractable.

        Returns (subgraph, target_local_edge_index).
        """
        rng = rng or np.random.default_rng(0)
        order, ptr, src_np = self._csr_in()
        u, v = int(self.src[edge_id]), int(self.dst[edge_id])
        frontier = {u, v}
        nodes = set(frontier)
        edges = {int(edge_id)}
        for _ in range(num_hops):
            nxt = set()
            for x in frontier:
                inc = order[ptr[x]:ptr[x + 1]]
                if len(inc) > max_in:
                    inc = rng.choice(inc, max_in, replace=False)
                edges.update(int(e) for e in inc)
                nxt.update(int(s) for s in src_np[inc])
            frontier = nxt - nodes
            nodes |= nxt
        eids = torch.as_tensor(sorted(edges), dtype=torch.long)
        sub = self.edge_subgraph(eids)
        target = int((eids == edge_id).nonzero()[0])
        return sub, target

    def edge_subgraph(self, eids: torch.Tensor) -> "EdgeGraph":
        eids = eids.to(self.src.device)
        s, d = self.src[eids], self.dst[eids]
        nodes, inv = torch.unique(torch.cat([s, d]), return_inverse=True)
        m = eids.numel()
        return EdgeGraph(inv[:m], inv[m:], self.efeat[eids], self.label[eids],
                         self.flow_id[eids], self.is_forward[eids], int(nodes.numel()))


def batch_graphs(graphs, targets):
    """Disjoint union of subgraphs; returns (graph, global target indices)."""
    src, dst, ef, lab, fid, fwd, tgt = [], [], [], [], [], [], []
    n_off, e_off = 0, 0
    for g, t in zip(graphs, targets):
        src.append(g.src + n_off); dst.append(g.dst + n_off)
        ef.append(g.efeat); lab.append(g.label); fid.append(g.flow_id); fwd.append(g.is_forward)
        tgt.append(t + e_off)
        n_off += g.num_nodes; e_off += g.num_edges
    G = EdgeGraph(torch.cat(src), torch.cat(dst), torch.cat(ef), torch.cat(lab),
                  torch.cat(fid), torch.cat(fwd), n_off)
    return G, torch.as_tensor(tgt, dtype=torch.long, device=G.device)
