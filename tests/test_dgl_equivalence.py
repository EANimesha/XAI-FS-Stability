"""Checks that the DGL and pure-torch backends give identical outputs.
Skipped automatically when DGL is not installed."""
import pytest, torch
dgl = pytest.importorskip("dgl")
from trustguard.graph import EdgeGraph
from trustguard import models
import numpy as np

def test_backends_match():
    r = np.random.default_rng(0)
    n = 300
    g = EdgeGraph.from_flows(r.integers(0, 40, n).astype(str), r.integers(0, 40, n).astype(str),
                             r.normal(size=(n, 6)).astype("float32"), r.integers(0, 3, n))
    torch.manual_seed(0)
    m = models.EGraphSAGE(6, 3).eval()
    models.set_backend("torch"); a = m(g)
    models.set_backend("dgl");   b = m(g)
    assert torch.allclose(a, b, atol=1e-5)
