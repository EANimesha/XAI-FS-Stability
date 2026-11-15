"""Training / evaluation for the baseline and OvA models (full-batch, as in the original code)."""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from .models import EGraphSAGE, OvAEGraphSAGE
from .utils import set_seed

log = logging.getLogger(__name__)


def _class_weights(y, C, mode="balanced"):
    """balanced = n/(C*count) (original TrustGuard); sqrt = its square root (softer); none = all 1."""
    counts = torch.bincount(y, minlength=C).float().clamp(min=1)
    w = y.numel() / (C * counts)
    if mode == "sqrt":
        w = w.sqrt()
    elif mode == "none":
        w = torch.ones_like(w)
    return w


def metrics(y_true, y_pred, classes):
    r = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes,
                              output_dict=True, zero_division=0)
    out = {
        "accuracy": r["accuracy"],
        "recall_w": r["weighted avg"]["recall"], "precision_w": r["weighted avg"]["precision"],
        "f1_w": r["weighted avg"]["f1-score"], "f1_macro": r["macro avg"]["f1-score"],
        "recall_macro": r["macro avg"]["recall"],
        "per_class": {c: {k: r[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes},
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
    }
    return out


def _eval_mask(g):
    return g.is_forward    # score every flow exactly once


# ------------------------------------------------------------------ baseline
def train_baseline(ds, cfg, device, seed=0):
    t = cfg["train"]
    set_seed(seed)
    g = ds.train.to(device)
    model = EGraphSAGE(g.num_feats, ds.num_classes, t["hidden"], t["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    crit = nn.CrossEntropyLoss(weight=_class_weights(g.label, ds.num_classes, t.get("class_weight", "balanced")).to(device))
    t0 = time.time()
    for ep in range(1, t["baseline_epochs"] + 1):
        model.train(); opt.zero_grad()
        loss = crit(model(g), g.label)
        loss.backward(); opt.step()
        if ep % t.get("log_every", 100) == 0:
            log.info("baseline epoch %d loss %.4f", ep, loss.item())
    return model, time.time() - t0


@torch.no_grad()
def predict_baseline(model, g):
    model.eval()
    t0 = time.time()
    pred = model(g).argmax(1)
    if g.device.type == "cuda":
        torch.cuda.synchronize()
    return pred, time.time() - t0


def evaluate_baseline(model, ds, device):
    g = ds.test.to(device)
    pred, sec = predict_baseline(model, g)
    m = _eval_mask(g)
    res = metrics(g.label[m].cpu().numpy(), pred[m].cpu().numpy(), ds.classes)
    res["test_s"] = sec
    res["test_ms_per_1000"] = 1000 * sec / max(1, int(m.sum())) * 1000
    return res, pred


# ------------------------------------------------------------------ OvA
def train_ova(ds, feat_idx, cfg, device, seed=0):
    t = cfg["train"]
    set_seed(seed)
    g = ds.train.to(device)
    C = ds.num_classes
    model = OvAEGraphSAGE(feat_idx, C, t["hidden"], t["dropout"], t.get("ova_shared_encoder", True)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    pos_w = []
    for c in range(C):
        p = (g.label == c).sum().clamp(min=1).float()
        pos_w.append(((g.label.numel() - p) / p).to(device))
    t0 = time.time()
    for ep in range(1, t["ova_epochs"] + 1):
        model.train()
        losses = []
        for c in range(C):          # one optimiser step per binary classifier (original scheme)
            opt.zero_grad()
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_w[c])
            loss = crit(model.logit(g, c), (g.label == c).float())
            loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % t.get("log_every", 100) == 0:
            log.info("ova epoch %d mean BCE %.4f", ep, float(np.mean(losses)))
    return model, time.time() - t0


@torch.no_grad()
def evaluate_ova(model, ds, device):
    g = ds.test.to(device)
    model.eval()
    t0 = time.time()
    pred = model(g).argmax(1)
    if g.device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    m = _eval_mask(g)
    res = metrics(g.label[m].cpu().numpy(), pred[m].cpu().numpy(), ds.classes)
    res["test_s"] = sec
    res["test_ms_per_1000"] = 1000 * sec / max(1, int(m.sum())) * 1000
    return res
