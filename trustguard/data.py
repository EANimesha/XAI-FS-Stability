"""Load a NetFlow-style CSV, preprocess it, and build train/test edge graphs.

Mirrors the original TrustGuard notebook pipeline:
  * node id = "IP:port"; optional source-IP randomisation (as in E-GraphSAGE)
  * label = attack category (optionally grouped), label-encoded
  * stratified train/test split
  * target encoding of categorical columns, standard scaling of the rest
  (all statistics fitted on the training split only)
"""
from __future__ import annotations

import json
import logging
import os
import socket
import struct
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .graph import EdgeGraph

log = logging.getLogger(__name__)



@dataclass
class Dataset:
    train: EdgeGraph
    test: EdgeGraph
    feature_names: list
    classes: list            # class names, index = label id

    @property
    def num_classes(self):
        return len(self.classes)


def _target_encode(train_col, y, test_col, smoothing=10.0, min_leaf=20):
    """Smoothed mean target encoding (fit on train only)."""
    prior = float(np.mean(y))
    df = pd.DataFrame({"c": train_col.astype(str).values, "y": y})
    stats = df.groupby("c")["y"].agg(["mean", "count"])
    w = 1.0 / (1.0 + np.exp(-(stats["count"] - min_leaf) / smoothing))
    enc = (prior * (1 - w) + stats["mean"] * w).to_dict()
    f = lambda col: col.astype(str).map(enc).fillna(prior).astype(np.float32).values
    return f(train_col), f(test_col)


def load_dataset(cfg: dict, seed: int = 0, cache_dir: str | None = None) -> Dataset:
    d = cfg["data"]
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"graphs_seed{seed}.pt")
        if os.path.exists(cache):
            log.info("Loading cached graphs from %s", cache)
            obj = torch.load(cache, weights_only=False)
            return Dataset(**obj)

    log.info("Reading %s", d["path"])
    df = pd.read_csv(d["path"], low_memory=False)
    if d.get("sample_frac"):
        df = df.groupby(d["attack_col"], group_keys=False).sample(frac=d["sample_frac"], random_state=seed)
    df.columns = [c.strip() for c in df.columns]

    rng = np.random.default_rng(seed)
    cs, cd, ps, pd_ = d.get("src_ip_col", "IPV4_SRC_ADDR"), d.get("dst_ip_col", "IPV4_DST_ADDR"), \
        d.get("src_port_col", "L4_SRC_PORT"), d.get("dst_port_col", "L4_DST_PORT")
    id_cols = [cs, cd, ps, pd_]
    src_ip = df[cs].astype(str)
    if d.get("randomize_src_ip", True):
        src_ip = pd.Series([socket.inet_ntoa(struct.pack(">I", int(x)))
                            for x in rng.integers(0xAC100001, 0xAC1F0001, len(df))], index=df.index)

    # ---- row cleaning -------------------------------------------------
    n0 = len(df)
    df = df.dropna(subset=[c for c in id_cols + [d["attack_col"]] if c in df.columns])
    if d.get("drop_na_rows", False):              # drop rows with ANY missing/inf feature
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if d.get("drop_duplicates", False):
        df = df.drop_duplicates()
    if len(df) < n0:
        log.info("Removed %d rows with missing endpoint/label%s (%d -> %d)", n0 - len(df),
                 " or features" if d.get("drop_na_rows") else "", n0, len(df))
    src_ip = src_ip.loc[df.index]
    df = df.reset_index(drop=True); src_ip = src_ip.reset_index(drop=True)

    src = (src_ip + ":" + df[ps].astype(str)).values
    dst = (df[cd].astype(str) + ":" + df[pd_].astype(str)).values
    labels = df[d["attack_col"]].astype(str).str.strip()
    groups = d.get("class_groups") or {}
    labels = labels.map(lambda s: groups.get(s, s))
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)

    drop = set(id_cols) | set(d.get("drop_cols", [])) | {d["attack_col"], d.get("binary_label_col", "Label")}
    feats = [c for c in df.columns if c not in drop]
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    cat_cols = [c for c in d.get("categorical_cols", []) if c in feats]

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=d.get("test_size", 0.3), random_state=seed, stratify=y)
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    for c in cat_cols:
        Xtr[c], Xte[c] = _target_encode(Xtr[c], y[tr], Xte[c])
    num_cols = [c for c in feats if c not in cat_cols]
    if num_cols:
        sc = StandardScaler().fit(Xtr[num_cols])
        Xtr[num_cols] = sc.transform(Xtr[num_cols]); Xte[num_cols] = sc.transform(Xte[num_cols])
    # drop constant columns (useless and break correlation analysis)
    keep = [c for c in feats if Xtr[c].std() > 0]
    if len(keep) < len(feats):
        log.info("Dropping %d constant features: %s", len(feats) - len(keep), sorted(set(feats) - set(keep)))
    Xtr, Xte = Xtr[keep].values.astype(np.float32), Xte[keep].values.astype(np.float32)

    train = EdgeGraph.from_flows(src[tr], dst[tr], Xtr, y[tr])
    test = EdgeGraph.from_flows(src[te], dst[te], Xte, y[te])
    ds = Dataset(train, test, keep, list(le.classes_))
    log.info("Train: %d flows, %d nodes | Test: %d flows | %d features | classes %s",
             len(tr), train.num_nodes, len(te), len(keep), ds.classes)
    log.info("Class counts (train): %s", dict(zip(ds.classes, np.bincount(y[tr]).tolist())))
    if cache:
        torch.save(ds.__dict__, cache)
        with open(os.path.join(cache_dir, "features.json"), "w") as f:
            json.dump({"features": keep, "classes": ds.classes}, f, indent=1)
    return ds
