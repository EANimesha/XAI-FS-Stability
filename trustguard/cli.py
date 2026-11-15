"""Command line entry point.

    python -m trustguard <command> --config configs/nf_bot_iot_v2.yaml [--device cuda:0]

commands
    baseline   train + evaluate the E-GraphSAGE baseline (saved, reused later)
    explain    Stage 1: E-PG / E-GNN class-wise importance for all classes, sizes, seeds
    stability  stability tables from importance.json
    prune      Stage 2: OvA models under each feature-selection strategy
    all        everything above in order (each step resumes if already done)
    merge      combine result CSVs of several datasets: --runs runs/*  (no --config)
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

import pandas as pd
import torch

from .data import load_dataset
from .models import EGraphSAGE, set_backend
from .study import run_explanations, run_pruning, stability_report
from .train import evaluate_baseline, train_baseline
from .utils import dump_json, get_device, load_config, load_json, set_seed, setup_logging

log = logging.getLogger("trustguard")


def _out(cfg):
    return os.path.join(cfg.get("output_dir", "runs"), cfg["name"])


def _baseline(cfg, ds, device, out, force=False):
    path = os.path.join(out, "baseline.pt")
    t = cfg["train"]
    if os.path.exists(path) and not force:
        model = EGraphSAGE(ds.train.num_feats, ds.num_classes, t["hidden"], t["dropout"]).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        log.info("Loaded baseline from %s", path)
    else:
        model, sec = train_baseline(ds, cfg, device, seed=t["seed"])
        torch.save(model.state_dict(), path)
        res, _ = evaluate_baseline(model, ds, device)
        res["train_s"] = sec; res["epochs"] = t["baseline_epochs"]
        dump_json(res, os.path.join(out, "baseline_metrics.json"))
        log.info("Baseline: recall_w=%.4f f1_w=%.4f f1_macro=%.4f", res["recall_w"], res["f1_w"], res["f1_macro"])
        for c, v in res["per_class"].items():
            log.info("  %-15s P=%.3f R=%.3f F1=%.3f n=%d", c, v["precision"], v["recall"], v["f1-score"], v["support"])
        log.info("  confusion (rows=true, cols=pred, order %s):\n%s", ds.classes,
                 "\n".join("    " + " ".join(f"{x:>8d}" for x in row) for row in res["confusion"]))
    res, pred = evaluate_baseline(model, ds, device)
    return model, pred


_SIG_KEYS = {"data": None, "train": ["seed", "hidden", "dropout", "lr", "baseline_epochs", "class_weight"]}


def _sig(cfg):
    t = cfg.get("train", {})
    return {"data": cfg.get("data"), "train": {k: t.get(k) for k in _SIG_KEYS["train"]}}


def _check_stale(cfg, out):
    """Refuse to reuse saved graphs/baseline/explanations made with different data or baseline settings
    (e.g. a quick sample_frac trial) -- they would silently be loaded and mixed with the new run."""
    old = load_json(os.path.join(out, "config_used.json"))
    if old and _sig(old) != _sig(cfg):
        raise SystemExit(
            f"\n{out} holds results from a run with different data/baseline settings "
            f"(e.g. a sample_frac trial).\nDelete that folder (rm -rf {out}) or change `name:` in the config, then rerun.\n"
            f"old: {_sig(old)}\nnew: {_sig(cfg)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trustguard")
    ap.add_argument("command", choices=["baseline", "explain", "stability", "prune", "all", "merge"])
    ap.add_argument("--config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", choices=["dgl", "torch"], help="override train.backend")
    ap.add_argument("--force", action="store_true", help="retrain baseline even if saved")
    ap.add_argument("--runs", nargs="*", help="run dirs for merge")
    a = ap.parse_args(argv)

    if a.command == "merge":
        setup_logging()
        dirs = [d for p in (a.runs or ["runs/*"]) for d in glob.glob(p)]
        for fname in ["stability.csv", "pg_vs_gnn.csv", "pruning.csv", "xai_timing.csv"]:
            frames = []
            for d in dirs:
                f = os.path.join(d, fname)
                if os.path.exists(f):
                    df = pd.read_csv(f); df["run"] = os.path.basename(d.rstrip("/")); frames.append(df)
            if frames:
                pd.concat(frames).to_csv(f"merged_{fname}", index=False)
                log.info("wrote merged_%s", fname)
        return

    cfg = load_config(a.config)
    out = _out(cfg)
    setup_logging(out)
    device = get_device(a.device)
    backend = set_backend(a.backend or cfg["train"].get("backend", "dgl"))
    log.info("Run %s on %s (backend: %s)", cfg["name"], device, backend)
    set_seed(cfg["train"]["seed"])
    _check_stale(cfg, out)
    ds = load_dataset(cfg, seed=cfg["train"]["seed"], cache_dir=out)
    dump_json(cfg, os.path.join(out, "config_used.json"))

    model, pred = _baseline(cfg, ds, device, out, force=a.force)
    if a.command == "baseline":
        return
    if a.command in ("explain", "all"):
        imp = run_explanations(model, ds, pred, cfg, device, out)
    else:
        imp = load_json(os.path.join(out, "importance.json"), {})
    if a.command in ("explain", "stability", "all"):
        stability_report(imp, ds.feature_names, out, name=cfg["name"])
    if a.command in ("prune", "all"):
        if not imp:
            raise SystemExit("importance.json missing: run `explain` first")
        run_pruning(ds, imp, cfg, device, out, name=cfg["name"])


if __name__ == "__main__":
    main()
