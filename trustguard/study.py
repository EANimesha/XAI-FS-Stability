"""Experiments for the stability / pruning study (and TrustGuard Stage 1 + 2)."""
from __future__ import annotations

import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .explain import EXPLAINERS, true_positive_edges
from .train import evaluate_ova, train_ova
from .utils import dump_json, load_json

log = logging.getLogger(__name__)


# ============================== Stage 1 ==================================
def run_explanations(model, ds, pred_test, cfg, device, out_dir):
    """importance[method][class][n][seed] = list(F). Checkpointed + resumable."""
    x = cfg["explain"]
    path = os.path.join(out_dir, "importance.json")
    imp = load_json(path, {})
    tpath = os.path.join(out_dir, "xai_timing.csv")
    g = ds.test.to(device)
    for method in x["methods"]:
        fn = EXPLAINERS[method]
        sizes = x["sample_sizes"][method]
        for ci, cname in enumerate(ds.classes):
            tp = true_positive_edges(pred_test, g, ci)
            if len(tp) == 0:
                log.warning("No true positives for class %s, skipping", cname)
                continue
            for n in sizes:
                for s in x["seeds"]:
                    slot = imp.setdefault(method, {}).setdefault(cname, {}).setdefault(str(n), {})
                    if str(s) in slot:
                        continue
                    rng = np.random.default_rng(s)
                    edges = rng.choice(tp, min(n, len(tp)), replace=False)
                    t0 = time.time()
                    slot[str(s)] = fn(model, g, edges, cfg, device, seed=s).tolist()
                    sec = time.time() - t0
                    log.info("[%s] %-15s n=%-4d seed=%d  %.1fs", method, cname, n, s, sec)
                    pd.DataFrame([dict(method=method, cls=cname, n=n, seed=s, n_used=len(edges), sec=sec)]
                                 ).to_csv(tpath, mode="a", index=False, header=not os.path.exists(tpath))
                    dump_json(imp, path)
    return imp


def consensus(by_seed: dict) -> np.ndarray:
    """Rank-average over seeds; higher = more important."""
    R = [pd.Series(v).rank(ascending=False).values for v in by_seed.values()]
    return -np.mean(R, axis=0)


def _topk_jacc(a, b, k):
    A = set(np.argsort(-np.asarray(a))[:k]); B = set(np.argsort(-np.asarray(b))[:k])
    return len(A & B) / len(A | B)


def stability_report(imp, feature_names, out_dir, topk_frac=0.25, name=""):
    F = len(feature_names); k = max(3, int(round(F * topk_frac)))
    rows = []
    for m, byc in imp.items():
        for c, byn in byc.items():
            for n, bys in byn.items():
                vecs = list(bys.values())
                taus = [kendalltau(a, b)[0] for a, b in itertools.combinations(vecs, 2)]
                jacs = [_topk_jacc(a, b, k) for a, b in itertools.combinations(vecs, 2)]
                rows.append(dict(dataset=name, method=m, cls=c, n=int(n), seeds=len(vecs),
                                 kendall=np.nanmean(taus) if taus else np.nan,
                                 kendall_sd=np.nanstd(taus) if taus else np.nan,
                                 jaccard_topk=np.mean(jacs) if jacs else np.nan, k=k))
    df = pd.DataFrame(rows).sort_values(["method", "cls", "n"])
    df.to_csv(os.path.join(out_dir, "stability.csv"), index=False)

    cross = []
    if "pg" in imp and "gnn" in imp:
        for c in imp["pg"]:
            common = set(imp["pg"][c]) & set(imp["gnn"].get(c, {}))
            if common:
                n = max(common, key=int)
                a, b = consensus(imp["pg"][c][n]), consensus(imp["gnn"][c][n])
                cross.append(dict(dataset=name, cls=c, n=int(n), kendall=kendalltau(a, b)[0],
                                  jaccard_topk=_topk_jacc(a, b, k)))
    cross = pd.DataFrame(cross)
    cross.to_csv(os.path.join(out_dir, "pg_vs_gnn.csv"), index=False)

    # human-readable top features per class (consensus, PG)
    top = {}
    for m, byc in imp.items():
        for c, byn in byc.items():
            nmax = max(byn, key=int)
            order = np.argsort(-consensus(byn[nmax]))
            top.setdefault(m, {})[c] = [feature_names[i] for i in order[:10]]
    dump_json(top, os.path.join(out_dir, "top_features.json"))
    if not df.empty:
        log.info("\n%s", df.groupby(["method", "n"])[["kendall", "jaccard_topk"]].mean().round(3))
    return df, cross


# ============================== Stage 2 ==================================
def feature_corr(ds, max_rows=50000, seed=0):
    X = ds.train.efeat[ds.train.is_forward].numpy()
    if len(X) > max_rows:
        X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
    rho = np.abs(np.atleast_2d(spearmanr(X).correlation))
    rho = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def corr_filter(ranked, k, rho, thr):
    kept, skipped = [], []
    for f in ranked:
        (skipped if any(rho[f, g] >= thr for g in kept) else kept).append(int(f))
    out = kept[:k]
    return out + skipped[: k - len(out)]


def rankings(imp, classes, method, single_n=50):
    single, cons = {}, {}
    for ci, c in enumerate(classes):
        byn = imp[method][c]
        n1 = min(byn, key=lambda n: abs(int(n) - single_n))
        s0 = sorted(byn[n1], key=int)[0]
        single[ci] = list(np.argsort(-np.asarray(byn[n1][s0])))
        nmax = max(byn, key=int)
        cons[ci] = list(np.argsort(-consensus(byn[nmax])))
    return single, cons


def build_jobs(ds, imp, cfg):
    p = cfg["prune"]
    C, F = ds.num_classes, len(ds.feature_names)
    single, cons = rankings(imp, ds.classes, p["method"], p.get("single_n", 50))
    rho = feature_corr(ds)
    jobs = [(dict(strategy="all", frac=1.0, k=F, rseed=-1), {c: list(range(F)) for c in range(C)})]
    for frac in p["fracs"]:
        k = max(2, int(round(F * frac)))
        if k >= F:
            continue
        strat = p["strategies"]
        if "xai_single" in strat:
            jobs.append((dict(strategy="xai_single", frac=frac, k=k, rseed=-1), {c: single[c][:k] for c in range(C)}))
        if "xai_consensus" in strat:
            jobs.append((dict(strategy="xai_consensus", frac=frac, k=k, rseed=-1), {c: cons[c][:k] for c in range(C)}))
        if "xai_cons_corr" in strat:
            jobs.append((dict(strategy="xai_cons_corr", frac=frac, k=k, rseed=-1),
                         {c: corr_filter(cons[c], k, rho, p["corr_thr"]) for c in range(C)}))
        if "random" in strat:
            for s in p["random_seeds"]:
                r = np.random.default_rng(1000 + s)
                jobs.append((dict(strategy="random", frac=frac, k=k, rseed=s),
                             {c: sorted(r.choice(F, k, replace=False).tolist()) for c in range(C)}))
    return jobs


def run_pruning(ds, imp, cfg, device, out_dir, name=""):
    path = os.path.join(out_dir, "pruning.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    sel_path = os.path.join(out_dir, "selected_features.json")
    sel_log = load_json(sel_path, {})
    train_seeds = cfg["prune"].get("train_seeds", [cfg["train"]["seed"]])
    for (tag0, sel), ts in itertools.product(build_jobs(ds, imp, cfg), train_seeds):
        tag = dict(tag0, tseed=ts)
        key = f"{tag['strategy']}|{tag['k']}|{tag['rseed']}|{ts}"
        if not done.empty and key in set(done["key"]):
            continue
        sel_log[f"{tag['strategy']}|{tag['k']}|{tag['rseed']}"] = {ds.classes[c]: [ds.feature_names[i] for i in v] for c, v in sel.items()}
        dump_json(sel_log, sel_path)
        model, ttrain = train_ova(ds, sel, cfg, device, seed=ts)
        res = evaluate_ova(model, ds, device)
        row = dict(dataset=name, key=key, **tag, recall_w=res["recall_w"], precision_w=res["precision_w"],
                   f1_w=res["f1_w"], f1_macro=res["f1_macro"], recall_macro=res["recall_macro"],
                   train_s=ttrain, test_s=res["test_s"],
                   **{f"f1_{c}": v["f1-score"] for c, v in res["per_class"].items()})
        log.info("%s  F1w=%.4f  F1macro=%.4f  (%.0fs)", key, row["f1_w"], row["f1_macro"], ttrain)
        pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))
        del model
    res = pd.read_csv(path)
    summarise_pruning(res, out_dir)
    return res


def summarise_pruning(res, out_dir):
    cols = ["recall_w", "f1_w", "f1_macro", "train_s", "test_s"]
    g = res.groupby(["strategy", "frac", "k"])[cols].agg(["mean", "std"])
    g.to_csv(os.path.join(out_dir, "pruning_summary.csv"))
    pct = res.groupby(["strategy", "frac"])[["recall_w", "f1_w", "f1_macro"]].agg(["mean", "std"]).mul(100).round(2)
    pcls = res.groupby(["strategy", "frac"])[[c for c in res.columns if c.startswith("f1_") and c not in ("f1_w", "f1_macro")]].mean().mul(100).round(1)
    pcls.to_csv(os.path.join(out_dir, "pruning_per_class_f1.csv"))
    log.info("\nper-class F1:\n%s", pcls)
    with open(os.path.join(out_dir, "pruning_table.tex"), "w") as f:
        f.write(pct.to_latex())
    log.info("\n%s", pct)
    return g
