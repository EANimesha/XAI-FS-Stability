"""Top-k selection-frequency heatmap: for each class x feature, the share of explainer runs
(all seeds x sample sizes) in which the feature is in the top-k. Values near 0 or 1 = stable,
values in between = seed-dependent. One panel per explainer.
usage: python topk_freq.py runs/NF-BoT-IoT-v2 figs/topk_freq_v2.pdf [frac=0.25]"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

run, out = sys.argv[1], sys.argv[2]; frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
imp = json.load(open(os.path.join(run, "importance.json")))
meta = json.load(open(os.path.join(run, "features.json")))
feats, classes = meta["features"], meta["classes"]
F = len(feats); k = max(3, int(round(F * frac)))
methods = [m for m in ("pg", "gnn") if m in imp]
mats = []
for m in methods:
    M = np.zeros((len(classes), F))
    for ci, c in enumerate(classes):
        runs = [v for byn in imp[m].get(c, {}).values() for v in byn.values()]
        for v in runs:
            M[ci, np.argsort(-np.asarray(v))[:k]] += 1
        M[ci] /= max(1, len(runs))
    mats.append(M)
order = np.argsort(-np.mean(mats, axis=(0, 1)))       # most-selected features first
w = min(7.0, 1.2 + 0.17 * F)
fig, axes = plt.subplots(len(methods), 1, figsize=(w, 0.35 + 0.95 * len(methods) * len(classes) / 5 + 0.9), sharex=True)
axes = np.atleast_1d(axes)
for ax, m, M in zip(axes, methods, mats):
    im = ax.imshow(M[:, order], cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=6)
    ax.set_ylabel({"pg": "E-PG", "gnn": "E-GNN"}[m], fontsize=7)
axes[-1].set_xticks(range(F)); axes[-1].set_xticklabels([feats[i] for i in order], rotation=90, fontsize=5)
cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01); cb.ax.tick_params(labelsize=6)
cb.set_label(f"share of runs in top-{k}", fontsize=6)
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
