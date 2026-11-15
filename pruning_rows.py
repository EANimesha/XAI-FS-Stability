"""One-row pruning figure for the paper: macro-F1 vs features kept, one panel per dataset.
usage: python pruning_row.py merged_pruning.csv figs/pruning_row.pdf"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else "pruning_row.pdf"
order = [r for r in ["NF-BoT-IoT", "NF-BoT-IoT-v2", "NF-ToN-IoT"] if r in set(df.run)]
names = {"xai_single": "XAI-single", "xai_consensus": "XAI-consensus", "xai_cons_corr": "XAI-cons+corr", "random": "Random"}
style = {"xai_single": ("#1f77b4", "o"), "xai_consensus": ("#ff7f0e", "s"), "xai_cons_corr": ("#2ca02c", "^"), "random": ("#7f7f7f", "x")}
fig, axes = plt.subplots(1, len(order), figsize=(7.0, 2.1))
for ax, run in zip(axes, order):
    d = df[df.run == run]
    full = d[d.strategy == "all"]["f1_macro"].mean() * 100
    g = d[d.strategy != "all"].groupby(["strategy", "frac"])["f1_macro"].agg(["mean", "std"]).reset_index()
    for s in names:
        x = g[g.strategy == s]
        if x.empty: continue
        c, m = style[s]
        ax.errorbar(x["frac"] * 100, x["mean"] * 100, yerr=x["std"].fillna(0) * 100, color=c, marker=m,
                    ms=3.5, lw=1, capsize=2, label=names[s])
    ax.axhline(full, color="k", ls=":", lw=1, label="All features")
    ax.set_title(run, fontsize=8); ax.set_xticks([25, 50, 75]); ax.tick_params(labelsize=7)
    ax.set_xlabel("features kept (%)", fontsize=7); ax.grid(alpha=.3)
axes[0].set_ylabel("macro-F1 (%)", fontsize=7)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, bbox_inches="tight")
print("wrote", out)
