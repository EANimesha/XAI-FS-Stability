"""Paper figures from merged CSVs (run `python -m trustguard merge --runs runs/*` first).

    python -m trustguard.plots --stability merged_stability.csv --pruning merged_pruning.csv --out figs
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def stability_fig(df, out):
    """Kendall's tau vs n; colour = dataset, solid/circle = E-PG, dashed/square = E-GNN; legend above plot."""
    g = df.groupby(["run", "method", "n"])["kendall"].mean().reset_index()
    runs = sorted(g["run"].unique())
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    col = {r: pal[i % len(pal)] for i, r in enumerate(runs)}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    h, l = {}, {}
    for m in ("pg", "gnn"):
        for r in runs:
            d = g[(g.run == r) & (g.method == m)].sort_values("n")
            if d.empty: continue
            name = "E-PG" if m == "pg" else "E-GNN"
            line, = ax.plot(d["n"], d["kendall"], color=col[r], marker="o" if m == "pg" else "s", ms=3.5, lw=1.1,
                            ls="-" if m == "pg" else "--", label=f"{r} ({name})")
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["10", "25", "50", "100"]); ax.minorticks_off()
    ax.set_xlabel("explained edges per class ($n$)", fontsize=7); ax.set_ylabel("Kendall's $\\tau$ across seeds", fontsize=7)
    ax.tick_params(labelsize=7); ax.grid(alpha=.3); ax.set_ylim(0.4, 1.0)
    # E-PG entries fill the left column, E-GNN the right (ncol=2 fills column-wise)
    ax.legend(fontsize=5.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, columnspacing=1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "stability.pdf"), bbox_inches="tight"); plt.close(fig)


def pruning_fig(df, out, metric="f1_macro"):
    for run, d in df.groupby("run"):
        full = d[d.strategy == "all"][metric].mean()
        g = d[d.strategy != "all"].groupby(["strategy", "frac"])[metric].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for s, x in g.groupby("strategy"):
            ax.errorbar(x["frac"] * 100, x["mean"] * 100, yerr=x["std"].fillna(0) * 100, marker="o", capsize=2, label=s)
        ax.axhline(full * 100, color="k", ls=":", label="all features")
        ax.set_xlabel("features kept (%)"); ax.set_ylabel(metric.replace("_", "-") + " (%)")
        ax.set_title(run, fontsize=8); ax.grid(alpha=.3); ax.legend(fontsize=5)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"pruning_{run}.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability"); ap.add_argument("--pruning"); ap.add_argument("--out", default="figs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.stability: stability_fig(pd.read_csv(a.stability), a.out)
    if a.pruning: pruning_fig(pd.read_csv(a.pruning), a.out)
    print("figures in", a.out)
