# Can We Trust Explanation-Guided Feature Selection? Stability and Redundancy in GNN-based Intrusion Detection

Code accompanying the paper of the same title (under double-blind review).

The study asks whether explanation-guided feature selection, as used to refine GNN-based network intrusion detectors (TrustGuard), is reliable. It tests two assumptions:

* **A1, stability:** class-wise feature rankings from GNN explainers reflect the model, not the random seed or the few explained flows.
* **A2, informativeness:** features chosen by the explainer are better than an equal-size random subset, even when features are correlated.

It is organised around three research questions:

| RQ | Question | What the code does |
|---|---|---|
| RQ1 | Are class-wise rankings stable? | Runs E-PGExplainer and E-GNNExplainer for every class, sample size n ∈ {10, 25, 50, 100} and seed ∈ {0, 1, 2}; reports Kendall's τ and top-k Jaccard |
| RQ2 | Does explanation-guided pruning beat random pruning? | Trains one-vs-all E-GraphSAGE detectors on per-class subsets (single run, seed consensus, consensus + correlation filter, random) at 75/50/25% of features, three training seeds each |
| RQ3 | What does it cost? | Logs explanation, retraining and inference time |

## Pipeline

```
NetFlow CSV -> flow graph (nodes = IP:port, edges = flows) -> E-GraphSAGE baseline
   -> Stage 1: explain correct test edges per class (E-PG, E-GNN; seeds x n)   -> RQ1 stability
   -> feature selection: single | consensus | consensus + corr. filter | random
   -> Stage 2: OvA E-GraphSAGE (shared encoder, one head per class) x 3 seeds   -> RQ2 detection, RQ3 cost
```

## Datasets

The NetFlow benchmarks (NF-BoT-IoT, NF-BoT-IoT-v2, NF-ToN-IoT) are publicly available from the University of Queensland NIDS dataset collection. Download the CSVs into `data/` or edit `data.path` in each config.

| Config | Dataset | Features | Classes |
|---|---|---|---|
| `configs/nf_bot_iot.yaml` | NF-BoT-IoT | 8 | 4 (DDoS merged into DoS) |
| `configs/nf_bot_iot_v2.yaml` | NF-BoT-IoT-v2, class-balanced subsample (≤10k flows per class) | 39 | 5 |
| `configs/nf_ton_iot.yaml` | NF-ToN-IoT | 8 | 7 (ddos→dos, ransomware→backdoor, mitm→scanning) |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
# optional: DGL matching your torch/CUDA version (otherwise a pure-PyTorch backend is used)
pytest -q                                                             # smoke test on synthetic data (~15 s, CPU)
```

DGL is the default backend. A pure-PyTorch backend computes the same model and is used automatically if DGL is missing (`--backend dgl|torch` to force one; `tests/test_dgl_equivalence.py` checks they agree).

## Reproduce the paper

```bash
# one dataset, all stages (baseline -> explain -> stability -> prune)
python -m trustguard all --config configs/nf_bot_iot.yaml --device cuda:0

# or step by step
python -m trustguard baseline --config configs/nf_bot_iot.yaml
python -m trustguard explain  --config configs/nf_bot_iot.yaml    # RQ1 explanations
python -m trustguard prune    --config configs/nf_bot_iot.yaml    # RQ2 / RQ3

# all three datasets in parallel (one per GPU)
./run_all.sh

# merge results and draw figures
python -m trustguard merge --runs runs/NF-BoT-IoT runs/NF-BoT-IoT-v2 runs/NF-ToN-IoT
python -m trustguard.plots --stability merged_stability.csv --pruning merged_pruning.csv --out figs
python scripts/pruning_row.py merged_pruning.csv figs/pruning_row.pdf
python scripts/topk_freq.py runs/NF-BoT-IoT-v2 figs/topk_freq_v2.pdf
```

Every stage is checkpointed, so an interrupted run resumes where it stopped. For a quick trial, set `data.sample_frac: 0.1` in a config and use a new `name:`.

All hyperparameters are in `configs/base.yaml` (E-GraphSAGE: 2 layers, hidden 128; baseline 3000 epochs; OvA 300 epochs; E-PG 30 epochs; E-GNN 200 epochs; correlation threshold 0.9).

## Outputs (`runs/<dataset>/`)

| File | Content |
|---|---|
| `baseline.pt`, `baseline_metrics.json` | baseline detector and its metrics |
| `importance.json` | class-wise importance for every explainer, n and seed |
| `stability.csv`, `pg_vs_gnn.csv`, `top_features.json` | RQ1 |
| `pruning.csv`, `pruning_summary.csv`, `pruning_per_class_f1.csv`, `pruning_table.tex`, `selected_features.json` | RQ2 |
| `xai_timing.csv` | RQ3 explanation cost (training/inference time is in `pruning.csv`) |
| `log.txt`, `config_used.json` | provenance |

## Repository layout

```
trustguard/   graph.py (flow graphs)  data.py (preprocessing)  models.py (E-GraphSAGE, OvA)
              explain.py (E-PG, E-GNN)  study.py (stability, selection strategies, pruning)
              train.py  cli.py  plots.py  utils.py
configs/      one YAML per dataset + shared base.yaml
scripts/      paper figures (pruning row, top-k selection frequency)
tests/        smoke test and backend-equivalence test
```

## Implementation notes

* Each flow is inserted in both directions (as in E-GraphSAGE) but scored once at test time.
* Explanation subgraphs are 2-hop and sample at most 50 incoming edges per node.
* E-GNNExplainer optimises all per-edge masks jointly on a disjoint batch (equivalent to separate runs, much faster).
* Constant features are dropped; categorical fields are target-encoded and the rest standardised using training statistics only.

## License

Released for research use. A permanent repository and citation will be added after review.
