#!/usr/bin/env bash
# Run each dataset on its own GPU in parallel. Edit the list to suit.
set -e
CONFIGS=(nf_bot_iot nf_ton_iot nf_bot_iot_v2 aci_iot_23)
for i in "${!CONFIGS[@]}"; do
  c=${CONFIGS[$i]}
  CUDA_VISIBLE_DEVICES=$i nohup python -m trustguard all --config configs/$c.yaml > log_$c.txt 2>&1 &
  echo "started $c on GPU $i (log_$c.txt)"
done
wait
python -m trustguard merge --runs runs/*
python -m trustguard.plots --stability merged_stability.csv --pruning merged_pruning.csv --out figs
