"""pytest -q  : end-to-end run on a tiny synthetic dataset (CPU, ~15 s)."""
import os, shutil, subprocess, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_end_to_end():
    os.chdir(ROOT)
    shutil.rmtree("runs_smoke", ignore_errors=True)
    subprocess.check_call([sys.executable, "tests/make_synthetic.py"])
    subprocess.check_call([sys.executable, "-m", "trustguard", "all", "--config", "configs/smoke.yaml", "--device", "cpu"])
    out = "runs_smoke/smoke"
    for f in ["baseline_metrics.json", "importance.json", "stability.csv", "pruning.csv", "pruning_table.tex"]:
        assert os.path.exists(os.path.join(out, f)), f
    p = pd.read_csv(os.path.join(out, "pruning.csv"))
    assert set(p.strategy) == {"all", "xai_single", "xai_consensus", "xai_cons_corr", "random"}
