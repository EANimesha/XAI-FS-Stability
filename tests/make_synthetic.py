"""Generate a small NetFlow-v2-like CSV with learnable class structure."""
import numpy as np, pandas as pd, sys

def make(n=6000, seed=0, path="tests/synthetic_nf.csv"):
    r = np.random.default_rng(seed)
    classes = ["Benign", "DoS", "Reconnaissance", "Theft"]
    p = [0.3, 0.3, 0.35, 0.05]
    y = r.choice(len(classes), n, p=p)
    hosts = [f"192.168.1.{i}" for i in range(1, 40)]
    src = r.choice(hosts, n); dst = r.choice(hosts, n)
    scanner = "192.168.1.250"
    src = np.where(y == 2, scanner, src)                      # recon fan-out
    df = pd.DataFrame({
        "IPV4_SRC_ADDR": src, "L4_SRC_PORT": r.integers(1024, 65535, n),
        "IPV4_DST_ADDR": dst, "L4_DST_PORT": np.where(y == 2, r.integers(1, 1024, n), r.choice([80, 443, 53], n)),
        "PROTOCOL": np.where(y == 1, 17, 6), "L7_PROTO": r.choice([7.0, 91.0, 5.0], n),
        "IN_BYTES": r.lognormal(5 + 2 * (y == 1), 1), "OUT_BYTES": r.lognormal(5 - (y == 2), 1),
        "IN_PKTS": r.poisson(5 + 20 * (y == 1)), "OUT_PKTS": r.poisson(4, n),
        "TCP_FLAGS": np.where(y == 2, 2, r.choice([24, 27], n)),
        "FLOW_DURATION_MILLISECONDS": r.exponential(100 + 900 * (y == 3)),
        "NOISE_1": r.normal(size=n), "NOISE_2": r.normal(size=n),
    })
    df["IN_BYTES_COPY"] = df["IN_BYTES"] * 1.01 + r.normal(scale=0.1, size=n)   # redundant feature
    df["Label"] = (y > 0).astype(int); df["Attack"] = [classes[i] for i in y]
    df.to_csv(path, index=False); print("wrote", path, df.shape)

if __name__ == "__main__":
    make(path=sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic_nf.csv")
