"""Merge all closed-loop runs and compute point estimates with 95% bootstrap CIs.
Unit of resampling = one simulation run. Ratio metrics (FPR) are recomputed as
sum(flags)/sum(pairs) inside each resample. Paired Wilcoxon (matched attack, speed, seed)
with Holm correction across the baseline comparisons."""
import json, os, numpy as np
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # run from anywhere
from scipy.stats import wilcoxon

NAME = {"Plain CUSUM": "Flow-based CUSUM"}
MAIN = ["Static threshold", "Flow-based CUSUM", "FAPRP-style kNN", "CNN-BiLSTM-RF", "Ref. CNN-LSTM/GRU", "MERIT-Net"]
ABL = ["MERIT-Net", "MERIT no V-Net", "MERIT no fusion", "MERIT linear λ"]
SPEEDS = [0, 5, 10, 15, 20]
ATT = ["rreq", "rreq_stealth", "hello", "data", "syn"]
B = 10000
rng = np.random.default_rng(2026)


def load_all():
    files = [("../ns3sim/results/ns3_scaled_300runs.jsonl", None), ("../ns3sim/results/ns3_cusum_60runs.jsonl", "Flow-based CUSUM"),
             ("../ns3sim/results/ns3_10seed_main.jsonl", None), ("../ns3sim/results/ns3_ablation_180runs.jsonl", None)]
    seen = {}
    for f, force in files:
        if not os.path.exists(f):
            continue
        for l in open(f):
            r = json.loads(l)
            r["detector"] = force or NAME.get(r["detector"], r["detector"])
            seen[(r["detector"], r["attack"], r["speed"], r["seed"])] = r
    return list(seen.values())


def boot(vals, fn=np.mean):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(vals), (B, len(vals)))
    bs = fn(vals[idx], axis=1)
    return (float(fn(vals[None, :], axis=1)[0]), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def boot_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(num), (B, len(num)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return (float(num.sum() / max(den.sum(), 1)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def metrics(rs):
    ben = [r for r in rs if r["attack"] == "benign"]; atk = [r for r in rs if r["attack"] != "benign"]
    return {
        "fpr": boot_ratio([r["honest_flags"] for r in ben], [r["honest_pairs"] for r in ben]),
        "det": boot([0.0 if np.isnan(r["delay"]) else 1.0 for r in atk]),
        "delay": boot([r["delay"] for r in atk]),
        "iso": boot([r["honest_isolated"] for r in rs]),
        "pdr_atk": boot([r["pdr"] for r in atk]),
        "pdr_ben": boot([r["pdr"] for r in ben]),
        "fconv": boot([r["false_conv"] for r in rs]),
        "n": len(rs), "seeds": sorted({r["seed"] for r in rs}),
    }


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def common_seeds(R, dets):
    """Seeds for which every listed detector has all 30 scenarios (6 conditions x 5 speeds)."""
    from collections import Counter
    c = Counter((r["detector"], r["seed"]) for r in R)
    seeds = sorted({r["seed"] for r in R})
    return [sd for sd in seeds if all(c[(d, sd)] == 30 for d in dets)]


def analyse(R=None):
    R = R or load_all()
    present = [d for d in dict.fromkeys(MAIN + ABL) if any(r["detector"] == d for r in R)]
    keep = set(common_seeds(R, present))
    R = [r for r in R if r["seed"] in keep]
    by = lambda d, **k: [r for r in R if r["detector"] == d and all(r[a] == v for a, v in k.items())]
    out = {"overall": {}, "speed": {}, "attack": {}, "tests": {}}
    for d in dict.fromkeys(MAIN + ABL):
        rs = by(d)
        if not rs:
            continue
        out["overall"][d] = metrics(rs)
        out["speed"][d] = {s: metrics(by(d, speed=s)) for s in SPEEDS}
        out["attack"][d] = {a: metrics(by(d, attack=a)) for a in ATT}
    # paired tests vs MERIT-Net on honest isolation and attack-run PDR
    key = lambda r: (r["attack"], r["speed"], r["seed"])
    m = {key(r): r for r in by("MERIT-Net")}
    for col in ("honest_isolated", "pdr"):
        names, ps, info = [], [], {}
        for d in MAIN + ABL[1:]:
            if d == "MERIT-Net":
                continue
            o = {key(r): r for r in by(d)}
            ks = [k for k in m if k in o and (col != "pdr" or k[0] != "benign")]
            if len(ks) < 5:
                continue
            a = np.array([m[k][col] for k in ks]); b = np.array([o[k][col] for k in ks])
            p = wilcoxon(a, b).pvalue if (a != b).any() else 1.0
            names.append(d); ps.append(p); info[d] = dict(n=len(ks), merit=float(a.mean()), other=float(b.mean()), p=float(p))
        if ps:
            base = [i for i, n in enumerate(names) if n in MAIN]
            adj = holm(np.array([ps[i] for i in base]))
            for i, a in zip(base, adj):
                info[names[i]]["p_holm"] = float(a)
        out["tests"][col] = info
    return out


if __name__ == "__main__":
    o = analyse()
    print("seeds used:", o["overall"]["MERIT-Net"]["seeds"])
    json.dump(o, open("stats.json", "w"), indent=1)
    f = lambda t, k=3: f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"
    for d, v in o["overall"].items():
        print(f"{d:18s} n={v['n']:3d} seeds={len(v['seeds'])}  FPR {f(v['fpr'],4)}  det {f(v['det'],2)}  "
              f"delay {f(v['delay'],1)}  iso {f(v['iso'],2)}  PDR {f(v['pdr_atk'])}")
    for col, info in o["tests"].items():
        for d, t in info.items():
            print(f"  {col:15s} vs {d:18s} n={t['n']} {t['merit']:.3f} vs {t['other']:.3f} p={t['p']:.2e} holm={t.get('p_holm', float('nan')):.2e}")
