"""Full-grid analysis: main comparison, per-attack, mobility trend, ablations,
significance tests. Reads results/ns3_full.jsonl (or results/ns3_runs.jsonl)."""
import json, os, sys
import numpy as np
from collections import defaultdict
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

path = "results/ns3_full.jsonl" if os.path.exists("results/ns3_full.jsonl") else "results/ns3_runs.jsonl"
R = [json.loads(l) for l in open(path)]
print(f"{len(R)} runs from {path}\n")

ORDER = ["Static threshold", "FAPRP-style kNN", "CNN-BiLSTM-RF", "Ref. CNN-LSTM/GRU",
         "MERIT-Net", "MERIT no V-Net", "MERIT no fusion", "MERIT linear λ"]
dets = [d for d in ORDER if any(r["detector"] == d for r in R)]
by = defaultdict(list)
for r in R:
    by[r["detector"]].append(r)


def fnum(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# 1. main comparison
print("== Main comparison (all runs) ==")
print(f"{'detector':20s} {'FPR':>8s} {'detect':>7s} {'delay':>7s} {'isolated/run':>12s} {'PDR_atk':>8s}")
for d in dets:
    rs = by[d]; ben = [r for r in rs if r["attack"] == "benign"]; atk = [r for r in rs if r["attack"] != "benign"]
    fpr = sum(r["honest_flags"] for r in ben) / max(sum(r["honest_pairs"] for r in ben), 1)
    det = np.mean([not np.isnan(r["delay"]) for r in atk]) if atk else float("nan")
    dl = np.nanmean([r["delay"] for r in atk]) if atk else float("nan")
    iso = np.mean([r["honest_isolated"] for r in rs])
    pdr = np.mean([r["pdr"] for r in atk]) if atk else float("nan")
    print(f"{d:20s} {fpr:8.4f} {det:7.2f} {dl:7.1f} {iso:12.2f} {pdr:8.3f}")

# 2. mobility trend: FPR and isolation vs speed (the speed-robustness claim)
print("\n== Benign FPR by speed (should stay flat for MERIT-Net) ==")
speeds = sorted(set(r["speed"] for r in R))
print(f"{'detector':20s} " + " ".join(f"v{s:>4d}" for s in speeds))
for d in dets:
    row = []
    for s in speeds:
        ben = [r for r in by[d] if r["attack"] == "benign" and r["speed"] == s]
        fpr = sum(r["honest_flags"] for r in ben) / max(sum(r["honest_pairs"] for r in ben), 1)
        row.append(fpr)
    print(f"{d:20s} " + " ".join(f"{v:5.3f}" for v in row))

# 3. per-attack detection rate
print("\n== Detection rate per attack ==")
atks = [a for a in ["rreq", "rreq_stealth", "hello", "data", "syn"] if any(r["attack"] == a for r in R)]
print(f"{'detector':20s} " + " ".join(f"{a[:8]:>9s}" for a in atks))
for d in dets:
    row = [np.mean([not np.isnan(r["delay"]) for r in by[d] if r["attack"] == a]) for a in atks]
    print(f"{d:20s} " + " ".join(f"{v:9.2f}" for v in row))

# 4. significance: MERIT-Net vs each baseline on false isolation and PDR (paired by scenario)
if wilcoxon and "MERIT-Net" in by:
    print("\n== MERIT-Net vs baselines (paired Wilcoxon) ==")
    key = lambda r: (r["attack"], r["speed"], r["seed"])
    m = {key(r): r for r in by["MERIT-Net"]}
    for b in dets:
        if b == "MERIT-Net" or b not in by:
            continue
        o = {key(r): r for r in by[b]}
        common = [k for k in m if k in o]
        for col in ["honest_isolated", "pdr"]:
            a = np.array([m[k][col] for k in common]); c = np.array([o[k][col] for k in common])
            ok = ~(np.isnan(a) | np.isnan(c))
            try:
                p = wilcoxon(a[ok], c[ok]).pvalue if (a[ok] != c[ok]).any() else 1.0
            except Exception:
                p = float("nan")
            print(f"  vs {b:20s} {col:16s} MERIT {np.nanmean(a):.3f} vs {np.nanmean(c):.3f}  p={p:.2e}")

# 5. ablations
print("\n== Ablations (MERIT variants) ==")
for d in dets:
    if not d.startswith("MERIT"):
        continue
    rs = by[d]; atk = [r for r in rs if r["attack"] != "benign"]
    print(f"{d:20s} detect={np.mean([not np.isnan(r['delay']) for r in atk]):.2f} "
          f"delay={np.nanmean([r['delay'] for r in atk]):.1f} "
          f"isolated/run={np.mean([r['honest_isolated'] for r in rs]):.2f} "
          f"falseConv/run={np.mean([r['false_conv'] for r in rs]):.1f}")
