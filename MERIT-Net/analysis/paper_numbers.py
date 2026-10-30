"""Every number quoted in the manuscript prose, computed from stats.json + the raw runs.
build_ieee.py and make_figs.py import N from here, so text, tables and figures cannot disagree."""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # run from anywhere
import json, numpy as np
from stats import load_all, common_seeds, MAIN, ABL

ST = json.load(open("stats.json"))
O, SP, AT, TS = ST["overall"], ST["speed"], ST["attack"], ST["tests"]
_R = load_all()
_keep = set(common_seeds(_R, [d for d in dict.fromkeys(MAIN + ABL) if any(r["detector"] == d for r in _R)]))
R = [r for r in _R if r["seed"] in _keep]
SPEEDS = ["0", "5", "10", "15", "20"]
REF, MN = "Ref. CNN-LSTM/GRU", "MERIT-Net"
LEARN = ["FAPRP-style kNN", "CNN-BiLSTM-RF", REF]
BASE_DET = ["Flow-based CUSUM"] + LEARN          # baselines that detect attacks


def f(x, k=2): return f"{x:.{k}f}"
def ci(t, k=2): return f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"
def rng(vals, k=1): return f"{min(vals):.{k}f} to {max(vals):.{k}f}"


def p(x):
    if x < 1e-3:
        e = int(np.floor(np.log10(x))); m = x / 10 ** e
        return f"{m:.1f}×10" + "".join("⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".index(c)] for c in str(e))
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def sp(d, key, s): return SP[d][s][key][0]


N = {}
N["n_seeds"] = len(O[MN]["seeds"]); N["n_det_runs"] = O[MN]["n"]
N["n_main_runs"] = sum(O[d]["n"] for d in MAIN); N["n_abl_runs"] = sum(O[d]["n"] for d in ABL[1:])
N["n_total_runs"] = N["n_main_runs"] + N["n_abl_runs"]
ben = [r for r in R if r["detector"] == MN and r["attack"] == "benign"]
N["n_ben"] = len(ben); N["ben_zero"] = sum(r["honest_flags"] == 0 for r in ben)
worst = max(ben, key=lambda r: r["honest_flags"])
N["worst_share"] = round(100 * worst["honest_flags"] / max(1, sum(r["honest_flags"] for r in ben)))
N["worst_speed"], N["worst_iso"] = worst["speed"], worst["honest_isolated"]
N["fpr_m"] = ci(O[MN]["fpr"], 4); N["fpr_ref"] = ci(O[REF]["fpr"], 4)
N["fpr_ratio"] = f(O[REF]["fpr"][0] / O[MN]["fpr"][0], 0)
N["fpr_m_moving_max"] = f(max(sp(MN, "fpr", s) for s in SPEEDS[1:]), 4)
N["fpr_m_static"] = f(sp(MN, "fpr", "0"), 4)
N["fpr_cusum_static"] = f(sp("Flow-based CUSUM", "fpr", "0"), 3)
N["iso_m"] = f(O[MN]["iso"][0]); N["iso_ref"] = f(O[REF]["iso"][0])
N["iso_ratio"] = f(O[REF]["iso"][0] / O[MN]["iso"][0], 0)
N["iso_other_rng"] = rng([O[d]["iso"][0] for d in ["CNN-BiLSTM-RF", "FAPRP-style kNN", "Flow-based CUSUM"]], 0)
N["iso_other_btw"] = "between " + N["iso_other_rng"].replace(" to ", " and ")
N["iso_cusum"] = f(O["Flow-based CUSUM"]["iso"][0], 1)
N["p_iso_max_holm"] = p(max(TS["honest_isolated"][d]["p_holm"] for d in BASE_DET))
N["p_iso_cusum"] = p(TS["honest_isolated"]["Flow-based CUSUM"]["p"])
N["det_m"] = f(O[MN]["det"][0]); N["det_static"] = f(100 * O["Static threshold"]["det"][0], 0)
N["det_m_stealth"] = f(AT[MN]["rreq_stealth"]["det"][0])
N["delay_m"] = f(O[MN]["delay"][0], 1); N["delay_cusum"] = f(O["Flow-based CUSUM"]["delay"][0], 1)
N["delay_deep_rng"] = rng([O[d]["delay"][0] for d in ["CNN-BiLSTM-RF", REF]], 1)
N["pdr_m"] = f(O[MN]["pdr_atk"][0], 3); N["pdr_ref"] = f(O[REF]["pdr_atk"][0], 3)
# VII-D: wrongful isolation by speed
N["iso_learn_static_rng"] = rng([sp(d, "iso", "0") for d in LEARN], 1)
N["iso_cusum_static"] = f(sp("Flow-based CUSUM", "iso", "0"), 1)
N["iso_moving_rng"] = rng([sp(d, "iso", s) for d in BASE_DET for s in SPEEDS[1:]], 1)
N["iso_m_rng"] = rng([sp(MN, "iso", s) for s in SPEEDS], 1)
N["iso_m_rng_dash"] = N["iso_m_rng"].replace(" to ", "–")
fc20 = {d: sp(d, "fconv", "20") for d in [MN] + BASE_DET}
N["fc20_m"] = f(fc20[MN], 1); N["fc20_ref"] = f(fc20[REF], 1); N["fc20_bil"] = f(fc20["CNN-BiLSTM-RF"], 1)
N["fc20_cusum"] = f(fc20["Flow-based CUSUM"], 0); N["fc20_knn"] = f(fc20["FAPRP-style kNN"], 0)
# VII-E: packet delivery
N["pdrb_m_rng"] = rng([sp(MN, "pdr_ben", s) for s in SPEEDS], 3)
N["pdrb_static_rng"] = rng([sp("Static threshold", "pdr_ben", s) for s in SPEEDS], 3)
N["pdrb_deep_loss"] = f(max(sp("Static threshold", "pdr_ben", s) - sp(d, "pdr_ben", s)
                            for d in ["CNN-BiLSTM-RF", REF] for s in SPEEDS), 2)
N["pdrb_cusum_min"] = f(min(sp("Flow-based CUSUM", "pdr_ben", s) for s in SPEEDS), 2)
N["pdrb_knn_min"] = f(min(sp("FAPRP-style kNN", "pdr_ben", s) for s in SPEEDS), 2)
N["pdra_m_rng"] = rng([sp(MN, "pdr_atk", s) for s in SPEEDS], 3)
best_other = [max(sp(d, "pdr_atk", s) for d in BASE_DET) for s in SPEEDS]
N["pdra_m_best_everywhere"] = all(sp(MN, "pdr_atk", s) >= b for s, b in zip(SPEEDS, best_other))
gap = [sp("Static threshold", "pdr_atk", s) - sp(MN, "pdr_atk", s) for s in SPEEDS]
N["pdra_static_gap_rng"] = rng(gap, 3)
# VII-F: delay per attack
dl = lambda d, a: AT[d][a]["delay"][0]
N["dl_m_rreq"], N["dl_m_stealth"], N["dl_m_data"], N["dl_m_syn"] = (f(dl(MN, a), 1) for a in ["rreq", "rreq_stealth", "data", "syn"])
N["dl_deep_rng"] = rng([dl(d, a) for d in ["CNN-BiLSTM-RF", REF] for a in ["rreq", "rreq_stealth", "data"]], 1)
N["dl_cusum_syn"], N["dl_knn_syn"] = f(dl("Flow-based CUSUM", "syn"), 1), f(dl("FAPRP-style kNN", "syn"), 1)
N["dl_deep_syn_rng"] = rng([dl(d, "syn") for d in ["CNN-BiLSTM-RF", REF]], 1)
N["fpr_m0"] = f(O[MN]["fpr"][0], 4)
N["dl_hello_max"] = f(max(dl(d, "hello") for d in [MN] + BASE_DET), 1)
N["n_atk_runs_fig6"] = sum(O[d]["n"] - N["n_ben"] for d in [MN] + BASE_DET)
# ablation
A = {d: O[d] for d in ABL}
TI = TS["honest_isolated"]
N["abl_rows"] = [
    ["MERIT-Net (full)", ci(A[MN]["det"]), ci(A[MN]["delay"], 1), ci(A[MN]["iso"]), ci(A[MN]["fconv"], 1), "—"],
    ["No V-Net", ci(A["MERIT no V-Net"]["det"]), ci(A["MERIT no V-Net"]["delay"], 1), ci(A["MERIT no V-Net"]["iso"]), ci(A["MERIT no V-Net"]["fconv"], 1), p(TI["MERIT no V-Net"]["p"])],
    ["Linear λ (no λ-Net)", ci(A["MERIT linear λ"]["det"]), ci(A["MERIT linear λ"]["delay"], 1), ci(A["MERIT linear λ"]["iso"]), ci(A["MERIT linear λ"]["fconv"], 1), p(TI["MERIT linear λ"]["p"])],
    ["No fusion", ci(A["MERIT no fusion"]["det"]), ci(A["MERIT no fusion"]["delay"], 1), ci(A["MERIT no fusion"]["iso"]), ci(A["MERIT no fusion"]["fconv"], 1), p(TI["MERIT no fusion"]["p"])],
]
for k, d in (("nov", "MERIT no V-Net"), ("nof", "MERIT no fusion"), ("lin", "MERIT linear λ")):
    N[f"{k}_det"] = f(A[d]["det"][0]); N[f"{k}_delay"] = f(A[d]["delay"][0], 1)
    N[f"{k}_iso"] = f(A[d]["iso"][0]); N[f"{k}_fconv"] = f(A[d]["fconv"][0], 1); N[f"{k}_p"] = p(TI[d]["p"])
    benr = [r for r in R if r["detector"] == d and r["attack"] == "benign"]
    N[f"{k}_ben_flagged"] = sum(r["honest_flags"] > 0 for r in benr)
N["m_ben_flagged"] = N["n_ben"] - N["ben_zero"]
N["m_fconv"] = f(A[MN]["fconv"][0], 1)
N["nov_det_rreq"] = f(AT["MERIT no V-Net"]["rreq"]["det"][0])
N["nov_delay_both"] = None   # filled below: delay on attacks both variants detect
key = lambda r: (r["attack"], r["speed"], r["seed"])
m = {key(r): r for r in R if r["detector"] == MN and r["attack"] != "benign"}
o = {key(r): r for r in R if r["detector"] == "MERIT no V-Net" and r["attack"] != "benign"}
both = [k for k in m if k in o and not np.isnan(m[k]["delay"]) and not np.isnan(o[k]["delay"])]
from scipy.stats import wilcoxon
a_ = np.array([m[k]["delay"] for k in both]); b_ = np.array([o[k]["delay"] for k in both])
N["nov_delay_both"] = f(b_.mean(), 1); N["m_delay_both"] = f(a_.mean(), 1); N["p_delay_vnet"] = p(wilcoxon(a_, b_).pvalue)

if __name__ == "__main__":
    for k, v in N.items():
        print(f"{k:24s} {v}")
