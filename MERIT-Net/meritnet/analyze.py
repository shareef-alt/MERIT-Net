"""Aggregate results/runs.jsonl into the comparison tables."""
import json, math
import numpy as np
import pandas as pd
from models import (LambdaNet, VNet, RefHybrid, CNNBiLSTM, n_params, macs_lambda, macs_vnet,
                    macs_cnnlstm, macs_cnngru, macs_cnnbilstm)
from detectors import V_IN

df = pd.DataFrame([json.loads(l) for l in open("results/runs.jsonl")])
ORDER = ["AODV (no defense)", "Static threshold", "Plain CUSUM", "FAPRP-style kNN", "CNN-BiLSTM-RF",
         "Ref. CNN-LSTM/GRU", "MERIT-Net", "MERIT linear λ", "MERIT no V-Net", "MERIT no fusion"]
ATT = ["rreq", "rreq-low", "rreq-stealth", "hello", "data", "syn"]
df["is_att"] = df.scenario != "benign"
out = {}


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


# 1. pair-level detection quality over all scenarios
rows = []
for d in ORDER:
    s = df[df.detector == d]
    TP = s.att_flags.sum(); FN = s.att_pairs.sum() - TP
    FP = s.honest_flags.sum(); TN = s.honest_pairs.sum() - FP
    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    a = s[s.is_att]
    det_rate = a.delay.notna().mean()
    rows.append(dict(detector=d, precision=prec, recall=rec, f1=f1, accuracy=acc,
                     fpr=FP / max(FP + TN, 1), detect_rate=det_rate,
                     delay=a.delay.mean(), false_conv=s.false_conv.mean(),
                     isolated=s.honest_isolated.mean()))
t1 = pd.DataFrame(rows)
out["overall"] = t1.to_dict("records")

# 2. benign false convictions and FPR by speed
b = df[df.scenario == "benign"]
t2 = b.groupby(["detector", "speed"]).agg(false_conv=("false_conv", "mean"),
                                          isolated=("honest_isolated", "mean"),
                                          hf=("honest_flags", "sum"), hp=("honest_pairs", "sum")).reset_index()
t2["fpr"] = t2.hf / t2.hp
out["benign_by_speed"] = t2.to_dict("records")

# 3. per-attack detection
t3 = df[df.is_att].groupby(["detector", "scenario"]).agg(
    detect_rate=("delay", lambda x: x.notna().mean()), delay=("delay", "mean"),
    tpr_num=("att_flags", "sum"), tpr_den=("att_pairs", "sum")).reset_index()
t3["tpr"] = t3.tpr_num / t3.tpr_den
out["per_attack"] = t3.to_dict("records")

# 4. network performance
t4 = df.groupby(["detector", "is_att"]).agg(pdr=("pdr", "mean"), overhead=("overhead", "mean"),
                                            throughput=("throughput", "mean")).reset_index()
out["network"] = t4.to_dict("records")
t4s = df.groupby(["detector", "scenario"]).agg(pdr=("pdr", "mean")).reset_index()
out["pdr_by_scenario"] = t4s.to_dict("records")
t4v = df[df.is_att].groupby(["detector", "speed"]).agg(pdr=("pdr", "mean")).reset_index()
out["pdr_attack_by_speed"] = t4v.to_dict("records")

# 5. efficiency
cfg = json.load(open("ckpt/config.json"))
lam, vn = LambdaNet(), VNet(V_IN)
eff = {
    "MERIT-Net": dict(params=n_params(lam) + n_params(vn), always_on_macs=macs_lambda(lam),
                      vnet_macs=macs_vnet(vn), kb_int8=(n_params(lam) + n_params(vn)) / 1024),
    "Ref. CNN-LSTM/GRU": dict(params=n_params(RefHybrid()), always_on_macs=macs_cnnlstm() + macs_cnngru(),
                              kb_int8=n_params(RefHybrid()) / 1024),
    "CNN-BiLSTM-RF": dict(params=n_params(CNNBiLSTM()), always_on_macs=macs_cnnbilstm(),
                          kb_int8=n_params(CNNBiLSTM()) / 1024, note="+ 100-tree random forest"),
}
ms = df.groupby("detector").ms_per_window.mean().to_dict()
for k in eff:
    eff[k]["ms_per_window_all_pairs"] = ms.get(k)
out["efficiency"] = eff
out["ms_per_window"] = ms
out["config"] = cfg
json.dump(out, open("results/summary.json", "w"), indent=1, default=float)

pd.set_option("display.width", 200)
print(t1.round(4).to_string(index=False))
print("\nBenign false convictions per run by speed")
print(t2.pivot(index="detector", columns="speed", values="false_conv").reindex(ORDER).round(2))
print("\nBenign pair FPR by speed")
print(t2.pivot(index="detector", columns="speed", values="fpr").reindex(ORDER).round(5))
print("\nDetection rate per attack")
print(t3.pivot(index="detector", columns="scenario", values="detect_rate").reindex(ORDER).round(2))
print("\nMean delay (s) per attack")
print(t3.pivot(index="detector", columns="scenario", values="delay").reindex(ORDER).round(1))
print("\nPDR by scenario")
print(t4s.pivot(index="detector", columns="scenario", values="pdr").reindex(ORDER).round(3))
print("\nNetwork (attack runs)")
print(t4[t4.is_att].set_index("detector").reindex(ORDER).round(3))
print("\nPDR under attack by speed")
print(t4v.pivot(index="detector", columns="speed", values="pdr").reindex(ORDER).round(3))
print("\nEfficiency", json.dumps(eff, indent=1, default=float))
print("ms/window", {k: round(v, 2) for k, v in ms.items()})
