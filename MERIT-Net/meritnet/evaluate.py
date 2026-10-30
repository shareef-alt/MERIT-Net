"""Online comparison: every detector runs inside the simulator on identical scenarios."""
import json, math, os, sys, time
import numpy as np
import torch
import joblib
from multiprocessing import Pool

from sim import ManetSim, F_MASK, F_RREQ, F_HELLO, F_DATA, F_SYN
from models import W, prep, LAMBDA_IN, BASE_IN, LambdaNet, VNet, RefHybrid, CNNBiLSTM
from detectors import (NoDefense, StaticThreshold, PlainCUSUM, SeqClassifier, MERITNet, V_IN)

SPEEDS = [0, 5, 10, 15, 20]
SCENARIOS = [("benign", None), ("rreq", 20.0), ("rreq-low", 3.0), ("rreq-stealth", 3.0), ("hello", 10.0),
             ("data", 40.0), ("syn", 20.0)]
SEEDS = [100, 101, 102]
T, START = 200, 60
PKT_BITS = 512 * 8

_DET = None


def build_detectors():
    cfg = json.load(open("ckpt/config.json"))
    lam = LambdaNet(); lam.load_state_dict(torch.load("ckpt/lambda_net.pt")); lam.eval()
    vn = VNet(V_IN); vn.load_state_dict(torch.load("ckpt/v_net.pt")); vn.eval()
    ref = RefHybrid(); ref.load_state_dict(torch.load("ckpt/ref_hybrid.pt")); ref.eval()
    cbl = CNNBiLSTM(); cbl.load_state_dict(torch.load("ckpt/cnn_bilstm.pt")); cbl.eval()
    knn = joblib.load("ckpt/knn.joblib"); rf = joblib.load("ckpt/rf.joblib")
    glm = joblib.load("ckpt/glm.joblib")
    sc = np.array(cfg["lam_scale"], np.float32)
    gsc = np.array(cfg["glm_scale"], np.float32)

    def lam_nn(s):
        with torch.no_grad():
            return lam(torch.from_numpy(s)).numpy() * sc

    def lam_glm(s):
        z = s[:, -1, :]
        return np.stack([np.maximum(g.predict(z), 1e-3) for g in glm], 1) * gsc

    def p_knn(seq):
        f = np.log1p(seq[..., [F_RREQ, F_HELLO, F_DATA, F_SYN]]).reshape(len(seq), -1)
        return knn.predict(f) > 0

    def p_ref(seq):
        with torch.no_grad():
            return torch.sigmoid(ref(torch.from_numpy(prep(seq)[..., BASE_IN]))).numpy() > 0.5

    def p_rf(seq):
        with torch.no_grad():
            e = cbl.embed(torch.from_numpy(prep(seq)[..., BASE_IN])).numpy()
        return rf.predict(e) > 0

    eps = cfg["eps"]
    hov = cfg.get("h_override")
    disp = cfg.get("disp")
    geps = [min(max(e, 0.0) + 0.05, 0.25) for e in cfg.get("glm_eps_cal", [0.0, 0.0])]
    dets = {
        "AODV (no defense)": lambda: NoDefense(),
        "Static threshold": lambda: StaticThreshold(cfg["static_thr"]),
        "Plain CUSUM": lambda: PlainCUSUM(cfg["plain_lam"]),
        "FAPRP-style kNN": lambda: SeqClassifier("FAPRP-style kNN", p_knn),
        "CNN-BiLSTM-RF": lambda: SeqClassifier("CNN-BiLSTM-RF", p_rf),
        "Ref. CNN-LSTM/GRU": lambda: SeqClassifier("Ref. CNN-LSTM/GRU", p_ref),
        "MERIT-Net": lambda: MERITNet(lam, vn, eps[0], eps[1], cfg["tau_v"], lam_fn=lam_nn, h_override=hov, disp=disp),
        "MERIT linear λ": lambda: MERITNet(lam, vn, geps[0], geps[1], cfg["tau_v"], lam_fn=lam_glm, h_override=hov, disp=disp,
                                           name="MERIT linear λ"),
        "MERIT no V-Net": lambda: MERITNet(lam, vn, eps[0], eps[1], cfg["tau_v"], lam_fn=lam_nn,
                                           use_v=False, h_override=hov, disp=disp, name="MERIT no V-Net"),
        "MERIT no fusion": lambda: MERITNet(lam, vn, eps[0], eps[1], cfg["tau_v"], lam_fn=lam_nn,
                                            use_fusion=False, h_override=hov, disp=disp, name="MERIT no fusion"),
    }
    return dets


def init():
    global _DET
    torch.set_num_threads(1)
    _DET = build_detectors()


def run(job):
    det_name, scen, rate, speed, seed = job
    attack = "rreq" if scen.startswith("rreq") else scen
    sim = ManetSim(speed=speed, attack=attack, attack_rate=rate, seed=seed * 1000 + speed,
                   T=T, attack_start=START, stealth=(scen == "rreq-stealth"))
    det = _DET[det_name]()
    det.reset(sim.n)
    pi = np.ones((sim.n, sim.n)); conv = np.zeros((sim.n, sim.n), bool)
    hp = hf = ap = af = 0
    false_conv_events = 0
    honest_ever = np.zeros(sim.n, bool)
    prev_conv = np.zeros((sim.n, sim.n), bool)
    delay = np.nan
    t_det = 0.0
    for t in range(T):
        X = sim.step(pi, conv)
        t1 = time.perf_counter()
        flag, pi, conv = det.step(X)
        t_det += time.perf_counter() - t1
        if t < W:
            continue
        mask = X[..., F_MASK] > 0
        hon = mask & ~sim.is_attacker[None, :]
        hp += hon.sum(); hf += (flag & hon).sum()
        new_conv = conv & ~prev_conv
        false_conv_events += (new_conv & ~sim.is_attacker[None, :]).sum()
        honest_ever |= (conv & ~sim.is_attacker[None, :]).any(0)
        prev_conv = conv
        if sim.attackers and t >= START:
            att = mask & sim.is_attacker[None, :]
            ap += att.sum(); af += (flag & att).sum()
            if np.isnan(delay):
                ok = []
                for a in sim.attackers:
                    obs = mask[:, a]
                    if obs.sum() == 0:
                        ok.append(False)
                    else:
                        ok.append((conv[:, a] & obs).sum() >= 0.5 * obs.sum())
                if all(ok):
                    delay = t - START + 1
    m = sim.m
    return dict(detector=det_name, scenario=scen, speed=speed, seed=seed,
                honest_pairs=int(hp), honest_flags=int(hf), att_pairs=int(ap), att_flags=int(af),
                false_conv=int(false_conv_events), honest_isolated=int(honest_ever.sum()),
                delay=float(delay), pdr=float(sim.pdr()),
                overhead=float(m["ctrl_tx"] / max(m["data_recv"], 1)),
                throughput=float(m["data_recv"] * PKT_BITS / 1024 / T),
                ms_per_window=1000 * t_det / T)


if __name__ == "__main__":
    which = sys.argv[1:] or None
    names = list(build_detectors().keys())
    if which:
        names = [n for n in names if n in which]
    jobs = [(d, sc, r, sp, sd) for d in names for sc, r in SCENARIOS for sp in SPEEDS for sd in SEEDS]
    out = "results/runs.jsonl"
    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["detector"], r["scenario"], r["speed"], r["seed"]))
    jobs = [j for j in jobs if (j[0], j[1], j[3], j[4]) not in done]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with Pool(2, initializer=init) as p, open(out, "a") as f:
        for k, r in enumerate(p.imap_unordered(run, jobs, chunksize=2)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if k % 20 == 0:
                print(k, f"{time.time()-t0:.0f}s", r["detector"], r["scenario"], r["speed"],
                      f"pdr={r['pdr']:.3f} fpr={r['honest_flags']/max(r['honest_pairs'],1):.4f}", flush=True)
