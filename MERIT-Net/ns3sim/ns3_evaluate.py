"""Closed-loop comparison of detectors in NS-3 (parallel, configurable subset).

For each (detector, attack, speed, seed) it runs NS-3 with the detector in the
loop (bridge) and records detection quality, false isolation, delay and PDR.
Runs are independent, so they are parallelised across workers; each worker
builds its own detectors (torch models are not picklable).

Env knobs for a narrowed first pass:
  EVAL_DETECTORS  comma list (default the 5 core detectors)
  EVAL_SPEEDS     comma list (default 0,20)
  EVAL_SEEDS      comma list (default 11)
  EVAL_SIM        sim seconds (default 160)
  EVAL_WORKERS    parallel workers (default 2)

Run after training the models on NS-3 data (meritnet/train.py).
Writes results/ns3_runs.jsonl (resumable).
"""
import os, sys, json, struct, subprocess, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "meritnet"))
from sim import NF, F_MASK, F_ACTFLOW, F_NEWFLOW

BIN = os.path.join(os.path.dirname(__file__), "meritnet-scenario")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mn_eval", os.path.join(os.path.dirname(__file__), "..", "meritnet", "evaluate.py"))

SCEN = [("benign", 0), ("rreq", 20), ("rreq_stealth", 20), ("hello", 20), ("data", 40), ("syn", 20)]
CLASS = {"benign": 0, "rreq": 1, "rreq_stealth": 1, "hello": 2, "data": 3, "syn": 4}
DEFAULT_DETS = ["Static threshold", "FAPRP-style kNN", "CNN-BiLSTM-RF",
                "Ref. CNN-LSTM/GRU", "MERIT-Net"]
SPEEDS = [int(x) for x in os.environ.get("EVAL_SPEEDS", "0,20").split(",")]
SEEDS = [int(x) for x in os.environ.get("EVAL_SEEDS", "11").split(",")]
DETS = os.environ.get("EVAL_DETECTORS", ",".join(DEFAULT_DETS)).split(",")
SIM = int(os.environ.get("EVAL_SIM", "160"))
WORKERS = int(os.environ.get("EVAL_WORKERS", "2"))
ASTART, NNODES = 60, 50

_FACTORY = None


def _init():
    global _FACTORY
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "meritnet"))  # ckpt/ is relative here
    import importlib.util as u
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    _FACTORY = m.build_detectors()
    import torch
    torch.set_num_threads(1)


def run_job(job):
    name, attack, rate, speed, seed = job
    detector = _FACTORY[name]()
    attackers = [] if attack == "benign" else ([3, 7] if attack == "data" else [3])
    cmd = [BIN, f"--nNodes={NNODES}", f"--speed={speed}", f"--sim={SIM}", f"--seed={seed}",
           f"--attack={attack}", f"--rate={rate}", f"--aStart={ASTART}", "--bridge=1"]
    if attackers:
        cmd.append("--attackers=" + ",".join(map(str, attackers)))
    errf = open(f"/tmp/ns3err_{name[:4]}_{attack}_{speed}_{seed}.txt", "w+")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=errf, bufsize=0)   # stderr -> file, never fills a pipe
    detector.reset(NNODES)
    nb = NNODES * NNODES * NF * 4
    prev_af = np.zeros((NNODES, NNODES), np.float32)
    is_att = np.zeros(NNODES, bool)
    for a in attackers:
        is_att[a] = True
    hp = hf = ap = afl = fc = 0
    isolated = np.zeros(NNODES, bool)
    prev_conv = np.zeros((NNODES, NNODES), bool)
    delay = np.nan
    t = 0
    while True:
        hdr = p.stdout.read(4)
        if len(hdr) < 4:
            break
        buf = b""
        while len(buf) < nb:
            c = p.stdout.read(nb - len(buf))
            if not c:
                break
            buf += c
        if len(buf) < nb:
            break
        X = np.frombuffer(buf, np.float32).reshape(NNODES, NNODES, NF).copy()
        m = X[..., F_MASK] > 0
        X[..., F_NEWFLOW] = np.maximum(0.0, X[..., F_ACTFLOW] - prev_af) * m
        prev_af = X[..., F_ACTFLOW].copy()
        flag, pi, convict = detector.step(X)
        # blacklist convicted sources (drop all); throttle RREQs of pre-alarm sources
        cv_i, cv_j = np.nonzero(convict & m)
        thr = (~convict) & (pi < 1.0) & m
        th_i, th_j = np.nonzero(thr)
        p.stdin.write(struct.pack("<I", len(cv_i) + len(th_i)))
        for i, j in zip(cv_i, cv_j):
            p.stdin.write(struct.pack("<IIff", int(i), int(j), 1.0, 0.0))
        for i, j in zip(th_i, th_j):
            p.stdin.write(struct.pack("<IIff", int(i), int(j), float(1.0 - pi[i, j]), 1.0))
        p.stdin.flush()
        if t >= 16:
            hon = m & ~is_att[None, :]
            hp += int(hon.sum()); hf += int((flag & hon).sum())
            nc = convict & ~prev_conv
            fc += int((nc & ~is_att[None, :]).sum())
            isolated |= (convict & ~is_att[None, :]).any(0)
            prev_conv = convict
            if attackers and t >= ASTART:
                at = m & is_att[None, :]
                ap += int(at.sum()); afl += int((flag & at).sum())
                if np.isnan(delay):
                    if all((convict[:, a] & m[:, a]).sum() >= 0.5 * max(m[:, a].sum(), 1) for a in attackers):
                        delay = t - ASTART + 1
        t += 1
    try:
        p.stdin.close()
    except Exception:
        pass
    p.wait()
    errf.seek(0); err = errf.read(); errf.close()
    try:
        os.remove(errf.name)
    except Exception:
        pass
    pdr = 0.0
    for tok in err.split():
        if tok.startswith("pdr="):
            pdr = float(tok[4:])
    return dict(detector=name, attack=attack, speed=speed, seed=seed,
                honest_pairs=hp, honest_flags=hf, att_pairs=ap, att_flags=afl,
                false_conv=fc, honest_isolated=int(isolated.sum()),
                delay=float(delay), pdr=pdr)


# Cluster sharding: SLURM array task i of N runs shard i (EVAL_SHARD, EVAL_NSHARDS).
# Each shard writes its own results file so array tasks never contend; merge after.
SHARD = int(os.environ.get("EVAL_SHARD", "0"))
NSHARDS = int(os.environ.get("EVAL_NSHARDS", "1"))


def main():
    os.makedirs("results", exist_ok=True)
    out = f"results/ns3_runs_{SHARD:03d}.jsonl" if NSHARDS > 1 else "results/ns3_runs.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done.add((r["detector"], r["attack"], r["speed"], r["seed"]))
    full = [(d, a, rate, sp, sd) for d in DETS for (a, rate) in SCEN
            for sp in SPEEDS for sd in SEEDS]
    shard = [j for n, j in enumerate(full) if n % NSHARDS == SHARD]   # deterministic split
    jobs = [j for j in shard if (j[0], j[1], j[3], j[4]) not in done]
    print(f"shard {SHARD}/{NSHARDS}: {len(jobs)}/{len(shard)} jobs to do "
          f"(grid {len(full)} total), sim={SIM}s, {WORKERS} workers", flush=True)
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool, open(out, "a") as f:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            f.write(json.dumps(r) + "\n"); f.flush()
            fpr = r["honest_flags"] / max(r["honest_pairs"], 1)
            print(f"{time.time()-t0:.0f}s [{k+1}/{len(jobs)}] {r['detector']:18s} {r['attack']:12s} "
                  f"v{r['speed']} pdr={r['pdr']:.3f} fpr={fpr:.4f} delay={r['delay']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
