"""Generate and package the MANET-FLOOD NS-3 dataset (resumable, configurable).

Runs the NS-3 scenario over a grid of {attack x speed x nodes x seed}, collects
the per-window 23-feature tensors with per-node attack labels, and writes:

  out/raw/<attack>_v<speed>_n<nodes>_s<seed>.npz   full (T, N, N, 23) tensor + meta
  out/manet_flood.csv                              flat per-(observer,neighbour,window) rows
  out/DATASHEET.md                                 dataset documentation (written once)

Resumable: a run whose .npz already exists is skipped, so the grid can be
extended or restarted freely.  The flat CSV is rebuilt from all raw files at the
end, so it never duplicates rows.

Usage:
  python3 dataset_gen.py --speeds 0,5,10,15,20 --nodes 50 --seeds 1-6 --workers 2
  python3 dataset_gen.py --csv-only            # just rebuild the CSV from out/raw
"""
import os, sys, argparse, subprocess, time, glob
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "meritnet"))
from sim import NF, F_MASK, F_ACTFLOW, F_NEWFLOW

BIN = os.path.join(os.path.dirname(__file__), "meritnet-scenario")
ATTACKS = ["benign", "rreq", "rreq_stealth", "hello", "data", "syn"]
CLASS = {"benign": 0, "rreq": 1, "rreq_stealth": 1, "hello": 2, "data": 3, "syn": 4}
RATE = {"rreq": 20, "rreq_stealth": 20, "hello": 20, "data": 40, "syn": 20, "benign": 0}
OUT = os.path.join(os.path.dirname(__file__), "out")
ASTART, SIM = 60, 200
FEAT = ["RREQ","HELLO","DATA","SYN","RERR","HMISS","DEGJ","CHURN","NEWFLOW","ACTFLOW","LINKAGE",
        "DDIST","DEGI","COMP","FB","UNKNOWN","ENTROPY","DUP","HSYM","PERFLOW","FBRATIO","HALFOPEN","MASK"]


def raw_path(attack, speed, n, seed):
    return f"{OUT}/raw/{attack}_v{speed}_n{n}_s{seed}.npz"


def one(args):
    attack, speed, n, seed = args
    if os.path.exists(raw_path(attack, speed, n, seed)):
        return "skip"
    attackers = [] if attack == "benign" else ([3, 7] if attack == "data" else [3])
    binf = f"/tmp/ds_{attack}_{speed}_{n}_{seed}.bin"
    cmd = [BIN, f"--nNodes={n}", f"--speed={speed}", f"--sim={SIM}", f"--seed={seed}",
           f"--attack={attack}", f"--rate={RATE[attack]}", f"--aStart={ASTART}", f"--out={binf}"]
    if attackers:
        cmd.append("--attackers=" + ",".join(map(str, attackers)))
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL, timeout=600)
    except Exception as e:
        return f"FAIL {args}: {e}"
    X = np.fromfile(binf, np.float32).reshape(-1, n, n, NF)
    os.remove(binf)
    af = X[..., F_ACTFLOW]
    X[..., F_NEWFLOW] = np.maximum(0.0, af - np.concatenate([af[:1], af[:-1]], 0)) * (X[..., F_MASK] > 0)
    label_j = np.zeros(n, np.int64)
    for a in attackers:
        label_j[a] = CLASS[attack]
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    np.savez_compressed(raw_path(attack, speed, n, seed),
                        X=X.astype(np.float16), attackers=np.array(attackers),
                        label_j=label_j, speed=speed, attack=attack, astart=ASTART, nodes=n)
    return "ok"


def rebuild_csv():
    files = sorted(glob.glob(f"{OUT}/raw/*.npz"))
    header = "attack,speed,nodes,seed,window,observer,neighbour,label," + ",".join(FEAT)
    nrec = 0
    with open(f"{OUT}/manet_flood.csv", "w") as f:
        f.write(header + "\n")
        for path in files:
            d = np.load(path, allow_pickle=True)
            X = d["X"].astype(np.float32); label_j = d["label_j"]; att = set(d["attackers"].tolist())
            attack = d["attack"].item(); speed = int(d["speed"]); n = int(d["nodes"]) if "nodes" in d else X.shape[1]; astart = int(d["astart"])
            seed = int(path.split("_s")[-1].split(".")[0])
            rng = np.random.default_rng(seed * 100 + speed)
            for t in range(X.shape[0]):
                m = X[t, :, :, F_MASK] > 0
                lab = (label_j[None, :] if t >= astart else np.zeros(n, int)[None, :]) * m
                is_att = np.array([j in att for j in range(n)])[None, :] & m & (t >= astart)
                keep = is_att | (m & (rng.random(m.shape) < 0.02))
                ii, jj = np.nonzero(keep)
                for i, j in zip(ii, jj):
                    feat = ",".join(f"{v:.4g}" for v in X[t, i, j])
                    f.write(f"{attack},{speed},{n},{seed},{t},{i},{j},{int(lab[i, j])},{feat}\n")
                    nrec += 1
    print(f"CSV rebuilt: {nrec} records from {len(files)} runs", flush=True)


def parse_ints(s):
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", default="0,5,10,15,20")
    ap.add_argument("--nodes", default="50")
    ap.add_argument("--seeds", default="1-6")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    if a.csv_only:
        rebuild_csv(); return
    speeds, nodes, seeds = parse_ints(a.speeds), parse_ints(a.nodes), parse_ints(a.seeds)
    grid = [(at, sp, n, sd) for at in ATTACKS for sp in speeds for n in nodes for sd in seeds]
    todo = [g for g in grid if not os.path.exists(raw_path(*g))]
    print(f"grid {len(grid)} runs, {len(todo)} to do, {len(grid)-len(todo)} already present", flush=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(one, todo)):
            if r.startswith("FAIL"):
                print(" ", r, flush=True)
            if (k + 1) % 5 == 0 or k + 1 == len(todo):
                print(f"  {k+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    rebuild_csv()
    print(f"done {time.time()-t0:.0f}s, {len(glob.glob(f'{OUT}/raw/*.npz'))} runs total", flush=True)


if __name__ == "__main__":
    main()
