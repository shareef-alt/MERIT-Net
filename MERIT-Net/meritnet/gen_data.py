"""Generate training / validation sequences from undefended simulation runs.

Each sample = observer i's last W windows of features about neighbour j.
Saved per split to data/<split>.npz with X (S, W, NF) float16, y (attack class of j),
speed, run id.
"""
import sys, os, time
import numpy as np
from multiprocessing import Pool
from sim import ManetSim, NF, F_MASK, ATTACK_TYPES
W = 16

SPEEDS = [0, 5, 10, 15, 20]
ATTACKS = ["benign", "rreq", "hello", "data", "syn"]
T, START = 200, 60
P_HONEST = 0.03


def one_run(args):
    speed, attack, seed = args
    rng = np.random.default_rng(seed * 7919 + speed * 31 + ATTACK_TYPES[attack])
    sim = ManetSim(speed=speed, attack=attack, seed=seed * 1000 + speed * 10 + ATTACK_TYPES[attack],
                   T=T, attack_start=START)
    ring = np.zeros((W, sim.n, sim.n, NF), np.float32)
    Xs, ys = [], []
    for t in range(T):
        X = sim.step()
        ring[t % W] = X
        if t < W - 1:
            continue
        lab = sim.labels()
        mask = X[..., F_MASK] > 0
        att_cols = sim.is_attacker[None, :] & mask
        keep = att_cols | (mask & (rng.random(mask.shape) < P_HONEST))
        ii, jj = np.nonzero(keep)
        if len(ii) == 0:
            continue
        order = [(t - W + 1 + k) % W for k in range(W)]
        seq = ring[order][:, ii, jj, :].transpose(1, 0, 2)      # (S, W, NF)
        Xs.append(seq.astype(np.float16))
        ys.append(lab[jj])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y, np.full(len(y), speed, np.int8)


def build(split, seeds, workers=2):
    jobs = [(s, a, sd) for s in SPEEDS for a in ATTACKS for sd in seeds]
    t0 = time.time()
    with Pool(workers) as p:
        res = p.map(one_run, jobs)
    X = np.concatenate([r[0] for r in res]); y = np.concatenate([r[1] for r in res])
    sp = np.concatenate([r[2] for r in res])
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(f"data/{split}.npz", X=X, y=y, speed=sp)
    print(split, X.shape, np.bincount(y), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    build("train", [0, 1, 2])
    build("val", [3])
