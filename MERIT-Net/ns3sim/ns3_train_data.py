"""Turn NS-3 dataset tensors (out/raw/*.npz) into the W-window training sequences
that meritnet/train.py consumes (data/train.npz, data/val.npz).

Each sample = observer i's last W windows about neighbour j, with j's attack
label.  Splits are by seed so train/val/test never share a run.  Point train.py
at the meritnet/data produced here and the whole model pipeline runs on NS-3
traffic unchanged.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "meritnet"))
from sim import NF, F_MASK, ATTACK_TYPES

W = 16
P_HONEST = 0.03
RAW = os.path.join(os.path.dirname(__file__), "out", "raw")
DATA = os.path.join(os.path.dirname(__file__), "..", "meritnet", "data")


def sequences_from_run(path, rng):
    d = np.load(path)
    X = d["X"].astype(np.float32)           # (T, n, n, 23)
    label_j = d["label_j"]
    astart = int(d["astart"])
    T, n = X.shape[0], X.shape[1]
    Xs, ys, sp = [], [], []
    speed = int(d["speed"])
    for t in range(W - 1, T):
        m = X[t, :, :, F_MASK] > 0
        lab = (label_j[None, :] if t >= astart else np.zeros(n, int)[None, :]) * m
        is_att = (label_j[None, :] > 0) & m & (t >= astart)
        keep = is_att | (m & (rng.random(m.shape) < P_HONEST))
        ii, jj = np.nonzero(keep)
        if len(ii) == 0:
            continue
        seq = X[t - W + 1:t + 1][:, ii, jj, :].transpose(1, 0, 2)   # (S, W, 23)
        Xs.append(seq.astype(np.float16))
        ys.append(lab[ii, jj])
        sp.append(np.full(len(ii), speed, np.int8))
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sp)


def build(split, seeds):
    rng = np.random.default_rng(hash(split) & 0xffff)
    Xs, ys, sp = [], [], []
    for path in sorted(glob.glob(f"{RAW}/*.npz")):
        seed = int(path.split("_s")[-1].split(".")[0])
        if seed not in seeds:
            continue
        r = sequences_from_run(path, rng)
        if r:
            Xs.append(r[0]); ys.append(r[1]); sp.append(r[2])
    X = np.concatenate(Xs); y = np.concatenate(ys); s = np.concatenate(sp)
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(f"{DATA}/{split}.npz", X=X, y=y, speed=s)
    print(split, X.shape, "class counts", np.bincount(y))


if __name__ == "__main__":
    seeds = sorted({int(p.split("_s")[-1].split(".")[0]) for p in glob.glob(f"{RAW}/*.npz")})
    assert len(seeds) >= 2, "need >= 2 seeds; run dataset_gen.py with more seeds"
    build("train", set(seeds[:-1]))
    build("val", {seeds[-1]})
    print("-> wrote meritnet/data; now run: cd ../meritnet && python3 train.py")
