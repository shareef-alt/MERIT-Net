"""Load the NS-3 scenario's per-window feature dump and feed the Python detectors.

The NS-3 program meritnet-scenario.cc writes a binary file of
    (T, N, N, 23) float32
one (observer i, neighbour j) feature vector per window, with the SAME 23
features and indices as meritnet/sim.py (F_RREQ .. F_MASK).

This loader turns a run into the same X tensors the detectors already consume,
so training (gen_data/train) and evaluation (evaluate) work unchanged on NS-3
traffic.  F_NEWFLOW, which the C++ side leaves at 0, is reconstructed here from
the rise in a neighbour's active-flow count.
"""
import subprocess, os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "meritnet"))
from sim import NF, F_MASK, F_ACTFLOW, F_NEWFLOW

BIN = os.path.join(os.path.dirname(__file__), "meritnet-scenario")


def run_scenario(n=50, speed=5, sim=200, seed=1, n_cbr=20, attackers="", out=None):
    """Run the NS-3 scenario once; return X of shape (T, n, n, 23)."""
    out = out or f"/tmp/ns3_{seed}_{speed}.bin"
    cmd = [BIN, f"--nNodes={n}", f"--speed={speed}", f"--sim={sim}", f"--seed={seed}",
           f"--nCbr={n_cbr}", f"--out={out}"]
    if attackers:
        cmd.append(f"--attackers={attackers}")
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    X = np.fromfile(out, dtype=np.float32).reshape(-1, n, n, NF)
    os.remove(out)
    # reconstruct F_NEWFLOW = positive rise in active-flow count vs previous window
    af = X[..., F_ACTFLOW]
    nf = np.maximum(0.0, af - np.concatenate([af[:1], af[:-1]], 0))
    X[..., F_NEWFLOW] = nf * (X[..., F_MASK] > 0)
    return X


if __name__ == "__main__":
    X = run_scenario(sim=60, seed=1)
    m = X[..., F_MASK] > 0
    print("NS-3 run OK:", X.shape, "mean neighbours/node/window",
          round(m.sum() / (X.shape[0] * X.shape[1]), 2))
