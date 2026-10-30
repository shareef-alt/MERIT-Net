"""Closed-loop bridge: NS-3 and a Python detector run in lockstep.

Each window the NS-3 scenario (built with --bridge=1) sends its (N, N, 23)
feature tensor on stdout and blocks reading the detector's decisions on stdin.
This driver runs any detector with the Base .step(X) -> (flag, pi, convict)
interface from meritnet/detectors.py and returns, per observer i and neighbour
j, the drop probability to apply in the radio: 1 - pi[i][j], forced to 1 where
convict[i][j] is set.

With the benign scenario and a detector that convicts no one, PDR matches the
open-loop benign run; convict a node and its traffic is dropped — which is how
we validate the loop before any attack code exists.
"""
import os, sys, struct, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "meritnet"))
from sim import NF, F_MASK, F_ACTFLOW, F_NEWFLOW

BIN = os.path.join(os.path.dirname(__file__), "meritnet-scenario")


def run_closed_loop(detector, n=50, speed=5, sim=200, seed=1, n_cbr=20, attackers=""):
    """Run NS-3 with `detector` in the loop. detector.reset(n) then .step(X) each window.
    Returns the NS-3 stderr summary line (carries PDR)."""
    cmd = [BIN, f"--nNodes={n}", f"--speed={speed}", f"--sim={sim}", f"--seed={seed}",
           f"--nCbr={n_cbr}", "--bridge=1"]
    if attackers:
        cmd.append(f"--attackers={attackers}")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=0)
    detector.reset(n)
    nbytes = n * n * NF * 4
    prev_af = np.zeros((n, n), np.float32)
    try:
        while True:
            hdr = p.stdout.read(4)
            if len(hdr) < 4:
                break
            buf = b""
            while len(buf) < nbytes:
                chunk = p.stdout.read(nbytes - len(buf))
                if not chunk:
                    break
                buf += chunk
            if len(buf) < nbytes:
                break
            X = np.frombuffer(buf, np.float32).reshape(n, n, NF).copy()
            m = X[..., F_MASK] > 0
            X[..., F_NEWFLOW] = np.maximum(0.0, X[..., F_ACTFLOW] - prev_af) * m
            prev_af = X[..., F_ACTFLOW].copy()
            flag, pi, convict = detector.step(X)
            cv_i, cv_j = np.nonzero(convict & m)
            thr = (~convict) & (pi < 1.0) & m
            th_i, th_j = np.nonzero(thr)
            p.stdin.write(struct.pack("<I", len(cv_i) + len(th_i)))
            for i, j in zip(cv_i, cv_j):
                p.stdin.write(struct.pack("<IIff", int(i), int(j), 1.0, 0.0))
            for i, j in zip(th_i, th_j):
                p.stdin.write(struct.pack("<IIff", int(i), int(j), float(1.0 - pi[i, j]), 1.0))
            p.stdin.flush()
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
    err = p.stderr.read().decode()
    p.wait()
    return err.strip()


# ----- validation detectors (no ML, just to prove the loop + enforcement) -----
class _ConvictNone:
    def reset(self, n): self.n = n
    def step(self, X):
        z = np.zeros((self.n, self.n), bool)
        return z, np.ones((self.n, self.n)), z


class _ConvictOne:
    """Everyone blacklists node `target` after t0 — to show its traffic is dropped."""
    def __init__(self, target, t0=30): self.target, self.t0, self.t = target, t0, 0
    def reset(self, n): self.n, self.t = n, 0
    def step(self, X):
        self.t += 1
        conv = np.zeros((self.n, self.n), bool)
        if self.t >= self.t0:
            conv[:, self.target] = True
        return conv.copy(), np.where(conv, 0.0, 1.0), conv


if __name__ == "__main__":
    print("baseline (convict none):")
    print(" ", run_closed_loop(_ConvictNone(), sim=80, seed=1))
    print("convict node 5 from t=30 (its relayed traffic should drop):")
    print(" ", run_closed_loop(_ConvictOne(5), sim=80, seed=1))
