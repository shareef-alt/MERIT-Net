"""Online detectors plugged into the simulator loop.

Every detector sees the observation tensor X (n, n, NF) of each window and returns
  flag[i, j]     local suspicion of observer i about j (for FPR / TPR accounting)
  pi[i, j]       forwarding probability i applies to traffic originated by j
  convict[i, j]  i has blacklisted j
"""
import math
import numpy as np
import torch
from scipy.optimize import brentq

from sim import (NF, F_MASK, F_RREQ, F_HELLO, F_DATA, F_SYN, F_COMP, F_LINKAGE,
                 CAUSE_IDX, SYMPTOM_IDX)
from models import W, prep, LAMBDA_IN, BASE_IN

torch.set_num_threads(1)
STREAMS = [F_RREQ, F_HELLO, F_DATA, F_SYN]
T_BLACKLIST = 30


def theta(kappa, eps):
    f = lambda th: kappa ** th - 1 - th * (kappa - 1) * (1 - eps)
    if eps >= 1 - math.log(kappa) / (kappa - 1):
        return None
    return brentq(f, 1e-6, 1.0 + 1e-9) if eps > 0 else 1.0


class Base:
    name = "base"

    def reset(self, n):
        self.n = n
        self.ring = np.zeros((W, n, n, NF), np.float32)
        self.t = 0
        self.bl_until = np.zeros((n, n))
        self.bl_len = np.full((n, n), float(T_BLACKLIST))

    def push(self, X):
        self.ring[self.t % W] = X
        self.t += 1

    def seq(self, ii, jj):
        order = [(self.t - W + k) % W for k in range(W)]
        return self.ring[order][:, ii, jj, :].transpose(1, 0, 2)

    def blacklist(self, conv_new):
        t = self.t
        ii, jj = np.nonzero(conv_new & (self.bl_until <= t))
        self.bl_until[ii, jj] = t + self.bl_len[ii, jj]
        self.bl_len[ii, jj] *= 2

    def convicted(self):
        return self.bl_until > self.t


class NoDefense(Base):
    name = "AODV (no defense)"

    def step(self, X):
        self.push(X)
        n = self.n
        z = np.zeros((n, n), bool)
        return z, np.ones((n, n)), z


class StaticThreshold(Base):
    """Fixed per-stream rate threshold (AODV RREQ_RATELIMIT-style), blacklist on breach."""
    name = "Static threshold"

    def __init__(self, thr):
        self.thr = np.asarray(thr, np.float32)

    def step(self, X):
        self.push(X)
        mask = X[..., F_MASK] > 0
        flag = mask & (X[..., STREAMS] > self.thr).any(-1)
        self.blacklist(flag)
        conv = self.convicted()
        return flag, np.where(conv, 0.0, 1.0), conv


class PlainCUSUM(Base):
    """Flow-based CUSUM with a fixed (mobility-blind) baseline per stream."""
    name = "Plain CUSUM"

    def __init__(self, lam, kappa=2.0, arl0=1e5):
        self.lam = np.asarray(lam, np.float32)
        self.kappa = kappa
        self.h = math.log(4 * arl0)

    def reset(self, n):
        super().reset(n)
        self.S = np.zeros((n, n, 4), np.float32)

    def step(self, X):
        self.push(X)
        mask = X[..., F_MASK] > 0
        N = X[..., STREAMS]
        inc = N * math.log(self.kappa) - (self.kappa - 1) * self.lam
        self.S = np.where(mask[..., None], np.maximum(0, self.S + inc), 0)
        flag = mask & (self.S >= self.h).any(-1)
        self.blacklist(flag)
        conv = self.convicted()
        return flag, np.where(conv, 0.0, 1.0), conv


class SeqClassifier(Base):
    """Per-window sequence classifier -> blacklist on positive (kNN / CNN-LSTM / CNN-BiLSTM-RF)."""

    def __init__(self, name, predict_fn):
        self.name, self.predict_fn = name, predict_fn

    def step(self, X):
        self.push(X)
        n = self.n
        mask = X[..., F_MASK] > 0
        flag = np.zeros((n, n), bool)
        if self.t >= W:
            ii, jj = np.nonzero(mask)
            if len(ii):
                p = self.predict_fn(self.seq(ii, jj))
                flag[ii, jj] = p
        self.blacklist(flag)
        conv = self.convicted()
        return flag, np.where(conv, 0.0, 1.0), conv


class MERITNet(Base):
    """MERIT-Net: lambda-Net + per-stream residual CUSUM + V-Net + one-bit fusion + graduated response."""

    def __init__(self, lam_net, v_net, eps_rreq, eps_data, tau_v, kappa=2.0, arl0=1e5,
                 m=2, use_v=True, use_fusion=True, lam_fn=None, name="MERIT-Net", h_override=None,
                 disp=None):
        self.name = name
        self.lam_net, self.v_net = lam_net, v_net
        self.lam_fn = lam_fn
        self.kappa, self.m, self.tau_v = kappa, m, tau_v
        self.use_v, self.use_fusion = use_v, use_fusion
        # dispersion per stream [RREQ, HELLO, DATA, SYN]; inf = Poisson, finite = NB
        self.disp = np.array(disp if disp is not None else [np.inf] * 4, np.float32)
        th = [theta(kappa, eps_rreq), 1.0, theta(kappa, eps_data), 1.0]
        self.h = np.array([math.log(4 * arl0) / t for t in th], np.float32)
        if h_override is not None:          # empirical thresholds (max of theory and benign quantile)
            self.h = np.maximum(self.h, np.asarray(h_override, np.float32))
        self.h_low = 0.5 * self.h

    def _increment(self, N, lam):
        """Per-window CUSUM log-LR increment per stream.
        Poisson for streams with disp=inf; negative-binomial (dispersion r) for
        the rest, which absorbs overdispersion (bursty TCP) instead of inflating
        the threshold. NB -> Poisson as r -> inf, so this generalises the design.
        """
        k = self.kappa
        mu = np.maximum(lam, 1e-3)
        inc = N * math.log(k) - (k - 1) * mu                 # Poisson default
        r = self.disp                                        # (4,), inf where Poisson
        fin = np.isfinite(r)
        if fin.any():
            rr = np.where(fin, r, 1.0)                        # avoid inf in the math
            a = np.log(k * (rr + mu) / (rr + k * mu))
            b = rr * np.log((rr + mu) / (rr + k * mu))
            nb = N * a + b
            inc = np.where(fin[None, :], nb, inc)
        return inc

    def reset(self, n):
        super().reset(n)
        self.S = np.zeros((n, n, 4), np.float32)
        self.flag_hist = np.zeros((3, n, n), bool)
        self.ta = np.zeros((n, n), np.float32)      # Beta trust: benign evidence
        self.tb = np.zeros((n, n), np.float32)      # malicious evidence

    def expected(self, ii, jj, X):
        lam = np.zeros((len(ii), 4), np.float32)
        if len(ii) == 0:
            return lam
        s = prep(self.seq(ii, jj))[..., LAMBDA_IN]
        if self.lam_fn is not None:
            out = self.lam_fn(s)
        else:
            with torch.no_grad():
                out = self.lam_net(torch.from_numpy(s)).numpy()
        lam[:, 0] = out[:, 0]
        lam[:, 2] = out[:, 1]
        lam[:, 1] = 1.0                                        # HELLO: protocol constant
        lam[:, 3] = X[ii, jj, F_COMP] + 0.5                    # SYN: completed handshakes + slack
        return lam

    def step(self, X):
        self.push(X)
        n, t = self.n, self.t
        mask = X[..., F_MASK] > 0
        ii, jj = np.nonzero(mask)
        flag = np.zeros((n, n), bool)
        z = np.zeros((n, n), np.float32)
        p_att = np.zeros((n, n), np.float32)
        if t >= W and len(ii):
            lam = self.expected(ii, jj, X)
            N = X[ii, jj][:, [F_RREQ, F_HELLO, F_DATA, F_SYN]]
            inc = self._increment(N, lam)
            S_new = np.maximum(0, self.S[ii, jj] + inc)
            self.S[:] = 0
            self.S[ii, jj] = S_new
            zz = (S_new / self.h).max(1)
            z[ii, jj] = zz
            alarm = zz >= 1.0
            flag[ii[alarm], jj[alarm]] = True
            pre = (zz >= 0.5) & ~alarm
            if self.use_v and pre.any():
                pi_, pj_ = ii[pre], jj[pre]
                vin = self.v_input(pi_, pj_, lam[pre], S_new[pre])
                with torch.no_grad():
                    pr = torch.softmax(self.v_net(torch.from_numpy(vin)), -1).numpy()
                pa = 1.0 - pr[:, 0]
                p_att[pi_, pj_] = pa
                hit = pa >= self.tau_v
                flag[pi_[hit], pj_[hit]] = True
            p_att[ii[alarm], jj[alarm]] = 1.0
        # Beta trust with forgetting
        self.ta = 0.98 * self.ta + np.where(mask, 1.0 - p_att, 0)
        self.tb = 0.98 * self.tb + np.where(mask, p_att, 0)
        trust = (self.ta + 1) / (self.ta + self.tb + 2)
        # fusion
        self.flag_hist = np.roll(self.flag_hist, 1, axis=0)
        self.flag_hist[0] = flag
        recent = self.flag_hist.any(0)                     # flags of the last 3 windows
        if self.use_fusion:
            nb_ok = mask & (X[..., F_LINKAGE] >= 3 / 30.0) & (trust >= 0.5)   # eligible neighbours l of i
            # votes[i, j] = sum_l nb_ok[i, l] * recent[l, j] * mask[l, j]
            obs_l = (recent & mask).astype(np.float32)
            votes = nb_ok.astype(np.float32) @ obs_l + recent
            eligible = nb_ok.astype(np.float32) @ mask.astype(np.float32) + 1
            convict_new = mask & (((eligible >= self.m) & (votes >= self.m) & recent) |
                                  ((eligible < self.m) & (z >= 1.0)))
        else:
            convict_new = mask & flag                      # no fusion: any local flag convicts
        self.blacklist(convict_new)
        conv = self.convicted()
        # re-admission: when a blacklist just expired, restart at h_low
        expired = (self.bl_until == t) & mask
        if expired.any():
            self.S[expired] = self.h_low
        # graduated forwarding
        pi = np.ones((n, n))
        band = (z >= 0.5) & (z < 1.0)
        pi[band] = np.exp(-(z[band] - 0.5) / 0.5 * math.log(10.0))
        pi[z >= 1.0] = 0.0
        pi[conv] = 0.0
        return flag, pi, conv

    def v_input(self, ii, jj, lam_now, S_now):
        s = prep(self.seq(ii, jj))
        sym = s[..., SYMPTOM_IDX]                          # 7
        cnt = s[..., [F_RREQ, F_HELLO, F_DATA, F_SYN]]      # 4 (log)
        res = np.zeros_like(cnt)
        res[:, -1, :] = np.log1p(np.maximum(0, np.expm1(cnt[:, -1, :]) - lam_now))
        Sn = np.zeros_like(cnt)
        Sn[:, -1, :] = S_now / self.h
        mask = s[..., [F_MASK]]
        return np.concatenate([sym, cnt, res, Sn, mask], -1).astype(np.float32)


V_IN = 7 + 4 + 4 + 4 + 1
