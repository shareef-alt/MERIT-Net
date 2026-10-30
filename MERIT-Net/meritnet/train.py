"""Train MERIT-Net components and the reimplemented baselines; calibrate thresholds."""
import json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import PoissonRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

from sim import F_RREQ, F_HELLO, F_DATA, F_SYN, F_COMP, F_MASK, SYMPTOM_IDX
from models import (W, prep, LAMBDA_IN, BASE_IN, LambdaNet, VNet, RefHybrid, CNNBiLSTM,
                    n_params)
from detectors import V_IN, theta

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)
os.makedirs("ckpt", exist_ok=True)
STREAMS = [F_RREQ, F_HELLO, F_DATA, F_SYN]
KAPPA = 2.0


def load(split):
    d = np.load(f"data/{split}.npz")
    return d["X"].astype(np.float32), d["y"], d["speed"]


def batches(n, bs, shuffle=True):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for k in range(0, n, bs):
        yield idx[k:k + bs]


# ---------------------------------------------------------------- lambda-Net
def train_lambda(Xtr, ytr, Xva, yva, epochs=10):
    ben = ytr == 0
    xin = torch.from_numpy(prep(Xtr[ben])[..., LAMBDA_IN])
    tgt = torch.from_numpy(Xtr[ben][:, -1][:, [F_RREQ, F_DATA]])
    net = LambdaNet()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        tot = 0
        for b in batches(len(xin), 1024):
            lam = net(xin[b])
            y = tgt[b]
            loss = (lam - y * torch.log(lam)).mean() + 0.05 * (F.relu(y - lam) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        print(f"  lambda-Net epoch {ep} loss {tot/len(xin):.4f}", flush=True)
    return net


def lam_predict(net, X):
    out = []
    with torch.no_grad():
        for b in batches(len(X), 8192, False):
            out.append(net(torch.from_numpy(prep(X[b])[..., LAMBDA_IN])).numpy())
    return np.concatenate(out)


def calib_eps(lam, N, nbins=20):
    """eps = max over lambda-quantile bins of 1 - mean(lam)/mean(N) (under-estimation)."""
    q = np.quantile(lam, np.linspace(0, 1, nbins + 1))
    worst, c = 0.0, 1.0
    for a, b in zip(q[:-1], q[1:]):
        sel = (lam >= a) & (lam <= b)
        if sel.sum() < 500 or N[sel].mean() < 1e-3:
            continue
        r = lam[sel].mean() / N[sel].mean()
        worst = max(worst, 1 - r)
        c = max(c, 1 / r)
    return worst, c


# ---------------------------------------------------------------- V-Net input (offline)
def v_offline(X, lam_last, h):
    """Build V-Net input like the online detector: S and residual at the last step."""
    s = prep(X)
    sym = s[..., SYMPTOM_IDX]
    cnt = s[..., [F_RREQ, F_HELLO, F_DATA, F_SYN]]
    raw = X[..., [F_RREQ, F_HELLO, F_DATA, F_SYN]]
    lam = np.zeros(raw.shape, np.float32)
    lam[..., 0] = lam_last[:, None, 0]
    lam[..., 2] = lam_last[:, None, 1]
    lam[..., 1] = 1.0
    lam[..., 3] = X[..., F_COMP] + 0.5
    S = np.zeros((len(X), 4), np.float32)
    for k in range(W):
        m = X[:, k, F_MASK:F_MASK + 1] > 0
        S = np.where(m, np.maximum(0, S + raw[:, k] * math.log(KAPPA) - (KAPPA - 1) * lam[:, k]), 0)
    res = np.zeros_like(cnt)
    res[:, -1] = np.log1p(np.maximum(0, raw[:, -1] - lam[:, -1]))
    Sn = np.zeros_like(cnt)
    Sn[:, -1] = S / h
    return np.concatenate([sym, cnt, res, Sn, s[..., [F_MASK]]], -1).astype(np.float32), (S / h).max(1)


def train_vnet(Xv, yv, Xva, yva, epochs=8):
    net = VNet(V_IN)
    xin, yt = torch.from_numpy(Xv), torch.from_numpy(yv).long()
    cnt = np.bincount(yv, minlength=5).astype(np.float32)
    wcls = torch.tensor((cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        tot = 0
        for b in batches(len(xin), 1024):
            logit = net(xin[b])
            ce = F.cross_entropy(logit, yt[b], weight=wcls, reduction="none")
            pt = torch.exp(-F.cross_entropy(logit, yt[b], reduction="none"))
            loss = ((1 - pt) ** 2 * ce).mean()                      # focal loss, gamma = 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        with torch.no_grad():
            pr = torch.softmax(net(torch.from_numpy(Xva)), -1).numpy()
        acc = (pr.argmax(1) == yva).mean()
        print(f"  V-Net epoch {ep} loss {tot/len(xin):.4f} val acc {acc:.4f}", flush=True)
    return net


def train_binary(net, Xtr, ytr, Xva, yva, epochs=6, lr=4e-4, tag=""):
    xin = torch.from_numpy(prep(Xtr)[..., BASE_IN])
    yt = torch.from_numpy((ytr > 0).astype(np.float32))
    pos_w = torch.tensor(float((ytr == 0).sum() / max((ytr > 0).sum(), 1)))
    opt = torch.optim.Adam(net.parameters(), lr)
    xva = torch.from_numpy(prep(Xva)[..., BASE_IN])
    for ep in range(epochs):
        net.train()
        tot = 0
        for b in batches(len(xin), 512):
            loss = F.binary_cross_entropy_with_logits(net(xin[b]), yt[b], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            pv = torch.cat([torch.sigmoid(net(xva[b])) for b in batches(len(xva), 8192, False)]).numpy()
        acc = ((pv > 0.5) == (yva > 0)).mean()
        print(f"  {tag} epoch {ep} loss {tot/len(xin):.4f} val acc {acc:.4f}", flush=True)
    return net


if __name__ == "__main__":
    t0 = time.time()
    Xtr, ytr, str_ = load("train")
    Xva, yva, sva = load("val")
    print("loaded", Xtr.shape, Xva.shape, f"{time.time()-t0:.0f}s", flush=True)
    cfg = {}

    # 1. lambda-Net
    lam_net = train_lambda(Xtr, ytr, Xva, yva)
    torch.save(lam_net.state_dict(), "ckpt/lambda_net.pt")
    ben_va = yva == 0
    lam_va = lam_predict(lam_net, Xva[ben_va])
    Nva = Xva[ben_va][:, -1][:, [F_RREQ, F_DATA]]
    e_r, c_r = calib_eps(lam_va[:, 0], Nva[:, 0])
    e_d, c_d = calib_eps(lam_va[:, 1], Nva[:, 1])
    print(f"raw eps: rreq {e_r:.3f} data {e_d:.3f}; calibration multipliers {c_r:.3f} {c_d:.3f}")
    cfg["lam_scale"] = [float(c_r), float(c_d)]
    e_r2, _ = calib_eps(lam_va[:, 0] * c_r, Nva[:, 0])
    e_d2, _ = calib_eps(lam_va[:, 1] * c_d, Nva[:, 1])
    cfg["eps_raw"] = [float(e_r), float(e_d)]
    cfg["eps"] = [float(min(e_r2 + 0.05, 0.25)), float(min(e_d2 + 0.05, 0.25))]
    print("eps after calibration (+0.05 margin):", cfg["eps"])

    # 2. linear Poisson GLM (ablation: MERIT-linear)
    ben = ytr == 0
    sub = np.random.choice(np.where(ben)[0], min(60000, int(ben.sum())), replace=False)
    Zl = prep(Xtr[sub][:, -1])[..., LAMBDA_IN]
    glm = [PoissonRegressor(alpha=1e-4, max_iter=300).fit(Zl, Xtr[sub][:, -1, s]) for s in (F_RREQ, F_DATA)]
    joblib.dump(glm, "ckpt/glm.joblib")
    lg = np.stack([g.predict(prep(Xva[ben_va][:, -1])[..., LAMBDA_IN]) for g in glm], 1)
    ge_r, gc_r = calib_eps(lg[:, 0], Nva[:, 0]); ge_d, gc_d = calib_eps(lg[:, 1], Nva[:, 1])
    cfg["glm_scale"] = [float(gc_r), float(gc_d)]
    cfg["glm_eps_raw"] = [float(ge_r), float(ge_d)]
    cfg["glm_eps_cal"] = [float(calib_eps(lg[:, 0] * gc_r, Nva[:, 0])[0]),
                          float(calib_eps(lg[:, 1] * gc_d, Nva[:, 1])[0])]
    print(f"GLM raw eps rreq {ge_r:.3f} data {ge_d:.3f}")
    # predictive quality comparison (Poisson deviance on val)
    def dev(lam, N):
        lam = np.maximum(lam, 1e-6)
        return float(np.mean(2 * (np.where(N > 0, N * np.log(np.maximum(N, 1e-9) / lam), 0) - (N - lam))))
    cfg["deviance"] = {"lambda_net": [dev(lam_va[:, 0], Nva[:, 0]), dev(lam_va[:, 1], Nva[:, 1])],
                       "glm": [dev(lg[:, 0], Nva[:, 0]), dev(lg[:, 1], Nva[:, 1])],
                       "constant": [dev(np.full(len(Nva), Nva[:, 0].mean()), Nva[:, 0]),
                                    dev(np.full(len(Nva), Nva[:, 1].mean()), Nva[:, 1])]}
    print("val deviance", cfg["deviance"])

    # 3. thresholds
    arl0 = 1e5
    h = np.array([math.log(4 * arl0) / theta(KAPPA, cfg["eps"][0]), math.log(4 * arl0),
                  math.log(4 * arl0) / theta(KAPPA, cfg["eps"][1]), math.log(4 * arl0)], np.float32)
    cfg["h"] = h.tolist()
    print("CUSUM thresholds", h)

    # 4. V-Net
    def lam_all(X):
        l = lam_predict(lam_net, X)
        return l * np.array(cfg["lam_scale"], np.float32)
    ben_idx = np.where(ytr == 0)[0]; att_idx = np.where(ytr > 0)[0]
    vsel = np.concatenate([np.random.choice(ben_idx, min(120000, len(ben_idx)), replace=False), att_idx])
    Xv, zv_tr = v_offline(Xtr[vsel], lam_all(Xtr[vsel]), h)
    Xvva, zv_va = v_offline(Xva, lam_all(Xva), h)
    v_net = train_vnet(Xv, ytr[vsel], Xvva, yva)
    torch.save(v_net.state_dict(), "ckpt/v_net.pt")
    with torch.no_grad():
        pva = 1 - torch.softmax(v_net(torch.from_numpy(Xvva)), -1).numpy()[:, 0]
    pre_ben = (yva == 0) & (zv_va >= 0.5)
    tau = float(np.quantile(pva[yva == 0], 0.995))
    cfg["tau_v"] = max(tau, 0.5)
    cfg["vnet_prealarm_benign_n"] = int(pre_ben.sum())
    print("tau_v", cfg["tau_v"], "benign pre-alarm samples", int(pre_ben.sum()))

    # 5. baselines -------------------------------------------------------
    last_ben = Xtr[ytr == 0][:, -1][:, [F_RREQ, F_HELLO, F_DATA, F_SYN]]
    cfg["static_thr"] = np.quantile(last_ben, 0.9999, axis=0).tolist()
    cfg["plain_lam"] = last_ben.mean(0).tolist()
    print("static thr", cfg["static_thr"], "plain lambda", cfg["plain_lam"])

    # FAPRP-style kNN on route-discovery (and other stream) frequency vectors
    def knn_feat(X):
        return np.log1p(X[..., [F_RREQ, F_HELLO, F_DATA, F_SYN]]).reshape(len(X), -1)
    ks = np.concatenate([np.random.choice(ben_idx, min(3000, len(ben_idx)), replace=False),
                         np.random.choice(att_idx, min(3000, len(att_idx)), replace=False)])
    knn = KNeighborsClassifier(5).fit(knn_feat(Xtr[ks]), (ytr[ks] > 0).astype(int))
    joblib.dump(knn, "ckpt/knn.joblib")
    acc = (knn.predict(knn_feat(Xva[::10])) == (yva[::10] > 0)).mean()
    print("kNN val acc", acc)

    # reference paper hybrid CNN-LSTM / CNN-GRU (Adam, lr 4e-4 as in the paper; BCE instead of MSE)
    bsel = np.concatenate([np.random.choice(ben_idx, min(150000, len(ben_idx)), replace=False), att_idx])
    ref = train_binary(RefHybrid(), Xtr[bsel], ytr[bsel], Xva[::4], yva[::4], tag="Ref CNN-LSTM/GRU")
    torch.save(ref.state_dict(), "ckpt/ref_hybrid.pt")

    # CNN-BiLSTM-RF
    cbl = train_binary(CNNBiLSTM(), Xtr[bsel], ytr[bsel], Xva[::4], yva[::4], lr=1e-3, tag="CNN-BiLSTM")
    torch.save(cbl.state_dict(), "ckpt/cnn_bilstm.pt")
    rs = np.concatenate([np.random.choice(ben_idx, min(20000, len(ben_idx)), replace=False),
                         np.random.choice(att_idx, min(20000, len(att_idx)), replace=False)])
    with torch.no_grad():
        emb = cbl.embed(torch.from_numpy(prep(Xtr[rs])[..., BASE_IN])).numpy()
    rf = RandomForestClassifier(100, max_depth=12, n_jobs=2, random_state=0).fit(emb, (ytr[rs] > 0).astype(int))
    joblib.dump(rf, "ckpt/rf.joblib")

    cfg["params"] = {"lambda_net": n_params(lam_net), "v_net": n_params(v_net),
                     "ref_hybrid": n_params(ref), "cnn_bilstm": n_params(cbl)}
    print("params", cfg["params"])
    json.dump(cfg, open("ckpt/config.json", "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")
