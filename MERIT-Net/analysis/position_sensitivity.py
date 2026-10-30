"""Sensitivity of lambda-Net (and hence Theorem 1) to the five position-derived cause features."""
import os, sys, json, math, numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__)); _M = os.path.join(_HERE, "..", "meritnet")
os.chdir(_M); sys.path.insert(0, _M)
torch.set_num_threads(2)
import train as T
import models as M
from sim import F_RREQ, F_DATA, F_DEGJ, F_CHURN, F_LINKAGE, F_DDIST, F_DEGI
from detectors import theta

POS = [F_DEGJ, F_CHURN, F_LINKAGE, F_DDIST, F_DEGI]
FULL_IN = list(M.LAMBDA_IN)
NOPOS_IN = [f for f in FULL_IN if f not in POS]
Xtr, ytr, _ = T.load("train"); Xva, yva, sva = T.load("val")
ben_va = yva == 0
Nva = Xva[ben_va][:, -1][:, [F_RREQ, F_DATA]]

def dev(lam, N):
    lam = np.maximum(lam, 1e-6)
    return float(np.mean(2 * (np.where(N > 0, N * np.log(np.maximum(N, 1e-9) / lam), 0) - (N - lam))))

def run(name, feats, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    M.LAMBDA_IN[:] = feats; T.LAMBDA_IN = M.LAMBDA_IN        # train/predict read this list
    orig = M.LambdaNet.__init__.__defaults__
    net = T.train_lambda.__wrapped__(Xtr, ytr, Xva, yva) if hasattr(T.train_lambda, "__wrapped__") else None
    return net

def train_and_eval(name, feats, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    M.LAMBDA_IN[:] = feats
    # LambdaNet's default nin was bound at import; build with the right width
    ben = ytr == 0
    xin = torch.from_numpy(M.prep(Xtr[ben])[..., feats]); tgt = torch.from_numpy(Xtr[ben][:, -1][:, [F_RREQ, F_DATA]])
    net = M.LambdaNet(nin=len(feats))
    opt = torch.optim.Adam(net.parameters(), 2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    F_ = torch.nn.functional
    for ep in range(10):
        for b in T.batches(len(xin), 1024):
            lam = net(xin[b]); y = tgt[b]
            loss = (lam - y * torch.log(lam)).mean() + 0.05 * (F_.relu(y - lam) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        lam_va = np.concatenate([net(torch.from_numpy(M.prep(Xva[ben_va][k:k+8192])[..., feats])).numpy()
                                 for k in range(0, int(ben_va.sum()), 8192)])
    return summarise(name, lam_va)

def summarise(name, lam_va):
    out = {"model": name}
    for s, lab in ((0, "rreq"), (1, "data")):
        e_raw, c = T.calib_eps(lam_va[:, s], Nva[:, s])
        e_cal, _ = T.calib_eps(lam_va[:, s] * c, Nva[:, s])
        eps = min(e_cal + 0.05, 0.25)
        th = theta(2.0, eps)
        out[lab] = dict(eps_raw=round(e_raw, 3), multiplier=round(c, 3), eps_cal=round(e_cal, 3), eps_used=round(eps, 3),
                        theta=round(th, 3), h=round(math.log(4e5) / th, 2), deviance=round(dev(lam_va[:, s] * c, Nva[:, s]), 4),
                        deviance_raw=round(dev(lam_va[:, s], Nva[:, s]), 4))
    return out

res = []
# deployed model on this validation split (reference only)
dep = M.LambdaNet(); dep.load_state_dict(torch.load("ckpt/lambda_net.pt")); dep.eval()
with torch.no_grad():
    lv = np.concatenate([dep(torch.from_numpy(M.prep(Xva[ben_va][k:k+8192])[..., FULL_IN])).numpy() for k in range(0, int(ben_va.sum()), 8192)])
res.append(summarise("deployed lambda-Net (reference)", lv))
for seed in (1, 2, 3):
    res.append(train_and_eval(f"full features, seed {seed}", FULL_IN, seed))
    res.append(train_and_eval(f"no position features, seed {seed}", NOPOS_IN, seed))
# constant baseline (no causes at all) for scale
cst = np.tile(Xtr[ytr == 0][:, -1][:, [F_RREQ, F_DATA]].mean(0), (int(ben_va.sum()), 1)).astype(np.float32)
res.append(summarise("constant mean (no cause features)", cst))
for r in res:
    print(f"{r['model']:36s} RREQ: eps_raw {r['rreq']['eps_raw']:.3f} x{r['rreq']['multiplier']:.2f} -> eps {r['rreq']['eps_used']:.3f} theta {r['rreq']['theta']:.3f} h {r['rreq']['h']:5.2f} dev {r['rreq']['deviance']:.4f} | "
          f"DATA: eps_raw {r['data']['eps_raw']:.3f} x{r['data']['multiplier']:.2f} dev {r['data']['deviance']:.4f}")
json.dump(res, open(os.path.join(_HERE, "position_sensitivity.json"), "w"), indent=1, default=float)
print("saved")
