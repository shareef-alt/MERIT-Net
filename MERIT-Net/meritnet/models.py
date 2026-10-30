"""Neural models: MERIT-Net (lambda-Net, V-Net) and reimplemented baselines."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import CAUSE_IDX, SYMPTOM_IDX, COUNT_IDX, F_MASK, NF

W = 16
LOG_FEATS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 19]   # count-like -> log1p


def prep(seq):
    """seq (..., W, NF) raw features -> normalised copy (log1p on count-like features)."""
    x = seq.astype(np.float32, copy=True) if isinstance(seq, np.ndarray) else seq.clone().float()
    if isinstance(x, np.ndarray):
        x[..., LOG_FEATS] = np.log1p(np.maximum(x[..., LOG_FEATS], 0))
    else:
        x[..., LOG_FEATS] = torch.log1p(torch.clamp(x[..., LOG_FEATS], min=0))
    return x


LAMBDA_IN = CAUSE_IDX + [F_MASK]                       # 12 inputs, never the counts
BASE_IN = list(range(22))                              # everything except mask


class CausalDS(nn.Module):
    """Causal depthwise-separable 1-D convolution."""
    def __init__(self, cin, cout, k=3, dil=1):
        super().__init__()
        self.pad = (k - 1) * dil
        self.dw = nn.Conv1d(cin, cin, k, dilation=dil, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)

    def forward(self, x):                               # x (B, C, T)
        return self.pw(self.dw(F.pad(x, (self.pad, 0))))


class LambdaNet(nn.Module):
    """Expected legitimate load for the RREQ and DATA streams (two softplus heads)."""
    def __init__(self, nin=len(LAMBDA_IN), ch=16):
        super().__init__()
        self.b1 = CausalDS(nin, ch, 3, 1)
        self.b2 = CausalDS(ch, ch, 3, 2)
        self.head = nn.Linear(ch, 2)

    def forward(self, x):                               # x (B, T, nin)
        h = x.transpose(1, 2)
        h1 = F.relu(self.b1(h))
        h2 = F.relu(self.b2(h1)) + h1
        out = F.softplus(self.head(h2[:, :, -1])) + 1e-3
        return out                                      # (B, 2): [lambda_rreq, lambda_data]


class VNet(nn.Module):
    """5-way verifier: benign, RREQ, Hello, data, SYN flooding."""
    def __init__(self, nin, ch=24, ncls=5):
        super().__init__()
        self.b1 = CausalDS(nin, ch, 3, 1)
        self.b2 = CausalDS(ch, ch, 3, 2)
        self.b3 = CausalDS(ch, ch, 3, 4)
        self.head = nn.Linear(ch, ncls)

    def forward(self, x):
        h = x.transpose(1, 2)
        h1 = F.relu(self.b1(h))
        h2 = F.relu(self.b2(h1)) + h1
        h3 = F.relu(self.b3(h2)) + h2
        return self.head(h3.mean(-1))


class CNNLSTM(nn.Module):
    """Reference-paper style CNN-LSTM (Conv1D 32 filters k=3, MaxPool 2, LSTM 32, dense)."""
    def __init__(self, nin=len(BASE_IN)):
        super().__init__()
        self.conv = nn.Conv1d(nin, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        h = F.relu(self.conv(x.transpose(1, 2)))
        h = F.max_pool1d(h, 2).transpose(1, 2)
        _, (hn, _) = self.lstm(h)
        return self.out(hn[-1]).squeeze(-1)


class CNNGRU(nn.Module):
    """Reference-paper CNN-GRU branch (Conv1D 64 filters k=5, MaxPool 4, dropout, GRU 32)."""
    def __init__(self, nin=len(BASE_IN)):
        super().__init__()
        self.conv = nn.Conv1d(nin, 64, 5, padding=2)
        self.drop = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        h = F.relu(self.conv(x.transpose(1, 2)))
        h = self.drop(F.max_pool1d(h, 4)).transpose(1, 2)
        _, hn = self.gru(h)
        return self.out(hn[-1]).squeeze(-1)


class RefHybrid(nn.Module):
    """Reference paper (Sci. Rep. 2025): CNN-LSTM and CNN-GRU branches, averaged."""
    def __init__(self, nin=len(BASE_IN)):
        super().__init__()
        self.a, self.b = CNNLSTM(nin), CNNGRU(nin)

    def forward(self, x):
        return 0.5 * (self.a(x) + self.b(x))


class CNNBiLSTM(nn.Module):
    """CNN-BiLSTM feature extractor (an RF is trained on its embedding)."""
    def __init__(self, nin=len(BASE_IN)):
        super().__init__()
        self.conv = nn.Conv1d(nin, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.out = nn.Linear(64, 1)

    def embed(self, x):
        h = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        _, (hn, _) = self.lstm(h)
        return torch.cat([hn[0], hn[1]], -1)

    def forward(self, x):
        return self.out(self.embed(x)).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def conv_macs(cin, cout, k, T, groups=1):
    return cin // groups * cout * k * T


def macs_lambda(m, T=1):
    """MACs per new window in streaming mode (one time step)."""
    nin = m.b1.dw.in_channels; ch = m.b1.pw.out_channels
    return (nin * 3 + nin * ch) * T + (ch * 3 + ch * ch) * T + ch * 2


def macs_vnet(m, T=W):
    nin = m.b1.dw.in_channels; ch = m.b1.pw.out_channels; nc = m.head.out_features
    blk = lambda c_in: c_in * 3 + c_in * ch
    return (blk(nin) + 2 * blk(ch)) * T + ch * nc


def macs_cnnlstm(nin=22, T=W):
    conv = nin * 32 * 3 * T
    lstm = 4 * (32 * 32 + 32 * 32) * (T // 2)
    return conv + lstm + 32


def macs_cnngru(nin=22, T=W):
    conv = nin * 64 * 5 * T
    gru = 3 * (64 * 32 + 32 * 32) * (T // 4)
    return conv + gru + 32


def macs_cnnbilstm(nin=22, T=W):
    return nin * 32 * 3 * T + 2 * 4 * (32 * 32 + 32 * 32) * T + 64
