"""All result figures from the 360 closed-loop NS-3 runs (same data as Table 1).
Color = method family (validated all-pairs: blue MERIT-Net, orange learning baselines,
aqua statistical baselines); marker + line style = detector, so identity is never color-alone."""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # run from anywhere
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_numbers import N as _N

ST = json.load(open("stats.json"))   # produced by stats.py: means with 95% bootstrap CIs
S = [0, 5, 10, 15, 20]
ATT = ["rreq", "rreq_stealth", "hello", "data", "syn"]
ATT_LBL = {"rreq": "RREQ", "rreq_stealth": "RREQ (stealth)", "hello": "Hello", "data": "Data", "syn": "SYN"}

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
STYLE = {   # order = legend order (fixed across all figures)
    "MERIT-Net":         dict(color=BLUE,   ls="-",  marker="o", lw=2.0, label="MERIT-Net"),
    "Ref. CNN-LSTM/GRU": dict(color=ORANGE, ls="-",  marker="s", lw=1.2, label="CNN-LSTM/GRU [1]"),
    "CNN-BiLSTM-RF":     dict(color=ORANGE, ls="--", marker="^", lw=1.2, label="CNN-BiLSTM-RF [6]"),
    "FAPRP-style kNN":   dict(color=ORANGE, ls=":",  marker="v", lw=1.2, label="FAPRP kNN [2]"),
    "Flow-based CUSUM":  dict(color=AQUA,   ls="--", marker="D", lw=1.2, label="Flow CUSUM [3]"),
    "Static threshold":  dict(color=AQUA,   ls=":",  marker="X", lw=1.2, label="Static threshold"),
}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})


def sel(d, **k):
    return [r for r in R if r["detector"] == d and all(r[a] == v for a, v in k.items())]


def style_ax(ax):
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2.5)


def line_plot(ax, key, logy=False, floor=None):
    """Mean per speed with a shaded 95% bootstrap CI band."""
    for d, st in STYLE.items():
        st = dict(st); lab = st.pop("label")
        trip = [ST["speed"][d][str(s)][key] for s in S]
        ys = np.array([t[0] for t in trip]); lo = np.array([t[1] for t in trip]); hi = np.array([t[2] for t in trip])
        if floor is not None:
            ys, lo, hi = np.maximum(ys, floor), np.maximum(lo, floor), np.maximum(hi, floor)
        ax.fill_between(S, lo, hi, color=st["color"], alpha=0.13 if d != "MERIT-Net" else 0.22, lw=0, zorder=1)
        ax.plot(S, ys, label=lab, ms=4, mec="white", mew=0.5, zorder=3 if d == "MERIT-Net" else 2, **st)
    ax.set_xticks(S); ax.set_xlabel("maximum node speed (m/s)", fontsize=7)
    if logy: ax.set_yscale("log")
    style_ax(ax)


def legend(ax, **kw):
    ax.legend(fontsize=5.4, frameon=False, ncol=3, columnspacing=0.9, handlelength=2.2,
              labelcolor=INK, **kw)


def save(fig, name, rect=None):
    fig.tight_layout(pad=0.4, rect=rect)
    fig.savefig("../figures/" + name + ".png", dpi=300); fig.savefig("../figures/" + name + ".pdf")
    plt.close(fig); print("wrote", name)


# Fig. 2 - benign false-positive rate vs speed
fig, ax = plt.subplots(figsize=(3.5, 2.55))
line_plot(ax, "fpr"); ax.set_ylabel("benign false-positive rate", fontsize=7)
ax.set_ylim(-0.001, 0.058); legend(ax, loc="upper center")
save(fig, "fig_fpr")

# Fig. 3 - honest nodes isolated per run vs speed
fig, ax = plt.subplots(figsize=(3.5, 2.55))
line_plot(ax, "iso")
ax.set_ylabel("honest nodes isolated per run", fontsize=7); ax.set_ylim(0, 68)
legend(ax, loc="upper center")
ax.annotate("MERIT-Net: " + _N["iso_m_rng_dash"], xy=(20, ST["speed"]["MERIT-Net"]["20"]["iso"][0]), xytext=(13.2, 9), fontsize=6, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
save(fig, "fig_iso")

# Fig. 4 - packet delivery ratio vs speed: (a) benign, (b) under attack
fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
line_plot(axs[0], "pdr_ben")
line_plot(axs[1], "pdr_atk")
axs[0].set_xlabel("")
axs[0].set_ylabel("PDR, no attack", fontsize=7); axs[1].set_ylabel("PDR under attack", fontsize=7)
for a, tag in zip(axs, "ab"):
    a.set_ylim(0.3, 1.02); a.text(0.01, 0.03, f"({tag})", transform=a.transAxes, fontsize=7, color=INK)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=5.4, frameon=False, ncol=3, columnspacing=0.9, handlelength=2.2,
           labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, 0.995))
save(fig, "fig_pdr", rect=(0, 0, 1, 0.93))

# Fig. 5 - mean detection delay per attack type (dot plot); static threshold excluded (misses most attacks)
fig, ax = plt.subplots(figsize=(3.5, 2.5))
dets = [d for d in STYLE if d != "Static threshold"]
off = np.linspace(-0.26, 0.26, len(dets))
for k, d in enumerate(dets):
    st = STYLE[d]
    trip = [ST["attack"][d][a]["delay"] for a in ATT]
    xs = np.array([t[0] for t in trip]); lo = np.array([t[1] for t in trip]); hi = np.array([t[2] for t in trip])
    ys = np.arange(len(ATT)) + off[k]
    ax.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt="none", ecolor=st["color"], elinewidth=0.7, capsize=1.2, zorder=2)
    ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=16 if d != "MERIT-Net" else 22,
               edgecolors="white", linewidths=0.5, label=st["label"], zorder=3)
ax.set_yticks(range(len(ATT))); ax.set_yticklabels([ATT_LBL[a] for a in ATT])
ax.invert_yaxis(); ax.set_xlabel("mean detection delay after attack onset (s)", fontsize=7)
ax.set_xlim(0, max(26, ax.get_xlim()[1])); style_ax(ax); ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
for y in np.arange(len(ATT) - 1) + 0.5: ax.axhline(y, color=GRID, lw=0.6)
ax.legend(fontsize=5.4, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22),
          columnspacing=0.9, handletextpad=0.3, labelcolor=INK)
save(fig, "fig_delay")

# Fig. 6 - false convictions per run vs speed (log scale)
fig, ax = plt.subplots(figsize=(3.5, 2.55))
line_plot(ax, "fconv", logy=True, floor=2.2)
ax.set_ylabel("false convictions per run (log scale)", fontsize=7); ax.set_ylim(2, 2e4)
ax.grid(axis="y", which="minor", color=GRID, lw=0.3)
legend(ax, loc="upper center")
save(fig, "fig_fconv")
