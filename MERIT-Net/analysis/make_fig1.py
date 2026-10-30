"""Fig. 1 - MERIT-Net per-neighbor architecture (matches Algorithms 1-3). Full page width."""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # run from anywhere
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#2a78d6"; BLUE_T = "#e3eefb"; INK = "#1f1f1e"; MUTED = "#5f5e5a"
IN_FILL = "#f3f2ee"; IN_EDGE = "#8a8983"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

W, H = 7.0, 3.05
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

KIND = {
    "input":   dict(fc=IN_FILL, ec=IN_EDGE, tc=INK, lw=0.8),
    "learned": dict(fc=BLUE_T, ec=BLUE, tc=INK, lw=1.0),
    "core":    dict(fc=BLUE, ec=BLUE, tc="white", lw=1.0),
    "control": dict(fc="white", ec=INK, tc=INK, lw=0.9),
}


def box(x, y, w, h, title, body, kind):
    k = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=k["fc"], ec=k["ec"], lw=k["lw"], zorder=2))
    tl = title.count("\n") + 1
    bl = body.count("\n") + 1 if body else 0
    TH, BH, GAP = 0.125, 0.098, 0.07          # line heights (in) for title / body, gap between
    block = tl * TH + (GAP + bl * BH if body else 0)
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=6.8, fontweight="bold",
            color=k["tc"], linespacing=1.15, zorder=3)
    if body:
        ax.text(x + w / 2, top - tl * TH - GAP, body, ha="center", va="top", fontsize=5.7,
                color=k["tc"], linespacing=1.35, zorder=3)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)


def arrow(pts, label=None, lpos=None, ha="center", va="bottom", color=MUTED, lw=0.9):
    """Orthogonal polyline; arrowhead on the last segment."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color,
                                lw=lw, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=5.6, color=INK, ha=ha, va=va, zorder=4)


# lanes and columns (inches)
YT, YM, YB = 2.08, 1.24, 0.38          # bottoms of top / middle / bottom lanes
HT, HM, HB = 0.62, 0.66, 0.62
cA, wA = 0.05, 0.92
cB, wB = 1.17, 1.38
cC, wC = 2.80, 1.42
cD, wD = 4.38, 1.08
cE, wE = 5.72, 1.25

A = box(cA, YB, wA, YT + HT - YB, "Overheard\ntraffic of\nneighbor j",
        "window t (Δ = 1 s)\n\nAODV RREQ,\nRERR, hello,\ndata, TCP SYN", "input")
Bt = box(cB, YT, wB, HT, r"Cause features $x_j$", "route errors, missed hellos,\nchurn, link age, flows", "input")
Bm = box(cB, YM, wB, HM, r"Counts $N_{j,s}$", "s ∈ {RREQ, HELLO,\nDATA, SYN}", "input")
Bb = box(cB, YB, wB, HB, r"Symptom features $y_j$", "unknown-dest. share, entropy,\nduplicates, half-open conns", "input")
Ct = box(cC, YT, wC, HT, "λ-Net (learned)", "causal TCN, 0.6k parameters\ntrained on attack-free traces", "learned")
Cm = box(cC, YM, wC, HM, "Residual CUSUM × 4",
         r"$S_{j,s}\leftarrow\max(0,\,S_{j,s}+\mathrm{inc}_s)$" "\n" r"$z_j=\max_s\,S_{j,s}/h_s$", "core")
Dm = box(cD, YM, wD, HM, "Band test on $z_j$",
         "z ≥ 1: alarm (no veto)\n½ ≤ z < 1: ask V-Net\nz < ½: normal", "control")
Db = box(cD, YB, wD, HB, "V-Net (learned)", "5-way verifier, 2.1k\nflag if $p_j ≥ τ_v$", "learned")
Et = box(cE, YT, wE, HT, "One-bit fusion", "convict if ≥ m eligible\nobservers agree", "control")
Em = box(cE, YM, wE, HM, "Graduated mitigation", "rate-limit RREQs\n→ quarantine\n→ blacklist", "control")

# inputs -> features
for b in (Bt, Bm, Bb):
    arrow([(A["r"], b["cy"]), (b["l"], b["cy"])])
# causes -> lambda-Net -> CUSUM ; counts -> CUSUM
arrow([(Bt["r"], Bt["cy"]), (Ct["l"], Ct["cy"])])
arrow([(Bm["r"], Bm["cy"]), (Cm["l"], Cm["cy"])])
arrow([(Ct["cx"], Ct["b"]), (Ct["cx"], Cm["t"])], r"$\hat\lambda_{j,\mathrm{RREQ}},\ \hat\lambda_{j,\mathrm{DATA}}$",
      (Ct["cx"] + 0.05, (Ct["b"] + Cm["t"]) / 2), ha="left", va="center")
ax.text(Cm["cx"], Cm["b"] - 0.05, r"$\hat\lambda_{j,\mathrm{HELLO}},\ \hat\lambda_{j,\mathrm{SYN}}$ from protocol state",
        fontsize=5.3, color=MUTED, ha="center", va="top")
# CUSUM -> band test
arrow([(Cm["r"], Cm["cy"]), (Dm["l"], Cm["cy"])], "$z_j$", ((Cm["r"] + Dm["l"]) / 2, Cm["cy"] + 0.03))
# symptoms -> V-Net (bottom lane)
arrow([(Bb["r"], Bb["cy"]), (Db["l"], Bb["cy"])], "$y_j$, residuals, $S_{j,\\cdot}/h$",
      ((Bb["r"] + Db["l"]) / 2, Bb["cy"] + 0.03))
# band test <-> V-Net (pre-alarm only)
xd, xu = Dm["cx"] - 0.2, Dm["cx"] + 0.2
arrow([(xd, Dm["b"]), (xd, Db["t"])], "pre-alarm", (xd - 0.04, (Dm["b"] + Db["t"]) / 2), ha="right", va="center")
arrow([(xu, Db["t"]), (xu, Dm["b"])], "$p_j$", (xu + 0.04, (Dm["b"] + Db["t"]) / 2), ha="left", va="center")
# band test -> fusion (flag) and -> mitigation (z)
xm = (Dm["r"] + cE) / 2
arrow([(Dm["r"], Dm["cy"] + 0.18), (xm, Dm["cy"] + 0.18), (xm, Et["cy"]), (Et["l"], Et["cy"])],
      "flag $f_j$", (xm - 0.04, (Dm["cy"] + Et["cy"]) / 2 + 0.08), ha="right", va="center")
arrow([(Dm["r"], Dm["cy"] - 0.12), (Em["l"], Dm["cy"] - 0.12)], "$z_j$", ((Dm["r"] + Em["l"]) / 2 + 0.02, Dm["cy"] - 0.09), ha="center")
# fusion -> mitigation
arrow([(Et["cx"], Et["b"]), (Et["cx"], Em["t"])], "convict", (Et["cx"] + 0.05, (Et["b"] + Em["t"]) / 2),
      ha="left", va="center")
# one-bit exchange with neighbors via hello
xi, xo = Et["l"] + 0.33, Et["r"] - 0.33
arrow([(xi, H - 0.12), (xi, Et["t"])])
arrow([(xo, Et["t"]), (xo, H - 0.12)])
ax.text(xi - 0.22, H - 0.2, "one-bit flags\nin hello messages", fontsize=5.6, color=INK, ha="right", va="center",
        linespacing=1.25)
ax.text(xi - 0.04, H - 0.26, "in", fontsize=5.3, color=MUTED, ha="right", va="center")
ax.text(xo + 0.04, H - 0.26, "out", fontsize=5.3, color=MUTED, ha="left", va="center")
# mitigation -> forwarding layer
arrow([(Em["cx"], Em["b"]), (Em["cx"], YB + 0.05)])
ax.text(Em["cx"], YB, "forward / rate-limit /\ndrop packets from j", fontsize=5.6, color=INK, ha="center", va="top",
        linespacing=1.3)

# legend
lx, ly = cA, 0.08
for kind, lab in (("input", "observations"), ("learned", "learned model"), ("core", "statistical core"),
                  ("control", "per-node control")):
    k = KIND[kind]
    ax.add_patch(FancyBboxPatch((lx, ly), 0.17, 0.11, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=k["fc"], ec=k["ec"], lw=0.8))
    ax.text(lx + 0.22, ly + 0.055, lab, fontsize=5.6, color=INK, va="center")
    lx += 1.15

fig.savefig("../figures/fig_arch.png", dpi=300); fig.savefig("../figures/fig_arch.pdf")
print("ok")
