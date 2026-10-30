"""Numerical checks quoted in the Appendix (Lemma 1, Corollary 1, Proposition 5)."""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # run from anywhere
import math, numpy as np
from scipy.optimize import brentq
k = 2.0
def theta_poisson(eps):
    if math.log(k) >= (k - 1) * (1 - eps): return None          # no positive root (A2 violated)
    return brentq(lambda t: k**t - 1 - t * (k - 1) * (1 - eps), 1e-9, 50)
print("epsilon limit 1 - ln(k)/(k-1) =", round(1 - math.log(k) / (k - 1), 4))
for eps in (0.0, 0.05, 0.1, 0.2, 0.365):
    th = theta_poisson(eps)
    print(f"Poisson eps={eps}: theta={th and round(th, 4)}  h(ARL0=1e5, 4 streams)={th and round(math.log(4e5) / th, 2)}")
c = 0.95
th_nb = brentq(lambda t: t * math.log(k) + math.log(1 - t * (k - 1) / (k * c)), 1e-6, k * c / (k - 1) - 1e-9)
print("NB large-load limit theta =", round(th_nb, 4), " h =", round(math.log(4e5) / th_nb, 2))
def psi(t, lam, cc, r):
    lh = cc * lam; a = math.log(k * (r + lh) / (r + k * lh)); b = r * math.log((r + lh) / (r + k * lh))
    d = 1 - (lam / r) * (math.exp(t * a) - 1)
    return math.inf if d <= 0 else t * b - r * math.log(d)
for r in (0.5, 1, 2, 5, 20):
    worst = max(psi(th_nb, l, cc, r) for l in np.geomspace(0.05, 500, 300) for cc in (0.95, 1.0, 1.3, 2.0))
    print(f"r={r}: max psi(theta_NB) = {worst:.2e}  (negative => Proposition 5 holds)")
kk, m, cc_, q = 6, 3, 1, 1e-3
j = m - cc_; print("Prop. 4 example bound:", (math.e * kk * q / j) ** j)
