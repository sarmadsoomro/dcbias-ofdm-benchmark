"""Zhang & Zhang 2014 (doi 10.1109/LCOMM.2014.2331068) Table I is not reproducible from its eq. (15).

Baseline B2 therefore credits Zhang's objective, not its numbers:

  1. eq. (15) as printed, d gamma / d sigma_DC = 2 sigma_DC [Phi(U) - Phi(L)] + 2 sigma_y [phi(L) - phi(U)],
     U = (sigma_U - sigma_DC)/sigma_y, L = (sigma_L - sigma_DC)/sigma_y, has no root on [0, 9] for case 4;
  2. the stated objective, the clipping-noise variance E[(clip(y, sigma_L, sigma_U) - y)^2] with
     y = x + sigma_DC, x ~ N(0, 1), is minimised at the midpoint (sigma_L + sigma_U)/2 in every case;
  3. Table I's published optima differ from the midpoint.

Usage: python -m dcbias.zhang_check   (exit 0; prints the comparison table)
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

TABLE_I = [  # (sigma_L, sigma_U, simulated optimum, theoretical optimum), N = 256, 16-QAM
    (0, 3, 1.6, 1.8),
    (0, 5, 2.5, 2.39),
    (1, 6, 3.5, 3.39),
    (1, 8, 3.6, 3.64),
]


def eq15(dc, lo, hi, sy=1.0):
    U, L = (hi - dc) / sy, (lo - dc) / sy
    return 2 * dc * (norm.cdf(U) - norm.cdf(L)) + 2 * sy * (norm.pdf(L) - norm.pdf(U))


def clip_noise(dc, lo, hi):
    x = np.linspace(-10, 10, 40001)
    w = norm.pdf(x)
    y = x + dc
    return np.sum((np.clip(y, lo, hi) - y) ** 2 * w) / np.sum(w)


def rows():
    out = []
    for lo, hi, sim, theo in TABLE_I:
        g = eq15(np.linspace(0, 9, 9001), lo, hi)
        best = minimize_scalar(clip_noise, bounds=(lo, hi), args=(lo, hi), method="bounded", options={"xatol": 1e-6}).x
        out.append({"lo": lo, "hi": hi, "table_sim": sim, "table_theory": theo, "eq15_has_root": bool(np.any(np.sign(g[:-1]) != np.sign(g[1:]))),
                    "argmin_clip_noise": best, "midpoint": (lo + hi) / 2})
    return out


if __name__ == "__main__":
    print(f"{'bounds':>8} {'TableI sim':>10} {'TableI th':>9} {'eq15 root':>9} {'argmin MSE':>10} {'midpoint':>8}")
    for r in rows():
        print(f"{r['lo']:>3}-{r['hi']:<4} {r['table_sim']:>10} {r['table_theory']:>9} {str(r['eq15_has_root']):>9} {r['argmin_clip_noise']:>10.4f} {r['midpoint']:>8}")
