"""Merton (1976) jump-diffusion price.

A Poisson-weighted sum of Black-Scholes prices, but each term carries its own
rate and its own variance. Reusing one r and one sigma across terms is the
common mistake and gives a wrong ground truth.
"""

import numpy as np
from scipy.stats import poisson

from vollab.pricing.black_scholes import bs_price


def compensator(mu_J, s_J):
    """kap = E[J] - 1 for lognormal jump sizes."""
    return np.exp(mu_J + 0.5 * s_J * s_J) - 1.0


def merton_price(kind, S, K, T, r, q, s, lam, mu_J, s_J, tol=1e-12):
    kap = compensator(mu_J, s_J)
    lam_p = lam * (1.0 + kap)
    n_max = int(poisson.isf(tol, max(lam_p * T, 1e-12))) + 5

    total = 0.0
    for n in range(n_max + 1):
        w = poisson.pmf(n, lam_p * T)
        if w < tol and n > lam_p * T:
            break
        s_n = np.sqrt(s * s + n * s_J * s_J / T)
        r_n = r - lam * kap + n * np.log1p(kap) / T
        total += w * bs_price(kind, S, K, T, r_n, q, s_n)
    return total
