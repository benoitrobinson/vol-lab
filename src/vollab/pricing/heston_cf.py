"""Heston (1993) semi-analytic price.

Uses the Albrecher et al. "little trap" form of the characteristic function,
which keeps the complex logarithm on its principal branch for long maturities
where the textbook form crosses a branch cut and returns nonsense.
"""

import numpy as np
from scipy.integrate import quad


def feller_ratio(kap_h, th_h, xi):
    """2*kappa*theta / xi^2. Above 1 the variance cannot reach zero."""
    return 2.0 * kap_h * th_h / (xi * xi)


def _phi(u, j, S, T, r, q, v0, kap_h, th_h, xi, rho):
    uj = 0.5 if j == 1 else -0.5
    bj = kap_h - rho * xi if j == 1 else kap_h
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - bj) ** 2 - xi * xi * (2 * uj * iu - u * u))
    num = bj - rho * xi * iu - d
    g = num / (bj - rho * xi * iu + d)
    ed = np.exp(-d * T)
    C = ((r - q) * iu * T
         + (kap_h * th_h / (xi * xi))
         * (num * T - 2.0 * np.log((1.0 - g * ed) / (1.0 - g))))
    D = (num / (xi * xi)) * ((1.0 - ed) / (1.0 - g * ed))
    return np.exp(C + D * v0 + iu * np.log(S))


def _P(j, S, K, T, r, q, v0, kap_h, th_h, xi, rho, limit=200):
    def integrand(u):
        val = np.exp(-1j * u * np.log(K)) * _phi(u, j, S, T, r, q, v0,
                                                 kap_h, th_h, xi, rho) / (1j * u)
        return val.real

    val, _ = quad(integrand, 1e-10, 200.0, limit=limit)
    return 0.5 + val / np.pi


def heston_price(kind, S, K, T, r, q, v0, kap_h, th_h, xi, rho):
    p1 = _P(1, S, K, T, r, q, v0, kap_h, th_h, xi, rho)
    p2 = _P(2, S, K, T, r, q, v0, kap_h, th_h, xi, rho)
    call = S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    if kind == "call":
        return call
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)
