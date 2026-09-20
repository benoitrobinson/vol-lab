"""Raw SVI parameterisation of a volatility smile (Gatheral).

Total implied variance as a function of log-moneyness k = log(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Implied volatility is sqrt(w / T). The slice is quoted in total variance rather
than volatility because the no-arbitrage conditions are natural there.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVIParams:
    a: float       # vertical shift of total variance
    b: float       # wing slope, b >= 0
    rho: float     # wing asymmetry, |rho| < 1
    m: float       # horizontal shift
    sigma: float   # curvature at the money, sigma > 0

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @staticmethod
    def from_array(x):
        return SVIParams(*(float(v) for v in x))


def total_variance(p, k):
    """w(k). Vectorised over log-moneyness."""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma ** 2))


def implied_vol(p, k, T):
    return np.sqrt(np.maximum(total_variance(p, k), 1e-12) / T)


def _derivatives(p, k):
    """w, w' and w'' at k, all in closed form."""
    k = np.asarray(k, dtype=float)
    d = k - p.m
    root = np.sqrt(d * d + p.sigma ** 2)
    w = p.a + p.b * (p.rho * d + root)
    dw = p.b * (p.rho + d / root)
    d2w = p.b * p.sigma ** 2 / (root ** 3)
    return w, dw, d2w


def durrleman_g(p, k):
    """Durrleman's function. Non-negative everywhere iff the slice is free of
    butterfly arbitrage, which is the same as saying the risk-neutral density
    it implies is non-negative."""
    w, dw, d2w = _derivatives(p, k)
    term = 1.0 - k * dw / (2.0 * w)
    return (term ** 2
            - 0.25 * dw ** 2 * (0.25 + 1.0 / w)
            + 0.5 * d2w)


def risk_neutral_density(p, k, T):
    """The density implied by the slice, up to the forward scaling."""
    w, dw, _ = _derivatives(p, k)
    g = durrleman_g(p, k)
    d_minus = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
    return g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus ** 2)


def butterfly_violation(p, k_grid):
    """How far the slice dips below zero on Durrleman's function. Zero is clean."""
    return float(max(0.0, -np.min(durrleman_g(p, k_grid))))


def min_total_variance(p):
    """Smallest w over all k. Must be positive for the slice to make sense."""
    return float(p.a + p.b * p.sigma * np.sqrt(max(1.0 - p.rho ** 2, 0.0)))


def wing_slopes(p):
    """Asymptotic slopes of w(k). Lee's moment formula caps both at 2."""
    return p.b * (1.0 - p.rho), p.b * (1.0 + p.rho)


def calendar_violation(near, far, k_grid):
    """Total variance must not decrease with maturity at any strike."""
    gap = total_variance(far, k_grid) - total_variance(near, k_grid)
    return float(max(0.0, -np.min(gap)))
