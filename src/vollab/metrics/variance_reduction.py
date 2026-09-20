"""Variance reduction for hedging-error estimators.

Two results, one positive and one negative, both derived rather than tuned.

**Antithetic sampling does not work here.** The one-step hedging error is
0.5*Gamma*S^2*s^2*dt*(z^2 - 1), which is *even* in z. Mirroring a path leaves
the error statistic unchanged, so an antithetic pair is perfectly positively
correlated and removes no variance at all. Measured correlation between the
statistic on z and on -z: exactly +1.0000.

**The natural control variate is realized minus implied variance.** From the
same expression, summing over steps gives an error proportional to
sum_i (z_i^2 - 1), whose mean is zero and which is computable from the driving
normals at no extra cost. Correlation with terminal P&L is about -0.79 and the
variance ratio about 0.37, so roughly a 2.7x effective increase in sample size.

The textbook choices fail, and they fail for an instructive reason. The
discounted terminal payoff and the terminal spot both correlate with P&L at
under 0.04 and buy nothing, because a working delta hedge removes precisely the
component of P&L that tracks the terminal value. What survives is quadratic
variation, so the control has to be a quadratic-variation quantity too.
"""

import numpy as np

from vollab.rng.scheme import normals_block


def realized_variance_control(seed, n_paths, n_mon, path_start=0):
    """sum_i (z_i^2 - 1) for each path. Mean zero by construction."""
    z = normals_block(seed, path_start, n_paths, n_mon)
    return (z * z - 1.0).sum(axis=1)


def apply_control(x, control, control_mean=0.0):
    """Regress x on a zero-mean control and return the adjusted sample.

    Returns (adjusted, beta, correlation). The adjusted sample has the same
    expectation as x and, when the correlation is strong, materially less
    variance.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(control, dtype=float)
    var_c = np.var(c, ddof=1)
    if var_c <= 0:
        return x.copy(), 0.0, 0.0
    beta = np.cov(x, c, ddof=1)[0, 1] / var_c
    rho = float(np.corrcoef(x, c)[0, 1])
    return x - beta * (c - control_mean), float(beta), rho


def variance_ratio(x, adjusted):
    """Variance of the adjusted estimator over the plain one. Below 1 is a win."""
    v0 = np.var(np.asarray(x), ddof=1)
    return float(np.var(np.asarray(adjusted), ddof=1) / v0) if v0 > 0 else 1.0
