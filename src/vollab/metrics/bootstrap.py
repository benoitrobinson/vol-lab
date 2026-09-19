"""Resampling statistics.

Comparisons are paired: both arms run on identical paths, the difference is
taken per path, and resampling is across paths. The two arms being perfectly
correlated on a given path is the point, not a problem; it removes variance
from the estimator. Paths are independent, so the bootstrap is valid.

sd gets its own bootstrap because it is nonlinear: its sampling error is not
s/sqrt(n), and both the law-recovery and the jump-floor findings depend on it.
"""

import numpy as np


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Mean difference and a 95 percent interval over resampled paths."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, d.size, d.size)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def bootstrap_sd(x, n_boot=2000, seed=0):
    """Sample sd and its bootstrap standard error."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    sds = np.empty(n_boot)
    for i in range(n_boot):
        sds[i] = x[rng.integers(0, x.size, x.size)].std(ddof=1)
    return float(x.std(ddof=1)), float(sds.std(ddof=1))
