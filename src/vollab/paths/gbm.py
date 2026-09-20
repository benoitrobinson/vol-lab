"""Geometric Brownian motion, simulated exactly in log space."""

import numpy as np

from vollab.rng.scheme import normals_block


def gbm_paths(S0, r, q, s, T, n_mon, seed, path_start, n_paths, mu=None):
    """Paths on the fine monitoring grid, shape (n_paths, n_mon + 1).

    Always generated on the finest grid; coarser rehedge frequencies subsample
    this, which keeps every frequency in a sweep on identical paths.
    """
    drift = (r - q) if mu is None else mu
    dt = T / n_mon
    z = normals_block(seed, path_start, n_paths, n_mon)
    incr = (drift - 0.5 * s * s) * dt + s * np.sqrt(dt) * z
    log_path = np.empty((n_paths, n_mon + 1), dtype=np.float64)
    log_path[:, 0] = 0.0
    np.cumsum(incr, axis=1, out=log_path[:, 1:])
    return S0 * np.exp(log_path)
