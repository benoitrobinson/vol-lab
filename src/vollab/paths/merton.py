"""Merton jump diffusion.

Jump times and sizes are drawn from a stream keyed by path alone, independent
of the step grid. The jump set is therefore identical across every monitoring
grid, which is what makes a frequency sweep under jumps a paired comparison.
Without it, F4's sweep would compare different worlds at each frequency.
"""

import numpy as np
from scipy.special import ndtri
from scipy.stats import poisson

from vollab.pricing.merton import compensator
from vollab.rng.scheme import JUMP_STREAM_SALT, normals_block, uniforms_block


def jump_budget(lam, T, tol=1e-15):
    """Fixed truncation so the per-path draw count cannot depend on the data."""
    return int(poisson.isf(tol, max(lam * T, 1e-12))) + 5


def merton_paths(S0, r, q, s, T, n_mon, seed, path_start, n_paths,
                 lam, mu_J, s_J, mu=None):
    kap = compensator(mu_J, s_J)
    drift = (r - q - lam * kap) if mu is None else (mu - lam * kap)
    dt = T / n_mon

    z = normals_block(seed, path_start, n_paths, n_mon)
    incr = (drift - 0.5 * s * s) * dt + s * np.sqrt(dt) * z

    j_max = jump_budget(lam, T)
    u = uniforms_block(seed ^ JUMP_STREAM_SALT, path_start, n_paths, 1 + 2 * j_max)

    counts = poisson.ppf(u[:, 0], lam * T).astype(np.int64)
    np.minimum(counts, j_max, out=counts)
    times = u[:, 1:1 + j_max] * T
    sizes = mu_J + s_J * ndtri(u[:, 1 + j_max:1 + 2 * j_max])

    live = np.arange(j_max)[None, :] < counts[:, None]
    # Bucket each live jump into the step whose interval contains its time.
    step = np.clip((times / dt).astype(np.int64), 0, n_mon - 1)
    for i in range(n_paths):
        if counts[i]:
            np.add.at(incr[i], step[i, live[i]], sizes[i, live[i]])

    log_path = np.empty((n_paths, n_mon + 1), dtype=np.float64)
    log_path[:, 0] = 0.0
    np.cumsum(incr, axis=1, out=log_path[:, 1:])
    return S0 * np.exp(log_path)
