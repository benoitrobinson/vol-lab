"""Heston paths via the Andersen (2008) quadratic-exponential scheme.

QE branches on psi, consuming a normal on one branch and a uniform on the
other. Both are always drawn and one is discarded, so the per-step draw count
is fixed at three and a data-dependent branch cannot desynchronise the stream
from a future C++ engine.
"""

import numpy as np
from scipy.special import ndtri

from vollab.rng.scheme import uniforms_block

PSI_C = 1.5


def heston_paths(S0, r, q, T, n_mon, seed, path_start, n_paths,
                 v0, kap_h, th_h, xi, rho, mu=None):
    drift = (r - q) if mu is None else mu
    dt = T / n_mon

    u = uniforms_block(seed, path_start, n_paths, 3 * n_mon)
    zv = ndtri(u[:, 0::3])
    uv = u[:, 1::3]
    zs = ndtri(u[:, 2::3])

    E = np.exp(-kap_h * dt)
    g1 = g2 = 0.5
    k0 = -rho * kap_h * th_h * dt / xi
    k1 = g1 * dt * (kap_h * rho / xi - 0.5) - rho / xi
    k2 = g2 * dt * (kap_h * rho / xi - 0.5) + rho / xi
    k3 = g1 * dt * (1.0 - rho * rho)
    k4 = g2 * dt * (1.0 - rho * rho)

    v = np.full(n_paths, float(v0))
    log_path = np.empty((n_paths, n_mon + 1), dtype=np.float64)
    log_path[:, 0] = 0.0

    for i in range(n_mon):
        m = th_h + (v - th_h) * E
        s2 = (v * xi * xi * E * (1.0 - E) / kap_h
              + th_h * xi * xi * (1.0 - E) ** 2 / (2.0 * kap_h))
        psi = s2 / np.maximum(m * m, 1e-300)

        # Quadratic branch.
        inv = 2.0 / np.maximum(psi, 1e-300)
        b2 = np.maximum(inv - 1.0 + np.sqrt(np.maximum(inv * (inv - 1.0), 0.0)), 0.0)
        a = m / (1.0 + b2)
        v_quad = a * (np.sqrt(b2) + zv[:, i]) ** 2

        # Exponential branch with an atom at zero.
        p = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 1.0 - 1e-15)
        beta = (1.0 - p) / np.maximum(m, 1e-300)
        ui = uv[:, i]
        v_exp = np.where(ui <= p, 0.0,
                         np.log(np.maximum((1.0 - p) / (1.0 - ui), 1e-300)) / beta)

        v_new = np.where(psi <= PSI_C, v_quad, v_exp)
        log_path[:, i + 1] = (log_path[:, i] + drift * dt + k0 + k1 * v + k2 * v_new
                              + np.sqrt(np.maximum(k3 * v + k4 * v_new, 0.0)) * zs[:, i])
        v = v_new

    return S0 * np.exp(log_path)
