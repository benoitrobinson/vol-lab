"""Rough Bergomi paths via the hybrid scheme of Bennedsen, Lunde and Pakkanen.

The variance is driven by a Volterra process with a power kernel,

    Y_t = int_0^t (t - s)^(H - 1/2) dW_s,    V_t = xi0 * exp(eta * sqrt(2H) * Y_t
                                                             - 0.5 * eta^2 * t^(2H))

so `E[V_t] = xi0` exactly, and the spot follows `dS = S sqrt(V) dB` with
`B = rho W + sqrt(1 - rho^2) Z`. For `H = 1/2` the kernel is flat, `Y` is a
Brownian motion and the model collapses to a lognormal (Bergomi) variance; for
`H < 1/2` the variance is rougher than any diffusion, which is what produces an
at-the-money skew that blows up as the maturity shortens.

The hybrid scheme treats the singular first interval exactly, by drawing the
Brownian increment and its kernel-weighted integral as a correlated pair, and
approximates the rest of the convolution on the optimal discretisation points
`b_k` of Bennedsen, Lunde and Pakkanen (2017). Exact simulation by Cholesky is
O(n^2) in memory per path and was rejected for that reason; the hybrid scheme
reproduces `Var(Y_t) = t^(2H) / (2H)` to well inside Monte Carlo error, which is
what `test_rbergomi.py` gates it on.

Three normals per step, always drawn: the pair for the Volterra process and one
orthogonal draw for the spot. The budget is fixed for the same reason as Heston,
so no branch can desynchronise the stream.
"""

import numpy as np
from scipy.signal import fftconvolve

from vollab.rng.scheme import normals_block


def _kernel_weights(a, n_mon, dt):
    """Kernel evaluated at the optimal discretisation points, for lags 2 and up.

    Index 0 is the lag-one term, which the correlated pair handles exactly, so
    it is zero here and never contributes to the convolution.
    """
    k = np.arange(2, n_mon + 1, dtype=np.float64)
    if a == 0.0:
        # H = 1/2: the kernel is identically one and b_k is undefined, because
        # there is nothing left to optimise.
        return np.concatenate(([0.0], np.ones_like(k)))
    b = ((k ** (a + 1.0) - (k - 1.0) ** (a + 1.0)) / (a + 1.0)) ** (1.0 / a)
    return np.concatenate(([0.0], (b * dt) ** a))


def volterra_paths(H, n_mon, dt, z1, z2):
    """The Volterra process on the fine grid, shape (n_paths, n_mon + 1).

    `z1` drives the Brownian motion, `z2` the part of the first-interval
    integral that the increment does not explain.
    """
    a = H - 0.5
    s11 = dt
    s12 = dt ** (a + 1.0) / (a + 1.0)
    s22 = dt ** (2.0 * a + 1.0) / (2.0 * a + 1.0)
    c1 = s12 / np.sqrt(s11)
    c2 = np.sqrt(max(s22 - c1 * c1, 0.0))

    dW = np.sqrt(dt) * z1
    first = (c1 * z1) + (c2 * z2)

    w = _kernel_weights(a, n_mon, dt)
    tail = fftconvolve(dW, w[None, :], mode="full", axes=1)[:, :n_mon]

    y = np.zeros((dW.shape[0], n_mon + 1), dtype=np.float64)
    y[:, 1:] = first + tail
    return y, dW


def rbergomi_paths(S0, r, q, T, n_mon, seed, path_start, n_paths,
                   xi0, H, eta, rho, mu=None):
    """Spot paths on the fine monitoring grid, shape (n_paths, n_mon + 1)."""
    if not 0.0 < H < 1.0:
        raise ValueError(f"H must be in (0, 1), got {H}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    drift = (r - q) if mu is None else mu
    dt = T / n_mon

    z = normals_block(seed, path_start, n_paths, 3 * n_mon)
    y, dW = volterra_paths(H, n_mon, dt, z[:, 0::3], z[:, 1::3])
    dZ = np.sqrt(dt) * z[:, 2::3]

    t = np.arange(n_mon + 1, dtype=np.float64) * dt
    v = xi0 * np.exp(eta * np.sqrt(2.0 * H) * y - 0.5 * eta * eta * t ** (2.0 * H))

    dB = (rho * dW) + (np.sqrt(1.0 - rho * rho) * dZ)
    incr = (drift - 0.5 * v[:, :n_mon]) * dt + np.sqrt(v[:, :n_mon]) * dB

    log_path = np.empty((n_paths, n_mon + 1), dtype=np.float64)
    log_path[:, 0] = 0.0
    np.cumsum(incr, axis=1, out=log_path[:, 1:])
    return S0 * np.exp(log_path)


def variance_paths(T, n_mon, seed, path_start, n_paths, xi0, H, eta):
    """The variance process alone, for the tests that gate the kernel."""
    dt = T / n_mon
    z = normals_block(seed, path_start, n_paths, 3 * n_mon)
    y, _ = volterra_paths(H, n_mon, dt, z[:, 0::3], z[:, 1::3])
    t = np.arange(n_mon + 1, dtype=np.float64) * dt
    return y, xi0 * np.exp(eta * np.sqrt(2.0 * H) * y - 0.5 * eta * eta * t ** (2.0 * H))
