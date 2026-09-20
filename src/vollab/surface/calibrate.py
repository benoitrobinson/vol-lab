"""SVI calibration, quasi-explicit, under no-arbitrage constraints.

Fitting all five raw parameters at once is a badly conditioned non-convex
problem: on a short-dated smile where total variance is order 0.01, a general
local optimiser misses a perfect fit by several volatility points even when the
target is itself an SVI slice.

The quasi-explicit reduction (Zeliade Systems, after Gatheral) avoids that. With
y = (k - m) / sigma the slice is

    w = a + d*y + c*sqrt(y^2 + 1),    d = rho*b*sigma,  c = b*sigma

which is *linear* in (a, d, c). So for any (m, sigma) the best (a, d, c) is a
bounded linear least squares solve, exact and instant, and only the two
remaining parameters need searching.

The inner bounds encode the admissibility conditions:
    0 <= c <= 4*sigma        wing slopes bounded (Lee's moment formula)
    |d| <= c                 |rho| <= 1
    |d| <= 4*sigma - c       right wing bounded
    0 <= a <= max(w)         total variance positive, no better than the data

A final constrained polish then enforces Durrleman's condition directly, so the
fitted slice implies a non-negative density rather than merely being checked for
one afterwards.
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize

from vollab.surface.svi import (
    SVIParams, durrleman_g, min_total_variance, total_variance, wing_slopes,
)

W_FLOOR = 1e-10


def _inner(m, sigma, k, w):
    """Exact bounded least squares for (a, d, c) at fixed (m, sigma)."""
    y = (k - m) / sigma
    root = np.sqrt(y * y + 1.0)
    X = np.column_stack([np.ones_like(y), y, root])
    w_max = float(np.max(w))
    lo = [0.0, -4.0 * sigma, 0.0]
    hi = [max(w_max, 1e-8), 4.0 * sigma, 4.0 * sigma]
    res = lsq_linear(X, w, bounds=(lo, hi), max_iter=200)
    a, d, c = res.x
    # Project onto |d| <= c and |d| <= 4*sigma - c.
    d = float(np.clip(d, -c, c))
    d = float(np.clip(d, -(4.0 * sigma - c), 4.0 * sigma - c))
    resid = X @ np.array([a, d, c]) - w
    return (a, d, c), float(resid @ resid)


def _to_params(m, sigma, a, d, c):
    b = c / sigma
    rho = 0.0 if c <= 0 else float(np.clip(d / c, -0.999, 0.999))
    return SVIParams(a=float(a), b=float(b), rho=rho, m=float(m), sigma=float(sigma))


def _outer(k, w, span):
    """Grid then Nelder-Mead over the only two parameters left."""
    def loss(z):
        m, log_sigma = z
        sigma = float(np.exp(log_sigma))
        if not (1e-4 <= sigma <= 10.0 * span):
            return 1e9
        return _inner(m, sigma, k, w)[1]

    best, best_val = None, np.inf
    for m0 in np.linspace(k.min(), k.max(), 9):
        for s0 in (0.02 * span, 0.05 * span, 0.15 * span, 0.4 * span, 1.0 * span):
            v = loss([m0, np.log(max(s0, 1e-4))])
            if v < best_val:
                best, best_val = [m0, np.log(max(s0, 1e-4))], v

    res = minimize(loss, best, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-14})
    m, sigma = res.x[0], float(np.exp(res.x[1]))
    (a, d, c), _ = _inner(m, sigma, k, w)
    return _to_params(m, sigma, a, d, c)


def _polish(p0, k, iv_mkt, T, weights, k_grid, span):
    """Enforce Durrleman's condition directly, starting from the quasi-explicit fit."""
    def obj(x):
        p = SVIParams.from_array(x)
        w = np.maximum(total_variance(p, k), W_FLOOR)
        return float(np.sum(((np.sqrt(w / T) - iv_mkt) * weights) ** 2))

    lo, hi = float(k.min()), float(k.max())
    bounds = [(-1.0, 5.0), (0.0, 10.0), (-0.999, 0.999),
              (lo - span, hi + span), (1e-5, 10.0 * span)]
    cons = [
        {"type": "ineq",
         "fun": lambda x: min_total_variance(SVIParams.from_array(x)) - W_FLOOR},
        {"type": "ineq",
         "fun": lambda x: 2.0 - max(wing_slopes(SVIParams.from_array(x)))},
        {"type": "ineq",
         "fun": lambda x: np.min(durrleman_g(SVIParams.from_array(x), k_grid))},
    ]
    res = minimize(obj, p0.as_array(), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 600, "ftol": 1e-14})
    return SVIParams.from_array(res.x) if res.success else None


def calibrate_svi(k, iv_mkt, T, weights=None, arb_free=True, n_grid=201):
    """Fit one slice. Returns (params, rmse_in_vol, diagnostics)."""
    k = np.asarray(k, dtype=float)
    iv_mkt = np.asarray(iv_mkt, dtype=float)
    if not np.all(np.isfinite(iv_mkt)) or not np.all(np.isfinite(k)):
        raise RuntimeError("SVI calibration needs finite strikes and vols")
    weights = np.ones_like(k) if weights is None else np.asarray(weights, float)

    w = iv_mkt ** 2 * T
    lo, hi = float(k.min()), float(k.max())
    span = max(hi - lo, 0.1)
    k_grid = np.linspace(lo - 0.5 * span, hi + 0.5 * span, n_grid)

    p = _outer(k, w, span)

    if arb_free:
        polished = _polish(p, k, iv_mkt, T, weights, k_grid, span)
        if polished is not None:
            p = polished
        elif np.min(durrleman_g(p, k_grid)) < 0:
            raise RuntimeError("no arbitrage-free SVI fit found for this slice")
    elif weights.std() > 0:
        free = _polish_unconstrained(p, k, iv_mkt, T, weights, span)
        if free is not None:
            p = free

    fitted = np.sqrt(np.maximum(total_variance(p, k), W_FLOOR) / T)
    rmse = float(np.sqrt(np.mean((fitted - iv_mkt) ** 2)))
    diag = {
        "rmse_vol_points": rmse * 100.0,
        "max_abs_err_vol_points": float(np.max(np.abs(fitted - iv_mkt))) * 100.0,
        "min_durrleman_g": float(np.min(durrleman_g(p, k_grid))),
        "min_total_variance": min_total_variance(p),
        "wing_left": wing_slopes(p)[0],
        "wing_right": wing_slopes(p)[1],
    }
    return p, rmse, diag


def _polish_unconstrained(p0, k, iv_mkt, T, weights, span):
    def obj(x):
        p = SVIParams.from_array(x)
        w = np.maximum(total_variance(p, k), W_FLOOR)
        return float(np.sum(((np.sqrt(w / T) - iv_mkt) * weights) ** 2))

    lo, hi = float(k.min()), float(k.max())
    bounds = [(-1.0, 5.0), (0.0, 10.0), (-0.999, 0.999),
              (lo - span, hi + span), (1e-5, 10.0 * span)]
    res = minimize(obj, p0.as_array(), method="SLSQP", bounds=bounds,
                   options={"maxiter": 600, "ftol": 1e-14})
    return SVIParams.from_array(res.x) if res.success else None
