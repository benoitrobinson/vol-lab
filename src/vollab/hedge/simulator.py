"""Delta-hedging simulator.

Vectorised over paths, looping over the monitoring grid, chunked over path
batches. Hedging a short call means holding +Delta_call shares, so the share
target is the long-option delta.
"""

import numpy as np

from vollab.hedge.config import HedgeResult
from vollab.paths.gbm import gbm_paths
from vollab.pricing.black_scholes import bs_delta, bs_gamma, bs_price
from vollab.rng.scheme import RNG_SCHEME_VERSION

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)


def _payoff(kind, S_T, K):
    return np.maximum(S_T - K, 0.0) if kind == "call" else np.maximum(K - S_T, 0.0)


def _simulate_chunk(cfg, start, m):
    c, v = cfg.contract, cfg.vols
    k = cfg.cost_bps * 1e-4
    dt = c.T / cfg.n_mon
    s_h = cfg.schedule.hedge_vol(v.s_hedge, k, dt)

    S = gbm_paths(c.S0, c.r, c.q, v.s_real, c.T, cfg.n_mon,
                  cfg.seed, start, m, cfg.mu)

    cash = np.full(m, bs_price(c.kind, c.S0, c.K, c.T, c.r, c.q, v.s_imp))
    held = np.zeros(m)
    n_reh = np.zeros(m, dtype=np.int64)
    turnover = np.zeros(m)
    mask_hash = np.full(m, _FNV_OFFSET, dtype=np.uint64)

    for i in range(cfg.n_mon + 1):
        if i > 0:
            cash = cash * np.exp(c.r * dt) + c.q * held * S[:, i - 1] * dt

        if i == cfg.n_mon:
            target = np.zeros(m)
        else:
            tau = c.T - i * dt
            want = bs_delta(c.kind, S[:, i], c.K, tau, c.r, c.q, s_h)
            gam = bs_gamma(c.kind, S[:, i], c.K, tau, c.r, c.q, v.s_imp)
            trade = cfg.schedule.should_trade(
                i, cfg.n_mon, want, held, S[:, i], gam, dt
            )
            target = np.where(trade, want, held)

        d = target - held
        traded = d != 0.0
        cash = cash - d * S[:, i] - k * np.abs(d) * S[:, i]
        turnover += np.abs(d) * S[:, i]
        n_reh += traded
        mask_hash = np.where(
            traded, (mask_hash ^ np.uint64(i)) * _FNV_PRIME, mask_hash
        )
        held = target

    pnl = cash - _payoff(c.kind, S[:, -1], c.K)
    return pnl, n_reh, turnover, mask_hash


def simulate(cfg, engine="numpy"):
    """Run the experiment. Never falls back silently between engines."""
    if engine == "cpp":
        raise NotImplementedError(
            "C++ engine lands in Phase A part 2; refusing to fall back to numpy"
        )
    if engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")

    parts = [[], [], [], []]
    for start in range(0, cfg.n_paths, cfg.chunk_paths):
        m = min(cfg.chunk_paths, cfg.n_paths - start)
        for acc, piece in zip(parts, _simulate_chunk(cfg, start, m)):
            acc.append(piece)

    pnl, n_reh, turnover, mask_hash = (np.concatenate(p) for p in parts)
    return HedgeResult(
        pnl=pnl,
        n_rehedges=n_reh,
        turnover=turnover,
        rehedge_mask_hash=mask_hash,
        engine_used="numpy",
        rng_scheme_version=RNG_SCHEME_VERSION,
    )
