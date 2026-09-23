"""Delta-hedging simulator.

Vectorised over paths, looping over the monitoring grid, chunked over path
batches. Hedging a short call means holding +Delta_call shares, so the share
target is the long-option delta.

Paths are generated once per chunk and every schedule in a sweep runs against
the same array. That is not only faster: it is what makes a frequency sweep
Brownian-nested on identical paths, which the paired comparisons require.
"""

import numpy as np

from vollab.hedge.attribution import Accumulator, concat
from vollab.hedge.config import HedgeResult
from vollab.paths.base import GBM, generate
from vollab.pricing.black_scholes import (
    bs_delta, bs_gamma, bs_price, bs_theta, bs_vega,
)
from vollab.rng.scheme import RNG_SCHEME_VERSION

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)


def _payoff(kind, S_T, K):
    return np.maximum(S_T - K, 0.0) if kind == "call" else np.maximum(K - S_T, 0.0)


def _chunk_paths(cfg, start, m):
    model = cfg.model if cfg.model is not None else GBM()
    return generate(model, cfg.contract, cfg.vols, cfg.n_mon,
                    cfg.seed, start, m, cfg.mu)


def _run_on_paths(cfg, S, schedule):
    """One schedule against one block of already-generated paths.

    Wealth is W = cash + held * S - option mark, so the per-step change is
    exactly carry + cost + held * dS - dV, which the attribution explains with
    the book's position greeks: (held - Delta_call), -Gamma_call, -Theta_call.
    """
    c, v = cfg.contract, cfg.vols
    m = S.shape[0]
    k = cfg.cost_bps * 1e-4
    dt = c.T / cfg.n_mon
    s_h = schedule.hedge_vol(v.s_hedge, k, dt)
    acc = Accumulator(m)
    attribute = getattr(cfg, "attribute", True)

    def mark(S_i, tau):
        if tau <= 0.0:
            return _payoff(c.kind, S_i, c.K)
        return bs_price(c.kind, S_i, c.K, tau, c.r, c.q, v.s_imp)

    def target_at(i, held_now):
        if i == cfg.n_mon:
            return np.zeros(m)
        tau = c.T - i * dt
        want = bs_delta(c.kind, S[:, i], c.K, tau, c.r, c.q, s_h)
        if cfg.mv_slope:
            # A delta hedge cannot touch vega, but part of the vega move is
            # predictable from the spot move when the two are correlated.
            # Carrying Vega * d(vol)/dS of extra stock hedges that part; the
            # rest is what F9's floor is made of.
            want = want + cfg.mv_slope * bs_vega(
                c.kind, S[:, i], c.K, tau, c.r, c.q, s_h
            )
        gam = bs_gamma(c.kind, S[:, i], c.K, tau, c.r, c.q, v.s_imp)
        trade = schedule.should_trade(
            i, cfg.n_mon, want, held_now, S[:, i], gam, dt, k
        )
        return np.where(trade, want, held_now)

    cash = np.full(m, bs_price(c.kind, c.S0, c.K, c.T, c.r, c.q, v.s_imp))
    held = np.zeros(m)
    n_reh = np.zeros(m, dtype=np.int64)
    turnover = np.zeros(m)
    mask_hash = np.full(m, _FNV_OFFSET, dtype=np.uint64)

    # Step 0: open the hedge. Its cost is the only P&L, since the premium
    # received exactly equals the option mark at s_imp.
    d = target_at(0, held) - held
    traded = d != 0.0
    cost_paid = -k * np.abs(d) * S[:, 0]
    cash = cash - d * S[:, 0] + cost_paid
    acc.cost += cost_paid
    turnover += np.abs(d) * S[:, 0]
    n_reh += traded
    mask_hash = np.where(traded, (mask_hash ^ np.uint64(0)) * _FNV_PRIME, mask_hash)
    held = held + d
    mark_prev = mark(S[:, 0], c.T)

    for i in range(1, cfg.n_mon + 1):
        S_prev, S_now = S[:, i - 1], S[:, i]
        tau_prev = c.T - (i - 1) * dt
        cash_prev, held_prev = cash, held
        w_prev = cash_prev + held_prev * S_prev - mark_prev

        cash = cash * np.exp(c.r * dt) + c.q * held * S_prev * dt

        d = target_at(i, held) - held
        traded = d != 0.0
        cost_paid = -k * np.abs(d) * S_now
        cash = cash - d * S_now + cost_paid
        turnover += np.abs(d) * S_now
        n_reh += traded
        mask_hash = np.where(traded, (mask_hash ^ np.uint64(i)) * _FNV_PRIME, mask_hash)
        held = held + d

        if not attribute:
            continue

        mark_now = mark(S_now, c.T - i * dt)
        realized = (cash + held * S_now - mark_now) - w_prev

        acc.step(
            delta_mark=held_prev - bs_delta(c.kind, S_prev, c.K, tau_prev, c.r, c.q, v.s_imp),
            gamma_mark=-bs_gamma(c.kind, S_prev, c.K, tau_prev, c.r, c.q, v.s_imp),
            theta_mark=-bs_theta(c.kind, S_prev, c.K, tau_prev, c.r, c.q, v.s_imp),
            dS=S_now - S_prev, dt=dt, cash_prev=cash_prev, r=c.r, q=c.q,
            held_prev=held_prev, S_prev=S_prev, cost_paid=cost_paid, realized=realized,
        )
        mark_prev = mark_now

    pnl = cash - _payoff(c.kind, S[:, -1], c.K)
    return (pnl, n_reh, turnover, mask_hash, acc.finish())


def _assemble(parts):
    pnl, n_reh, turnover, mask_hash = (np.concatenate(p) for p in parts[:4])
    return HedgeResult(
        pnl=pnl, n_rehedges=n_reh, turnover=turnover, rehedge_mask_hash=mask_hash,
        engine_used="numpy", rng_scheme_version=RNG_SCHEME_VERSION,
        attribution=concat(parts[4]),
    )


def cpp_available():
    try:
        from vollab import _core  # noqa: F401
        return True
    except ImportError:
        return False


def _check_engine(engine):
    if engine == "cpp":
        if not cpp_available():
            raise NotImplementedError(
                "C++ extension is not built; refusing to fall back to numpy. "
                "Run: uv sync --reinstall-package vollab"
            )
        return
    if engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")


def _simulate_cpp(cfg):
    """Fast path. Supports GBM with a FixedTime schedule only; anything else
    raises rather than quietly running the reference engine."""
    from vollab import _core

    from vollab.hedge.schedule import FixedTime

    model = cfg.model if cfg.model is not None else GBM()
    if not isinstance(model, GBM):
        raise NotImplementedError(f"cpp engine supports GBM only, got {model.name}")
    if not isinstance(cfg.schedule, FixedTime):
        raise NotImplementedError("cpp engine supports FixedTime only")
    if cfg.contract.kind != "call":
        raise NotImplementedError("cpp engine supports calls only")

    c, v = cfg.contract, cfg.vols
    pnl, n_reh = _core.hedge_gbm(
        c.S0, c.K, c.T, c.r, c.q, v.s_imp, v.s_hedge, v.s_real,
        float("nan") if cfg.mu is None else cfg.mu, cfg.cost_bps,
        cfg.n_mon, cfg.schedule.every, cfg.seed, 0, cfg.n_paths,
    )
    return HedgeResult(
        pnl=pnl, n_rehedges=n_reh, turnover=np.zeros_like(pnl),
        rehedge_mask_hash=np.zeros(pnl.size, dtype=np.uint64),
        engine_used="cpp", rng_scheme_version=RNG_SCHEME_VERSION,
    )
    # turnover and rehedge_mask_hash are not computed by the C++ engine. They
    # are left as zeros rather than silently wrong values; callers that need
    # them must use the reference engine. See test_engine_parity.


def simulate(cfg, engine="numpy"):
    """Run one schedule over all paths."""
    _check_engine(engine)
    if engine == "cpp":
        return _simulate_cpp(cfg)
    parts = [[], [], [], [], []]
    for start in range(0, cfg.n_paths, cfg.chunk_paths):
        m = min(cfg.chunk_paths, cfg.n_paths - start)
        S = _chunk_paths(cfg, start, m)
        for acc, piece in zip(parts, _run_on_paths(cfg, S, cfg.schedule)):
            acc.append(piece)
    return _assemble(parts)


def sweep(cfg, schedules, engine="numpy"):
    """Run several schedules over one set of paths, generated once.

    Returns results in the order the schedules were given.
    """
    _check_engine(engine)
    per_sched = [[[], [], [], [], []] for _ in schedules]
    for start in range(0, cfg.n_paths, cfg.chunk_paths):
        m = min(cfg.chunk_paths, cfg.n_paths - start)
        S = _chunk_paths(cfg, start, m)
        for parts, sched in zip(per_sched, schedules):
            for acc, piece in zip(parts, _run_on_paths(cfg, S, sched)):
                acc.append(piece)
    return [_assemble(parts) for parts in per_sched]
