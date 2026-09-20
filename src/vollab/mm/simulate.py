"""Market-making simulation.

The mid is arithmetic Brownian motion, which is the Avellaneda-Stoikov setting
and keeps the inventory risk linear in the position. Fills are Bernoulli draws
per step at probability lam(d) * dt, with lam(d) = A * exp(-kappa * d).

All randomness comes from the project's counter-based scheme, keyed per path, so
two quoting strategies compared on the same seed see the same mid path and the
same fill randomness. The comparison is therefore paired: any difference is the
strategy, not the draws.
"""

from dataclasses import dataclass

import numpy as np

from vollab.mm.quoting import (
    optimal_half_spreads, reservation_price, symmetric_half_spreads,
)
from vollab.rng.scheme import JUMP_STREAM_SALT, normals_block, uniforms_block

STRATEGIES = {"avellaneda_stoikov": optimal_half_spreads,
              "symmetric": symmetric_half_spreads}


@dataclass(frozen=True)
class MMResult:
    pnl: np.ndarray            # (n_paths,) terminal wealth
    inventory_end: np.ndarray  # (n_paths,)
    inventory_max_abs: np.ndarray
    n_fills: np.ndarray
    spread_captured: np.ndarray
    strategy: str


def simulate_mm(market, dealer, strategy, seed, n_paths, path_start=0):
    """Run one quoting strategy over n_paths independent sessions."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {sorted(STRATEGIES)}")
    half_spreads = STRATEGIES[strategy]

    n, dt = market.n_steps, market.T / market.n_steps
    sq = np.sqrt(dt)

    dz = normals_block(seed, path_start, n_paths, n)
    # Fill draws come from a salted stream so they are disjoint from the mid.
    u = uniforms_block(seed ^ JUMP_STREAM_SALT, path_start, n_paths, 2 * n)

    s = np.full(n_paths, float(market.s0))
    q = np.zeros(n_paths, dtype=np.int64)
    cash = np.zeros(n_paths)
    inv_max = np.zeros(n_paths, dtype=np.int64)
    fills = np.zeros(n_paths, dtype=np.int64)
    captured = np.zeros(n_paths)

    for i in range(n):
        tau = market.T - i * dt
        d_a, d_b = half_spreads(q, dealer.gam, market.sigma, tau, market.kappa)

        # Half-spreads are not floored at zero. A heavily long dealer quotes an
        # ask through the mid to unwind, which is the model's behaviour, not an
        # artefact; the fill probability is what gets capped.
        p_a = np.clip(market.A * np.exp(-market.kappa * d_a) * dt, 0.0, 1.0)
        p_b = np.clip(market.A * np.exp(-market.kappa * d_b) * dt, 0.0, 1.0)

        # A dealer at its position limit stops quoting the side that would breach it.
        can_sell = q > -dealer.max_inventory
        can_buy = q < dealer.max_inventory

        hit_ask = (u[:, 2 * i] < p_a) & can_sell
        hit_bid = (u[:, 2 * i + 1] < p_b) & can_buy

        cash += np.where(hit_ask, s + d_a, 0.0)
        cash -= np.where(hit_bid, s - d_b, 0.0)
        captured += np.where(hit_ask, d_a, 0.0) + np.where(hit_bid, d_b, 0.0)
        q = q + hit_bid.astype(np.int64) - hit_ask.astype(np.int64)
        fills += hit_ask.astype(np.int64) + hit_bid.astype(np.int64)
        inv_max = np.maximum(inv_max, np.abs(q))

        s = s + market.sigma * sq * dz[:, i]

    return MMResult(
        pnl=cash + q * s,
        inventory_end=q,
        inventory_max_abs=inv_max,
        n_fills=fills,
        spread_captured=captured,
        strategy=strategy,
    )


def reservation_offset(market, dealer, q, tau):
    """How far the quote centre sits from the mid, for a given inventory."""
    return reservation_price(0.0, q, dealer.gam, market.sigma, tau)
