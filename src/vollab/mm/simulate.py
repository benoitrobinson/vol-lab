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
    glft_half_spreads, optimal_half_spreads, reservation_price,
    symmetric_half_spreads,
)
from vollab.rng.scheme import JUMP_STREAM_SALT, normals_block, uniforms_block

def _glft(q, gam, sigma, tau, kappa):
    """GLFT ignores tau by construction; the signature is shared with the others."""
    return glft_half_spreads(q, gam, sigma, tau, kappa)


STRATEGIES = {"avellaneda_stoikov": optimal_half_spreads,
              "symmetric": symmetric_half_spreads,
              "glft": _glft}


@dataclass(frozen=True)
class MMResult:
    pnl: np.ndarray            # (n_paths,) terminal wealth, after unwinding
    inventory_end: np.ndarray  # (n_paths,)
    inventory_max_abs: np.ndarray
    n_fills: np.ndarray
    spread_captured: np.ndarray
    strategy: str
    # Terminal inventory marked at the mid, before paying to unwind it. The two
    # always reconcile: pnl = pnl_gross - liq_paid.
    pnl_gross: np.ndarray = None
    liq_paid: np.ndarray = None
    # Summed over fills: the mid at the markout horizon against the fill price,
    # signed so a positive number means the market moved the dealer's way.
    # Negative is adverse selection.
    markout: np.ndarray = None

    def markout_per_fill(self):
        """Mean markout per fill, averaged over the paths that traded."""
        traded = self.n_fills > 0
        if not traded.any():
            return 0.0
        return float((self.markout[traded] / self.n_fills[traded]).mean())


def simulate_mm(market, dealer, strategy, seed, n_paths, path_start=0):
    """Run one quoting strategy over n_paths independent sessions."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {sorted(STRATEGIES)}")
    half_spreads = STRATEGIES[strategy]

    n, dt = market.n_steps, market.T / market.n_steps
    sq = np.sqrt(dt)
    phi = getattr(market, "phi", 0.0)
    h = getattr(market, "markout_steps", 10)

    dz = normals_block(seed, path_start, n_paths, n)
    # Fill draws come from a salted stream so they are disjoint from the mid.
    u = uniforms_block(seed ^ JUMP_STREAM_SALT, path_start, n_paths, 2 * n)

    # The whole mid path up front: the markout needs the future mid, and the
    # informed flow below needs the step the mid is about to take. Neither is
    # visible to the quoting rule, which only ever sees s[:, i].
    mid = np.empty((n_paths, n + 1), dtype=np.float64)
    mid[:, 0] = float(market.s0)
    np.cumsum(market.sigma * sq * dz, axis=1, out=mid[:, 1:])
    mid[:, 1:] += float(market.s0)

    q = np.zeros(n_paths, dtype=np.int64)
    cash = np.zeros(n_paths)
    inv_max = np.zeros(n_paths, dtype=np.int64)
    fills = np.zeros(n_paths, dtype=np.int64)
    captured = np.zeros(n_paths)
    markout = np.zeros(n_paths)

    for i in range(n):
        tau = market.T - i * dt
        s = mid[:, i]
        d_a, d_b = half_spreads(q, dealer.gam, market.sigma, tau, market.kappa)

        # Half-spreads are not floored at zero. A heavily long dealer quotes an
        # ask through the mid to unwind, which is the model's behaviour, not an
        # artefact; the fill probability is what gets capped.
        p_a = np.clip(market.A * np.exp(-market.kappa * d_a) * dt, 0.0, 1.0)
        p_b = np.clip(market.A * np.exp(-market.kappa * d_b) * dt, 0.0, 1.0)

        if phi:
            # Informed flow. A fraction phi of the arrivals know the next move,
            # so buyers lift the ask more often when the mid is about to rise
            # and sellers hit the bid more often when it is about to fall. The
            # unconditional arrival rate is unchanged: what changes is which
            # side arrives, which is exactly what adverse selection means.
            up = np.sign(dz[:, i])
            p_a = np.clip(p_a * (1.0 + phi * up), 0.0, 1.0)
            p_b = np.clip(p_b * (1.0 - phi * up), 0.0, 1.0)

        # A dealer at its position limit stops quoting the side that would breach it.
        can_sell = q > -dealer.max_inventory
        can_buy = q < dealer.max_inventory

        hit_ask = (u[:, 2 * i] < p_a) & can_sell
        hit_bid = (u[:, 2 * i + 1] < p_b) & can_buy

        cash += np.where(hit_ask, s + d_a, 0.0)
        cash -= np.where(hit_bid, s - d_b, 0.0)
        captured += np.where(hit_ask, d_a, 0.0) + np.where(hit_bid, d_b, 0.0)

        # Markout against the mid at the time of the fill, not against the fill
        # price: the half-spread is already counted in `captured`, and mixing
        # the two hides the thing worth measuring. Sold, so the dealer wants
        # the mid lower h steps later; bought, so it wants it higher.
        future = mid[:, min(i + h, n)]
        markout += np.where(hit_ask, s - future, 0.0)
        markout += np.where(hit_bid, future - s, 0.0)

        q = q + hit_bid.astype(np.int64) - hit_ask.astype(np.int64)
        fills += hit_ask.astype(np.int64) + hit_bid.astype(np.int64)
        inv_max = np.maximum(inv_max, np.abs(q))

    gross = cash + q * mid[:, n]
    liq = (np.abs(q) * getattr(dealer, "liq_cost", 0.0)
           + q.astype(np.float64) ** 2 * getattr(dealer, "liq_impact", 0.0))

    return MMResult(
        pnl=gross - liq,
        inventory_end=q,
        inventory_max_abs=inv_max,
        n_fills=fills,
        spread_captured=captured,
        strategy=strategy,
        pnl_gross=gross,
        liq_paid=liq,
        markout=markout,
    )
