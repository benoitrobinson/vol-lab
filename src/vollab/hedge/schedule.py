"""Trade-decision rules.

Every schedule carries the same signature so the simulator calls them
identically. The condition is evaluated at every monitoring step; trading
happens only on the subset where the mask is true. Band schedules therefore
reduce trading cost, not compute.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedTime:
    every: int

    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        on = (step % self.every == 0) or step == n_mon
        return np.full(delta_target.shape, on, dtype=bool)

    def hedge_vol(self, s_hedge, k, dt):
        return s_hedge


@dataclass(frozen=True)
class DeltaBand:
    h: float

    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        if step == 0 or step == n_mon:
            return np.ones(delta_target.shape, dtype=bool)
        return np.abs(delta_target - delta_held) > self.h

    def hedge_vol(self, s_hedge, k, dt):
        return s_hedge


@dataclass(frozen=True)
class Leland:
    """Fixed-time hedging at a cost-adjusted volatility.

    The sign of the adjustment depends on the side: a written option needs a
    cost buffer (wider effective vol), a held one can run at a reduced one.
    """

    every: int
    short: bool = True

    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        on = (step % self.every == 0) or step == n_mon
        return np.full(delta_target.shape, on, dtype=bool)

    def hedge_vol(self, s_hedge, k, dt):
        le = np.sqrt(2.0 / np.pi) * k / (s_hedge * np.sqrt(dt))
        sign = 1.0 if self.short else -1.0
        return float(s_hedge * np.sqrt(max(1.0 + sign * le, 1e-12)))


@dataclass(frozen=True)
class WhalleyWilmott:
    """Asymptotic no-trade band under proportional costs.

    Half-width goes as (k * S * Gamma^2 / gam_ra)^(1/3). S enters to the first
    power: Delta is dimensionless so Gamma is 1/currency, and with risk aversion
    also 1/currency the bracket is dimensionless, as a delta band must be.
    """

    gam_ra: float

    def band_width(self, S, gamma, k):
        return np.cbrt(1.5 * k * S * gamma * gamma / self.gam_ra)

    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        if step == 0 or step == n_mon:
            return np.ones(delta_target.shape, dtype=bool)
        return np.abs(delta_target - delta_held) > self.band_width(S, gamma, self._k)

    def hedge_vol(self, s_hedge, k, dt):
        # The simulator calls this once before the step loop, which is how the
        # band gains access to k without widening the shared should_trade signature.
        object.__setattr__(self, "_k", k)
        return s_hedge
