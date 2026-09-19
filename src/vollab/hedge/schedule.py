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
