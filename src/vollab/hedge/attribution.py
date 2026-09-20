"""Per-step P&L explain, summed into per-path arrays.

The decomposition is the one a desk produces: a second-order Taylor expansion of
the position value plus financing and costs. Under GBM the residual is third
order and negligible; a jump is not small, so under a jump model the residual is
large. Its magnitude is therefore a direct measurement of the risk delta hedging
cannot see, which is why costs get their own line rather than being absorbed
into it.

Mark greeks are evaluated at s_imp, because the short option is marked there.
The traded hedge is sized at s_hedge. They are different numbers whenever the
two volatilities differ, which is the F2 configuration.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Attribution:
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    carry: np.ndarray
    cost: np.ndarray
    residual_sum: np.ndarray
    residual_abs_sum: np.ndarray
    residual_max_abs: np.ndarray

    def total(self):
        return (self.delta + self.gamma + self.theta + self.vega
                + self.carry + self.cost + self.residual_sum)


class Accumulator:
    """Sums per-step terms into per-path totals.

    Three residual statistics are kept, not one: a signed sum alone cancels
    two-signed jump residuals across steps, which would destroy exactly the
    measurement the residual exists to provide.
    """

    def __init__(self, m):
        self.delta = np.zeros(m)
        self.gamma = np.zeros(m)
        self.theta = np.zeros(m)
        self.carry = np.zeros(m)
        self.cost = np.zeros(m)
        self.res_sum = np.zeros(m)
        self.res_abs = np.zeros(m)
        self.res_max = np.zeros(m)

    def step(self, delta_mark, gamma_mark, theta_mark, dS, dt,
             cash_prev, r, q, held_prev, S_prev, cost_paid, realized):
        d = delta_mark * dS
        g = 0.5 * gamma_mark * dS * dS
        th = theta_mark * dt
        carry = cash_prev * (np.expm1(r * dt)) + q * held_prev * S_prev * dt
        res = realized - (d + g + th + carry + cost_paid)

        self.delta += d
        self.gamma += g
        self.theta += th
        self.carry += carry
        self.cost += cost_paid
        self.res_sum += res
        self.res_abs += np.abs(res)
        self.res_max = np.maximum(self.res_max, np.abs(res))

    def finish(self):
        return Attribution(
            delta=self.delta, gamma=self.gamma, theta=self.theta,
            vega=np.zeros_like(self.delta),   # implied vol is held fixed in Phase A
            carry=self.carry, cost=self.cost,
            residual_sum=self.res_sum, residual_abs_sum=self.res_abs,
            residual_max_abs=self.res_max,
        )


def concat(parts):
    f = {}
    for name in ("delta", "gamma", "theta", "vega", "carry", "cost",
                 "residual_sum", "residual_abs_sum", "residual_max_abs"):
        f[name] = np.concatenate([getattr(p, name) for p in parts])
    return Attribution(**f)
