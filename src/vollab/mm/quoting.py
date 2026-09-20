"""Avellaneda-Stoikov market making.

A dealer quotes two sides around a mid price, earns the spread when filled, and
carries the inventory risk of whatever it is left holding. The model's answer is
that the quotes should be centred not on the mid but on a reservation price that
leans against inventory:

    r(s, q, t) = s - q * gam * sigma^2 * (T - t)

and that the total spread should be

    d_a + d_b = gam * sigma^2 * (T - t) + (2 / gam) * ln(1 + gam / kappa)

The first term prices the risk of holding inventory to the horizon; the second
prices the trade-off between quoting tight enough to get filled and wide enough
to be paid. Fills arrive as a Poisson process whose intensity decays with
distance from the mid, lam(d) = A * exp(-kappa * d).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketParams:
    sigma: float = 2.0        # absolute vol of the mid, per unit time
    A: float = 140.0          # arrival intensity scale
    kappa: float = 1.5        # intensity decay with quote distance
    T: float = 1.0
    n_steps: int = 200
    s0: float = 100.0


@dataclass(frozen=True)
class DealerParams:
    gam: float = 0.1          # inventory risk aversion
    max_inventory: int = 50   # hard position limit; quotes are pulled at the cap


def reservation_price(s, q, gam, sigma, tau):
    return s - q * gam * sigma ** 2 * tau


def optimal_half_spreads(q, gam, sigma, tau, kappa):
    """Half-spreads to the ask and the bid, measured from the mid.

    Quotes are centred on the reservation price, so relative to the mid the ask
    sits at (r - s) + total/2 and the bid at (s - r) + total/2. With
    r - s = -q*gam*sigma^2*tau, a long dealer quotes a *closer* ask and a
    *further* bid, which is what makes inventory mean-revert. Getting this sign
    backwards amplifies inventory instead of damping it, and the symmetric
    control in this module is what exposes that.
    """
    total = gam * sigma ** 2 * tau + (2.0 / gam) * np.log1p(gam / kappa)
    skew = q * gam * sigma ** 2 * tau
    return 0.5 * total - skew, 0.5 * total + skew


def symmetric_half_spreads(q, gam, sigma, tau, kappa):
    """The naive control: the same total width, never skewed for inventory.

    Isolating the skew is the point. Quoting the same width both sides means any
    difference in outcome is attributable to leaning against inventory rather
    than to quoting wider or tighter overall.
    """
    total = gam * sigma ** 2 * tau + (2.0 / gam) * np.log1p(gam / kappa)
    return 0.5 * total, 0.5 * total
