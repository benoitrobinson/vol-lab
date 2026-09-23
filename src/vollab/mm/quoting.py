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
    # Adverse selection. A fraction of the flow is informed: it arrives on the
    # side the mid is about to move towards, so the dealer's fills are worse
    # than a coin flip. phi = 0 is the Avellaneda-Stoikov idealisation, where
    # fills carry no information and markouts are zero by construction.
    phi: float = 0.0
    # Horizon in steps over which a fill is marked out.
    markout_steps: int = 10


@dataclass(frozen=True)
class DealerParams:
    gam: float = 0.1          # inventory risk aversion
    max_inventory: int = 50   # hard position limit; quotes are pulled at the cap
    # What it costs to go home flat. Marking terminal inventory at the mid
    # assumes the dealer can unwind for free, which flatters whichever strategy
    # carries the most inventory. liq_cost is paid per unit crossed, liq_impact
    # per unit squared.
    liq_cost: float = 0.0
    liq_impact: float = 0.0


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


def glft_half_spreads(q, gam, sigma, tau, kappa, A=None):
    """Gueant, Lehalle and Fernandez-Tapia (2013) closed-form quotes.

    Avellaneda-Stoikov solves a control problem whose exact solution is a system
    of ODEs; the widely used formulas are its small-inventory expansion. GLFT
    show that under exponential fill intensities the system linearises and admits
    a closed form, and the resulting quotes differ from AS in two ways that
    matter on a real book.

    The half-spread is

        d = 1/gam * ln(1 + gam/kappa)  +  (2q + 1)/2 * sqrt(
                sigma^2 * gam / (2 * kappa * A) * (1 + gam/kappa)^(1 + kappa/gam))

    The first term is the same spread AS charges for the fill trade-off. The
    second is the inventory term, and unlike AS it does not carry `tau`: the
    steady-state solution does not widen as the horizon approaches, because a
    dealer who keeps quoting has no horizon. AS quotes collapse toward the mid
    as `tau -> 0`, which is an artefact of the finite-horizon formulation rather
    than desk behaviour.

    The `(2q + 1)/2` asymmetry is the second difference: the inventory
    adjustment is not symmetric in `q`, so a flat dealer already quotes a
    fractionally skewed market.
    """
    if A is None:
        A = 140.0
    base = np.log1p(gam / kappa) / gam
    inner = (sigma ** 2 * gam / (2.0 * kappa * A)
             * (1.0 + gam / kappa) ** (1.0 + kappa / gam))
    scale = np.sqrt(max(inner, 0.0))
    q = np.asarray(q, dtype=float)
    # A long dealer quotes a closer ask and a further bid, so the inventory term
    # is subtracted from the ask and added to the bid. Reversing these two lines
    # produces a book that amplifies inventory instead of damping it; that is
    # the error `test_every_strategy_leans_against_inventory` exists to catch,
    # and it has caught it twice in this module.
    d_a = base - (2.0 * q - 1.0) / 2.0 * scale
    d_b = base + (2.0 * q + 1.0) / 2.0 * scale
    return d_a, d_b
