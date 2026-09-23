"""F13: the smile's slope does not tell you how to hedge it.

F9 showed that a delta hedge leaves a vega floor under stochastic volatility.
Part of that vega move is predictable from the spot move, because the two are
correlated, so carrying `Vega * d(vol)/dS` of extra stock should remove part of
the floor. The question is what to put in for `d(vol)/dS`.

The tempting answer is to read it off the smile. Implied volatility is a
function of log-moneyness, so if `sigma` depends on `ln(K/S)` then
`d sigma/dS = -(d sigma/dk)/S`, and a downward-sloping smile gives a positive
number. That reasoning assumes the smile is glued to moneyness and slides with
spot, which is an assumption about dynamics, not something the smile's shape
can tell you. Under rough Bergomi with negative correlation the true dynamics
are the other way: spot up, variance down.

Measured on 8,000 paths, a 256-step grid, three seeds, H = 0.10, eta = 1.5,
rho = -0.7:

    plain Black-Scholes delta        rough 3.7063 +/- 0.1215   gbm 0.6553
    minimum variance, slope -0.0015  rough 3.5583 +/- 0.1273   gbm 1.2661
    sticky-delta sign, slope +0.0021 rough 4.2980 +/- 0.1055   gbm 1.6376

so the minimum-variance adjustment cuts the hedging error by 4.0%, the sign the
smile's slope suggests raises it by 16.0%, and under GBM, where there is no
volatility risk to hedge, every adjustment is pure noise.
"""

import numpy as np
import pytest

# Research sweep: see the slow marker in pyproject.toml.
pytestmark = pytest.mark.slow

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import simulate
from vollab.paths.base import GBM, RoughBergomi

SEEDS = [11, 23]
ROUGH = RoughBergomi(H=0.10, eta=1.5, rho=-0.7)
# The optimum of a sweep run once, before this test existed. It is a parameter
# of the experiment, not something fitted inside it.
MV_OPTIMUM = -0.0015
# What the smile's slope suggests if implied vol is assumed to slide with spot.
STICKY = 0.0021


def _sd(model, mv, seed, n_paths=4000):
    cfg = HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1), model=model,
        n_mon=256, cost_bps=0.0, n_paths=n_paths, seed=seed, chunk_paths=2000,
        attribute=False, mv_slope=mv,
    )
    return simulate(cfg).pnl.std(ddof=1)


def _mean_sd(model, mv):
    return float(np.mean([_sd(model, mv, s) for s in SEEDS]))


@pytest.fixture(scope="module")
def table():
    return {
        ("rough", mv): _mean_sd(ROUGH, mv) for mv in (0.0, MV_OPTIMUM, STICKY)
    } | {("gbm", mv): _mean_sd(GBM(), mv) for mv in (0.0, MV_OPTIMUM, STICKY)}


def test_a_zero_slope_changes_nothing(table):
    """The control that matters most: every finding before this one ran with
    the plain delta, and they have to be reproduced exactly."""
    plain = _sd(GBM(), 0.0, SEEDS[0])
    cfg = HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1), model=GBM(),
        n_mon=256, cost_bps=0.0, n_paths=4000, seed=SEEDS[0], chunk_paths=2000,
        attribute=False,
    )
    assert simulate(cfg).pnl.std(ddof=1) == plain


def test_the_minimum_variance_delta_lowers_the_floor(table):
    assert table[("rough", MV_OPTIMUM)] < table[("rough", 0.0)]
    cut = 1.0 - table[("rough", MV_OPTIMUM)] / table[("rough", 0.0)]
    assert cut > 0.02, f"only cut {cut:.1%}"


def test_the_sign_the_smile_suggests_makes_it_worse(table):
    """The trap. A downward-sloping smile does not mean implied vol rises when
    spot rises; it means options struck lower are dearer. Confusing the two
    reverses the hedge adjustment."""
    assert table[("rough", STICKY)] > table[("rough", 0.0)]
    worse = table[("rough", STICKY)] / table[("rough", 0.0)] - 1.0
    assert worse > 0.05, f"only worse by {worse:.1%}"


def test_without_volatility_risk_any_adjustment_is_noise(table):
    """Under GBM the implied volatility never moves, so there is nothing for
    the extra stock to hedge and it can only add variance."""
    for mv in (MV_OPTIMUM, STICKY):
        assert table[("gbm", mv)] > table[("gbm", 0.0)]


def test_the_adjustment_is_second_order_next_to_the_floor(table):
    """Worth stating plainly: this removes a few percent, not the floor. The
    rest of the vega risk needs an option, not more stock."""
    assert table[("rough", MV_OPTIMUM)] > 0.5 * table[("rough", 0.0)]
