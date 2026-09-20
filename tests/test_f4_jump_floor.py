"""F4: delta hedging cannot remove gap risk.

Between jumps the diffusive error hedges away as N^-0.5, exactly as in F1. Each
Poisson jump delivers an unhedgeable convexity P&L that no rehedge frequency can
pre-empt, with a variance set by lam, the jump law and T, none of which depend
on N. Total variance therefore converges to a floor instead of to zero.

Measured on 6,000 paths, a 1,024-step grid:
    GBM     sd 3.489 -> 0.319   ratio 0.091   slope -0.503
    Merton  sd 4.813 -> 3.084   ratio 0.641   slope -0.088
"""

import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import simulate
from vollab.metrics.stats import sd_vs_frequency
from vollab.paths.base import GBM, Merton

EVERY = [128, 64, 32, 16, 8, 4, 2, 1]
JUMPS = Merton(lam=1.0, mu_J=-0.10, s_J=0.15)


def _cfg(model, **kw):
    base = dict(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1), model=model,
        n_mon=1024, cost_bps=0.0, n_paths=6000, seed=77, chunk_paths=3000,
    )
    base.update(kw)
    return HedgeConfig(**base)


@pytest.fixture(scope="module")
def curves():
    return {"gbm": sd_vs_frequency(_cfg(GBM()), EVERY),
            "merton": sd_vs_frequency(_cfg(JUMPS), EVERY)}


def test_gbm_control_still_recovers_the_law(curves):
    """The control. If this breaks, the comparison below means nothing."""
    n, sd = curves["gbm"]
    assert abs(np.polyfit(np.log(n), np.log(sd), 1)[0] - (-0.5)) < 0.05


def test_jump_dispersion_barely_falls_with_frequency(curves):
    _, sd = curves["merton"]
    assert sd[-1] / sd[0] > 0.5


def test_gbm_dispersion_falls_by_an_order_of_magnitude(curves):
    _, sd = curves["gbm"]
    assert sd[-1] / sd[0] < 0.15


def test_jump_slope_is_far_from_minus_one_half(curves):
    n, sd = curves["merton"]
    assert np.polyfit(np.log(n), np.log(sd), 1)[0] > -0.2


def test_jump_curve_plateaus_at_the_dense_end(curves):
    """The floor itself: doubling frequency three more times buys almost nothing."""
    _, sd = curves["merton"]
    assert sd[-1] > 0.9 * sd[-4]


def test_hedged_jump_book_has_a_fat_left_tail():
    pnl = simulate(_cfg(JUMPS, n_mon=512, n_paths=8000)).pnl
    z = (pnl - pnl.mean()) / pnl.std(ddof=1)
    assert z.min() < -5.0
    from scipy.stats import skew
    assert skew(pnl) < -1.0


def test_attribution_residual_is_large_under_jumps():
    """A jump is not small, so the second-order expansion cannot absorb it.

    The discriminating statistic is the worst single step, not the sum: a jump
    concentrates its unhedgeable P&L into one step rather than spreading it
    across the path. Measured worst-step ratios are 0.0031 under GBM against
    0.0655 under Merton, a factor of 21. The summed ratio only separates by
    2.7x because diffusive residuals accumulate over every step.
    """
    a = simulate(_cfg(JUMPS, n_mon=512, n_paths=4000)).attribution
    g = simulate(_cfg(GBM(), n_mon=512, n_paths=4000)).attribution
    jump_worst = a.residual_max_abs.mean() / np.abs(a.gamma).mean()
    gbm_worst = g.residual_max_abs.mean() / np.abs(g.gamma).mean()
    assert jump_worst > 10 * gbm_worst

    jump_sum = a.residual_abs_sum.mean() / np.abs(a.gamma).mean()
    gbm_sum = g.residual_abs_sum.mean() / np.abs(g.gamma).mean()
    assert jump_sum > 2 * gbm_sum
