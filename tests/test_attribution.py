import numpy as np

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import simulate

FIELDS = ("delta", "gamma", "theta", "vega", "carry", "cost",
          "residual_sum", "residual_abs_sum", "residual_max_abs")


def cfg(**kw):
    base = dict(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1),
        n_mon=512, cost_bps=0.0, n_paths=4000, seed=5, chunk_paths=1000,
    )
    base.update(kw)
    return HedgeConfig(**base)


def test_components_have_path_shape():
    a = simulate(cfg()).attribution
    for name in FIELDS:
        assert getattr(a, name).shape == (4000,)


def test_vega_is_identically_zero_in_phase_a():
    assert np.all(simulate(cfg()).attribution.vega == 0.0)


def test_closure_holds_as_a_nan_guard():
    """True by construction of the residual, so this only guards NaN and shape."""
    r = simulate(cfg())
    assert np.allclose(r.attribution.total(), r.pnl, rtol=0, atol=1e-8)


def _residual_ratio(n_mon):
    """Summed absolute residual against summed absolute gamma P&L.

    Averaged over paths, not a single worst step: the per-step maximum is
    dominated by one tail path and does not scale cleanly.
    """
    a = simulate(cfg(n_mon=n_mon, n_paths=2000)).attribution
    return a.residual_abs_sum.mean() / np.abs(a.gamma).mean()


def test_residual_is_small_relative_to_gamma_under_gbm():
    """The expansion is exact to second order, so the residual is third order
    and must be negligible next to the gamma term."""
    assert _residual_ratio(512) < 0.05


def test_residual_scales_as_sqrt_dt():
    """The test with real power. Third-order terms vanish as the grid refines;
    a first-order bug (a mislabelled carry or an unattributed cost) would leave
    a residual that does not shrink. Measured: 0.069, 0.035, 0.017 for n_mon
    128, 512, 2048, which is a factor of 2 per 4x refinement."""
    ratios = [_residual_ratio(n) for n in (128, 512, 2048)]
    assert ratios[1] < 0.6 * ratios[0]
    assert ratios[2] < 0.6 * ratios[1]


def test_cost_term_captures_transaction_costs():
    """Costs must land in their own line, never in the residual."""
    free = simulate(cfg(cost_bps=0.0)).attribution
    paid = simulate(cfg(cost_bps=20.0)).attribution
    assert np.all(free.cost == 0.0)
    assert paid.cost.mean() < 0.0
    moved = abs(paid.residual_abs_sum.mean() - free.residual_abs_sum.mean())
    assert moved < 0.1 * abs(paid.cost.mean())


def test_gamma_is_negative_for_a_short_call():
    """Position greeks: short convexity loses on large moves."""
    assert simulate(cfg()).attribution.gamma.mean() < 0.0


def test_theta_is_positive_for_a_short_call():
    assert simulate(cfg()).attribution.theta.mean() > 0.0
