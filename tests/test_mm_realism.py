"""The two frictions the idealised market-making model leaves out.

Avellaneda-Stoikov marks terminal inventory at the mid and draws fills that know
nothing about the next price move. Both are flattering: the first hands the
dealer a free unwind, and the second removes adverse selection by construction.
These tests gate the two knobs that put them back, and check that switching them
off reproduces the original model exactly.
"""

import numpy as np
import pytest

from vollab.mm.quoting import DealerParams, MarketParams
from vollab.mm.simulate import simulate_mm

SEED = 4242
PATHS = 4000


def _run(phi=0.0, liq_cost=0.0, liq_impact=0.0, strategy="glft", markout_steps=10):
    return simulate_mm(
        MarketParams(phi=phi, markout_steps=markout_steps),
        DealerParams(liq_cost=liq_cost, liq_impact=liq_impact),
        strategy, seed=SEED, n_paths=PATHS,
    )


def test_a_free_unwind_leaves_the_pnl_untouched():
    r = _run()
    assert np.array_equal(r.pnl, r.pnl_gross)
    assert not r.liq_paid.any()


def test_the_unwind_cost_is_exactly_what_it_says():
    """Linear in the inventory crossed, quadratic in the impact term. If this
    drifts, every number downstream is a different experiment."""
    r = _run(liq_cost=0.5, liq_impact=0.005)
    q = r.inventory_end.astype(float)
    assert np.allclose(r.liq_paid, np.abs(q) * 0.5 + q ** 2 * 0.005)
    assert np.allclose(r.pnl, r.pnl_gross - r.liq_paid)


def test_the_unwind_cost_falls_hardest_on_the_strategy_that_carries_inventory():
    skew = _run(liq_cost=0.5, liq_impact=0.005, strategy="glft")
    control = _run(liq_cost=0.5, liq_impact=0.005, strategy="symmetric")
    assert np.abs(control.inventory_end).mean() > 3 * np.abs(skew.inventory_end).mean()
    assert control.liq_paid.mean() > 3 * skew.liq_paid.mean()


def test_uninformed_flow_marks_out_at_zero():
    """The dealer's fills carry no information in the base model, so the mid
    goes nowhere in particular afterwards. This is the control for the test
    below, and it is also what makes the base model flattering."""
    assert abs(_run().markout_per_fill()) < 0.01


def test_informed_flow_marks_out_against_the_dealer():
    assert _run(phi=0.3).markout_per_fill() < -0.02


def test_adverse_selection_scales_with_the_informed_fraction():
    m03 = _run(phi=0.3).markout_per_fill()
    m06 = _run(phi=0.6).markout_per_fill()
    assert m06 < m03 < 0.0
    # Linear in phi by construction, so doubling the fraction doubles the cost.
    assert abs(m06 / m03 - 2.0) < 0.15


def test_informed_flow_does_not_change_how_often_the_dealer_trades():
    """What changes is which side arrives, not how much flow there is. If the
    fill count moved, the experiment would be confounded by volume."""
    base = _run().n_fills.mean()
    informed = _run(phi=0.3).n_fills.mean()
    assert abs(informed / base - 1.0) < 0.02


def test_the_markout_is_information_not_drift():
    """The tilt acts on the next step only, so extending the horizon adds noise
    but no further loss. A markout that kept growing with the horizon would be
    a drift in the mid, which would be a broken simulator."""
    short = _run(phi=0.3, markout_steps=1).markout_per_fill()
    long = _run(phi=0.3, markout_steps=20).markout_per_fill()
    assert abs(long / short - 1.0) < 0.25


def test_informed_flow_costs_the_dealer_money():
    assert _run(phi=0.3).pnl.mean() < _run().pnl.mean()


@pytest.mark.parametrize("strategy", ["symmetric", "avellaneda_stoikov", "glft"])
def test_every_strategy_reconciles_gross_to_net(strategy):
    r = _run(phi=0.3, liq_cost=0.5, liq_impact=0.005, strategy=strategy)
    assert np.allclose(r.pnl, r.pnl_gross - r.liq_paid)
    assert r.markout_per_fill() < 0.0
