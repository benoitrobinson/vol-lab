import numpy as np
import pytest

from vollab.mm.quoting import (
    DealerParams, MarketParams, optimal_half_spreads, reservation_price,
    symmetric_half_spreads,
)
from vollab.mm.simulate import simulate_mm

M, D = MarketParams(), DealerParams()


def test_reservation_price_is_the_mid_when_flat():
    assert reservation_price(100.0, 0, D.gam, M.sigma, 1.0) == 100.0


def test_reservation_price_leans_against_inventory():
    assert reservation_price(100.0, 10, D.gam, M.sigma, 1.0) < 100.0
    assert reservation_price(100.0, -10, D.gam, M.sigma, 1.0) > 100.0


def test_reservation_price_returns_to_the_mid_at_the_horizon():
    """Named for what it tests. The previous name referred to a function it
    never called, which is worse than no test at all."""
    assert reservation_price(100.0, 20, D.gam, M.sigma, 0.0) == 100.0


def test_quotes_are_symmetric_when_flat():
    a, b = optimal_half_spreads(0, D.gam, M.sigma, 1.0, M.kappa)
    assert a == pytest.approx(b)


def test_long_inventory_pulls_the_ask_closer_than_the_bid():
    """The sign that makes inventory mean-revert. Inverted, it amplifies it."""
    a, b = optimal_half_spreads(10, D.gam, M.sigma, 1.0, M.kappa)
    assert a < b


def test_short_inventory_pulls_the_bid_closer_than_the_ask():
    a, b = optimal_half_spreads(-10, D.gam, M.sigma, 1.0, M.kappa)
    assert b < a


def test_total_width_is_unchanged_by_inventory():
    """The skew moves the centre; it does not widen the quote."""
    flat = sum(optimal_half_spreads(0, D.gam, M.sigma, 1.0, M.kappa))
    long = sum(optimal_half_spreads(25, D.gam, M.sigma, 1.0, M.kappa))
    assert flat == pytest.approx(long)


def test_symmetric_control_never_skews():
    for q in (-20, 0, 20):
        a, b = symmetric_half_spreads(q, D.gam, M.sigma, 1.0, M.kappa)
        assert a == pytest.approx(b)


def test_control_matches_the_model_width_exactly():
    """Any difference in outcome is the skew, not the width."""
    assert sum(symmetric_half_spreads(0, D.gam, M.sigma, 1.0, M.kappa)) == pytest.approx(
        sum(optimal_half_spreads(0, D.gam, M.sigma, 1.0, M.kappa)))


def test_wider_quotes_when_more_risk_averse():
    lo = sum(optimal_half_spreads(0, 0.05, M.sigma, 1.0, M.kappa))
    hi = sum(optimal_half_spreads(0, 0.5, M.sigma, 1.0, M.kappa))
    assert hi > lo


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        simulate_mm(M, D, "gut_feel", seed=1, n_paths=10)


@pytest.fixture(scope="module")
def runs():
    return {s: simulate_mm(M, D, s, seed=5, n_paths=4000)
            for s in ("avellaneda_stoikov", "symmetric")}


def test_inventory_stays_within_the_position_limit(runs):
    for r in runs.values():
        assert r.inventory_max_abs.max() <= D.max_inventory


def test_skew_controls_inventory(runs):
    """The headline: leaning against inventory keeps the book far smaller."""
    a, s = runs["avellaneda_stoikov"], runs["symmetric"]
    assert a.inventory_max_abs.mean() < 0.6 * s.inventory_max_abs.mean()
    assert a.inventory_end.std(ddof=1) < 0.5 * s.inventory_end.std(ddof=1)


def test_skew_halves_pnl_dispersion(runs):
    a, s = runs["avellaneda_stoikov"], runs["symmetric"]
    assert a.pnl.std(ddof=1) < 0.6 * s.pnl.std(ddof=1)


def test_skew_costs_some_expected_pnl(runs):
    """Nothing is free: the inventory control is paid for in mean P&L."""
    from vollab.metrics.bootstrap import paired_bootstrap

    a, s = runs["avellaneda_stoikov"], runs["symmetric"]
    diff, lo, hi = paired_bootstrap(a.pnl, s.pnl, n_boot=1000)
    assert hi < 0


def test_skew_improves_risk_adjusted_return(runs):
    a, s = runs["avellaneda_stoikov"], runs["symmetric"]
    assert (a.pnl.mean() / a.pnl.std(ddof=1)) > 1.5 * (s.pnl.mean() / s.pnl.std(ddof=1))


def test_both_strategies_capture_spread(runs):
    for r in runs.values():
        assert r.spread_captured.mean() > 0
        assert r.n_fills.mean() > 10


def test_paths_are_shared_between_strategies():
    """Same seed means the same mid path, so comparisons are paired."""
    a = simulate_mm(M, D, "avellaneda_stoikov", seed=9, n_paths=100)
    b = simulate_mm(M, D, "avellaneda_stoikov", seed=9, n_paths=100)
    assert np.array_equal(a.pnl, b.pnl)
