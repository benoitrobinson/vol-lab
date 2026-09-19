import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import simulate, sweep
from vollab.metrics.stats import loglog_slope, sd_vs_frequency

EVERY = [256, 128, 64, 32, 16, 8, 4, 2, 1]   # N_reh 8 .. 2048


def _cfg(**kw):
    base = dict(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1),
        n_mon=2048, cost_bps=0.0, n_paths=8_000, seed=11, chunk_paths=4_000,
    )
    base.update(kw)
    return HedgeConfig(**base)


@pytest.fixture(scope="module")
def curve():
    """One sweep shared by every assertion about it; it is the slow fixture."""
    return sd_vs_frequency(_cfg(), EVERY)


def test_loglog_slope_on_exact_power_law():
    x = np.array([1.0, 10.0, 100.0])
    assert abs(loglog_slope(x, 3.0 * x ** -0.5) - (-0.5)) < 1e-12


def test_sd_is_monotone_decreasing_in_frequency(curve):
    _, sd = curve
    assert np.all(np.diff(sd) < 0)


def test_sd_scales_as_inverse_sqrt_n(curve):
    """Boyle and Emanuel (1980). The engine's primary correctness test."""
    n, sd = curve
    assert abs(loglog_slope(n, sd) - (-0.5)) < 0.03


def test_realized_rehedge_counts_track_the_requested_grid(curve):
    """At or just below nominal: a step where the delta is unchanged to the last
    bit books no trade, which happens where delta saturates at 0 or 1."""
    n, _ = curve
    nominal = np.array([2048 / e + 1 for e in EVERY])
    assert np.all(n <= nominal)
    assert np.all(n > 0.9 * nominal)


def test_sweep_matches_individual_simulate_calls():
    """The sweep must be a pure speedup, not a different computation."""
    import dataclasses

    c = _cfg(n_mon=64, n_paths=1000, chunk_paths=256)
    scheds = [FixedTime(1), FixedTime(4), FixedTime(16)]
    for s, got in zip(scheds, sweep(c, scheds)):
        one = simulate(dataclasses.replace(c, schedule=s))
        assert np.array_equal(got.pnl, one.pnl)
        assert np.array_equal(got.n_rehedges, one.n_rehedges)
