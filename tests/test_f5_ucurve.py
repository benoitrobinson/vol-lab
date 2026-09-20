import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.metrics.stats import cost_curve

EVERY = [128, 64, 32, 16, 8, 4, 2, 1]        # ascending rehedge frequency


def _cfg(cost_bps):
    return HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1), n_mon=512,
        cost_bps=cost_bps, n_paths=6000, seed=31, chunk_paths=3000,
    )


@pytest.fixture(scope="module")
def curves():
    return {bps: cost_curve(_cfg(bps), EVERY)[1] for bps in (0.0, 5.0, 60.0)}


def test_zero_cost_curve_is_monotone_decreasing(curves):
    """With no friction, hedge as often as you can."""
    assert np.all(np.diff(curves[0.0]) < 0)


def test_cost_curve_has_an_interior_minimum(curves):
    c = curves[60.0]
    i = int(np.argmin(c))
    assert 0 < i < len(c) - 1, f"minimum at edge index {i}: {c}"


def test_optimum_moves_to_less_frequent_hedging_as_costs_rise(curves):
    """EVERY is ascending in frequency, so a lower index is less frequent."""
    assert int(np.argmin(curves[60.0])) < int(np.argmin(curves[5.0]))


def test_costs_shift_the_whole_curve_up(curves):
    assert np.all(curves[60.0] > curves[0.0])
