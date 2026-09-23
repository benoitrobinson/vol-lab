"""F9: vol-of-vol floors the hedging error, and roughness is not what does it.

F1 measured the square-root law: a delta hedge on a constant-volatility path
loses error as N^-0.5. F4 showed jumps floor that law, because a gap is not a
diffusion. Stochastic volatility floors it for a different reason: the delta
hedge removes spot risk, and nothing in it removes vega. Rehedging faster
removes the discretisation error and leaves the vega error untouched.

The rough Bergomi model separates the two knobs. `eta` is the volatility of
volatility and `H` is the roughness. Only one of them moves the floor.

Measured on 4,000 paths, a 1,024-step grid, H = 0.10:
    eta 0.0   sd 3.439 -> 0.333   ratio 0.097   slope -0.494
    eta 0.5   sd 3.612 -> 1.288   ratio 0.357   slope -0.217
    eta 1.0   sd 4.302 -> 2.432   ratio 0.565   slope -0.116
    eta 1.5   sd 5.172 -> 3.553   ratio 0.687   slope -0.075

and at eta = 1.5, H = 0.45 floors at 0.737 against H = 0.10's 0.687, so
roughness leaves the floor where it is. What roughness does change is the
at-the-money skew, which `test_rbergomi.py` measures as a power law in maturity.
"""

import numpy as np
import pytest

# Research sweep: see the slow marker in pyproject.toml.
pytestmark = pytest.mark.slow

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.metrics.stats import sd_vs_frequency
from vollab.paths.base import RoughBergomi

EVERY = [128, 64, 32, 16, 8, 4, 2, 1]


def _cfg(model):
    return HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1), model=model,
        n_mon=1024, cost_bps=0.0, n_paths=4000, seed=77, chunk_paths=2000,
    )


def _curve(H, eta):
    n, sd = sd_vs_frequency(_cfg(RoughBergomi(H=H, eta=eta, rho=-0.7)), EVERY)
    return np.polyfit(np.log(n), np.log(sd), 1)[0], sd[-1] / sd[0]


@pytest.fixture(scope="module")
def curves():
    return {("H0.10", eta): _curve(0.10, eta) for eta in (0.0, 0.5, 1.0, 1.5)} | {
        ("H0.45", 1.5): _curve(0.45, 1.5)
    }


def test_zero_vol_of_vol_recovers_the_square_root_law(curves):
    """The control. Rough Bergomi with eta = 0 is Black-Scholes, so if this
    slope is not -0.5 the model is broken and nothing below means anything."""
    slope, _ = curves[("H0.10", 0.0)]
    assert abs(slope - (-0.5)) < 0.05


def test_vol_of_vol_floors_the_error(curves):
    slope, ratio = curves[("H0.10", 1.5)]
    assert slope > -0.20
    assert ratio > 0.5


def test_the_floor_deepens_monotonically_with_vol_of_vol(curves):
    slopes = [curves[("H0.10", eta)][0] for eta in (0.0, 0.5, 1.0, 1.5)]
    assert all(b > a for a, b in zip(slopes, slopes[1:]))


def test_roughness_does_not_set_the_floor(curves):
    """H = 0.10 and H = 0.45 are a rough and an almost-diffusive variance, and
    they floor in the same place. The floor is vega risk, not roughness."""
    _, rough = curves[("H0.10", 1.5)]
    _, smooth = curves[("H0.45", 1.5)]
    assert abs(rough - smooth) < 0.15


def test_the_floor_is_not_a_jump_floor_in_disguise(curves):
    """Merton's floor comes from gaps in the spot. Here the spot is continuous:
    every path is a diffusion, and the error still stalls."""
    slope, _ = curves[("H0.10", 1.0)]
    assert -0.25 < slope < -0.05
