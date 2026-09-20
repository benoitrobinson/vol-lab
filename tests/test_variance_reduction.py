import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import simulate
from vollab.metrics.variance_reduction import (
    apply_control, realized_variance_control, variance_ratio,
)
from vollab.paths.base import GBM
from vollab.paths.gbm import gbm_paths
from vollab.pricing.black_scholes import bs_price
from vollab.rng.scheme import normals_block

# Sample sizes are the smallest that keep these meaningful. A 3-standard-
# error band widens as n falls, so a smaller sample makes the test less
# likely to fail spuriously, not more. What it catches is a structurally
# wrong formula, which is off by many standard errors at any n. The
# research-precision versions live behind the slow marker.
SEED, N, N_MON = 101, 8_000, 256


@pytest.fixture(scope="module")
def pnl():
    cfg = HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1), model=GBM(),
        n_mon=N_MON, cost_bps=0.0, n_paths=N, seed=SEED, chunk_paths=N,
    )
    return simulate(cfg).pnl


def test_control_has_mean_zero():
    c = realized_variance_control(SEED, 4000, 128)
    assert abs(c.mean()) < 4 * c.std(ddof=1) / np.sqrt(c.size)


def test_control_variate_preserves_the_estimate(pnl):
    c = realized_variance_control(SEED, N, N_MON)
    adj, _, _ = apply_control(pnl, c)
    se = adj.std(ddof=1) / np.sqrt(N)
    assert abs(adj.mean() - pnl.mean()) < 4 * se


@pytest.mark.slow
def test_control_variate_cuts_variance_by_more_than_half(pnl):
    """Derived from the F1 mechanism, not tuned. Measured ratio about 0.37."""
    c = realized_variance_control(SEED, N, N_MON)
    adj, _, rho = apply_control(pnl, c)
    assert rho < -0.7
    assert variance_ratio(pnl, adj) < 0.5


@pytest.mark.slow
def test_terminal_payoff_control_is_useless(pnl):
    """The instructive negative result: a working hedge removes exactly the
    component of P&L that tracks the terminal value."""
    S = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, N_MON, SEED, 0, N)
    payoff = np.maximum(S[:, -1] - 100.0, 0.0)
    mu = bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.3)
    adj, _, rho = apply_control(pnl, payoff, mu)
    assert abs(rho) < 0.1
    assert variance_ratio(pnl, adj) > 0.95


def test_antithetic_sampling_cannot_help():
    """The hedging error is even in z, so a mirrored path gives it back
    unchanged. The correlation is exactly one, not merely close to it."""
    z = normals_block(SEED, 0, 2000, 256)
    stat = (z ** 2 - 1).sum(axis=1)
    stat_mirrored = ((-z) ** 2 - 1).sum(axis=1)
    assert np.allclose(stat, stat_mirrored, rtol=0, atol=0)
    assert np.corrcoef(stat, stat_mirrored)[0, 1] == pytest.approx(1.0)


def test_degenerate_control_is_a_no_op():
    x = np.arange(10.0)
    adj, beta, rho = apply_control(x, np.zeros(10))
    assert np.array_equal(adj, x) and beta == 0.0 and rho == 0.0
