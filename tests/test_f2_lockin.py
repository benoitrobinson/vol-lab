import numpy as np
import pytest

# Research sweep: see the slow marker in pyproject.toml.
pytestmark = pytest.mark.slow

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import simulate
from vollab.pricing.black_scholes import bs_price

S0, K, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
S_IMP, S_REAL = 0.35, 0.25          # sold rich: implied above realized


def _cfg(s_hedge):
    return HedgeConfig(
        contract=Contract("call", S0, K, T, r, q),
        vols=VolSpec(s_imp=S_IMP, s_hedge=s_hedge, s_real=S_REAL),
        schedule=FixedTime(1), n_mon=2048, cost_bps=0.0,
        n_paths=8_000, seed=21, chunk_paths=4_000,
    )


def _edge():
    return (bs_price("call", S0, K, T, r, q, S_IMP)
            - bs_price("call", S0, K, T, r, q, S_REAL))


@pytest.fixture(scope="module")
def runs():
    return {"at_real": simulate(_cfg(S_REAL)).pnl,
            "at_imp": simulate(_cfg(S_IMP)).pnl}


def test_drift_is_pinned_to_r_minus_q():
    """The equality of the two means holds only under this drift."""
    assert _cfg(S_REAL).mu is None


def test_selling_rich_vol_has_a_positive_edge():
    assert _edge() > 0


def test_hedging_at_realized_locks_in_the_edge(runs):
    pnl = runs["at_real"]
    se = pnl.std(ddof=1) / np.sqrt(pnl.size)
    assert abs(pnl.mean() - _edge()) < 4 * se


def test_hedging_at_realized_has_almost_no_dispersion(runs):
    """Dispersion here is only the discretisation error of F1, not vol risk."""
    assert runs["at_real"].std(ddof=1) < 0.05 * abs(_edge())


def test_hedging_at_implied_has_the_same_mean(runs):
    pnl = runs["at_imp"]
    se = pnl.std(ddof=1) / np.sqrt(pnl.size)
    assert abs(pnl.mean() - _edge()) < 4 * se


def test_hedging_at_implied_is_far_more_path_dependent(runs):
    assert runs["at_imp"].std(ddof=1) > 5 * runs["at_real"].std(ddof=1)


def test_hedging_at_implied_has_the_sign_of_s_imp_minus_s_real(runs):
    """Short vol: you profit when realized comes in below implied. The sign
    follows s_imp - s_real, matching the sign of the mean."""
    assert (runs["at_imp"] > 0).mean() > 0.99
