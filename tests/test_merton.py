import numpy as np
import pytest

from vollab.paths.merton import jump_budget, merton_paths
from vollab.pricing.black_scholes import bs_price
from vollab.pricing.merton import compensator, merton_price

ARGS = dict(S=100.0, K=105.0, T=0.5, r=0.03, q=0.0, s=0.35)
JUMPS = dict(lam=0.8, mu_J=-0.12, s_J=0.25)


def test_compensator_matches_lognormal_mean():
    mu_J, s_J = -0.12, 0.25
    assert compensator(mu_J, s_J) == pytest.approx(np.exp(mu_J + 0.5 * s_J ** 2) - 1)


def test_zero_intensity_reduces_to_black_scholes():
    ref = bs_price("call", **ARGS)
    got = merton_price("call", **ARGS, lam=0.0, mu_J=-0.1, s_J=0.2)
    assert abs(got - ref) < 1e-12


def test_jumps_raise_the_price_of_an_out_of_the_money_call():
    """Extra kurtosis is worth something to an OTM option."""
    assert merton_price("call", **ARGS, **JUMPS) > bs_price("call", **ARGS)


def test_put_call_parity_holds_under_jumps():
    c = merton_price("call", **ARGS, **JUMPS)
    p = merton_price("put", **ARGS, **JUMPS)
    lhs = c - p
    rhs = ARGS["S"] * np.exp(-ARGS["q"] * ARGS["T"]) - ARGS["K"] * np.exp(-ARGS["r"] * ARGS["T"])
    assert abs(lhs - rhs) < 1e-10


def test_monte_carlo_matches_the_series_price():
    """The ground truth that gates every Merton finding."""
    n = 300_000
    P = merton_paths(ARGS["S"], ARGS["r"], ARGS["q"], ARGS["s"], ARGS["T"],
                     64, 7, 0, n, **JUMPS)
    disc = np.exp(-ARGS["r"] * ARGS["T"]) * np.maximum(P[:, -1] - ARGS["K"], 0.0)
    se = disc.std(ddof=1) / np.sqrt(n)
    assert abs(disc.mean() - merton_price("call", **ARGS, **JUMPS)) < 3 * se


def test_jump_budget_covers_the_poisson_tail():
    assert jump_budget(1.0, 1.0) > 10
    assert jump_budget(5.0, 1.0) > jump_budget(1.0, 1.0)


def test_paths_start_at_spot_and_stay_positive():
    P = merton_paths(100.0, 0.0, 0.0, 0.3, 1.0, 32, 5, 0, 500, **JUMPS)
    assert np.all(P[:, 0] == 100.0)
    assert np.all(P > 0)


def test_terminal_distribution_is_left_skewed_for_negative_jumps():
    from scipy.stats import skew
    P = merton_paths(100.0, 0.0, 0.0, 0.2, 1.0, 64, 11, 0, 40_000,
                     lam=2.0, mu_J=-0.25, s_J=0.1)
    assert skew(np.log(P[:, -1])) < -0.2
