"""Inverse options, gated by two independent pricing routes."""

import numpy as np
import pytest

from vollab.pricing.black_scholes import bs_delta, bs_price
from vollab.pricing.inverse import (
    fiat_delta_mismatch, inverse_delta, inverse_gamma, inverse_payoff,
    inverse_price, share_measure_drift,
)
from vollab.rng.scheme import normals_block

S0, K, T, R, Q, S_VOL = 100.0, 110.0, 0.5, 0.03, 0.0, 0.6      # crypto-like vol
# Sample sizes are the smallest that keep these meaningful. A 3-standard-
# error band widens as n falls, so a smaller sample makes the test less
# likely to fail spuriously, not more. What it catches is a structurally
# wrong formula, which is off by many standard errors at any n. The
# research-precision versions live behind the slow marker.
N = 80_000


@pytest.fixture(scope="module")
def z():
    return normals_block(5, 0, N, 1)[:, 0]


def _claim(kind):
    return inverse_price(kind, S0, K, T, R, Q, S_VOL)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_dollar_route_matches(kind, z):
    """Discount the dollar payoff under Q and convert at today's spot."""
    ST = S0 * np.exp((R - Q - 0.5 * S_VOL ** 2) * T + S_VOL * np.sqrt(T) * z)
    intrinsic = (np.maximum(ST - K, 0.0) if kind == "call"
                 else np.maximum(K - ST, 0.0))
    mc = np.exp(-R * T) * intrinsic / S0
    se = mc.std(ddof=1) / np.sqrt(N)
    assert abs(mc.mean() - _claim(kind)) < 3 * se


@pytest.mark.parametrize("kind", ["call", "put"])
def test_share_measure_route_matches(kind, z):
    """Take the coin payoff under the share measure and discount at q.

    Independent of the first route: a different measure, a different payoff
    and a different discount factor, agreeing on the same number.
    """
    mu = share_measure_drift(R, Q, S_VOL)
    ST = S0 * np.exp((mu - 0.5 * S_VOL ** 2) * T + S_VOL * np.sqrt(T) * z)
    mc = np.exp(-Q * T) * inverse_payoff(kind, ST, K)
    se = mc.std(ddof=1) / np.sqrt(N)
    assert abs(mc.mean() - _claim(kind)) < 3 * se


def test_naive_expectation_is_not_the_price(z):
    """The trap: E^Q[coin payoff] is not a price, and is not close to one."""
    ST = S0 * np.exp((R - Q - 0.5 * S_VOL ** 2) * T + S_VOL * np.sqrt(T) * z)
    naive = inverse_payoff("call", ST, K).mean()
    assert abs(naive - _claim("call")) > 20 * (
        inverse_payoff("call", ST, K).std(ddof=1) / np.sqrt(N))


def test_payoff_times_spot_recovers_the_vanilla_payoff():
    ST = np.array([50.0, 110.0, 200.0])
    assert np.allclose(inverse_payoff("call", ST, K) * ST,
                       np.maximum(ST - K, 0.0))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_a_finite_difference(kind):
    h = 1e-4
    fd = (inverse_price(kind, S0 + h, K, T, R, Q, S_VOL)
          - inverse_price(kind, S0 - h, K, T, R, Q, S_VOL)) / (2 * h)
    assert abs(inverse_delta(kind, S0, K, T, R, Q, S_VOL) - fd) < 1e-7


@pytest.mark.parametrize("kind", ["call", "put"])
def test_gamma_matches_a_finite_difference(kind):
    h = 1e-3
    fd = (inverse_delta(kind, S0 + h, K, T, R, Q, S_VOL)
          - inverse_delta(kind, S0 - h, K, T, R, Q, S_VOL)) / (2 * h)
    assert abs(inverse_gamma(kind, S0, K, T, R, Q, S_VOL) - fd) < 1e-6


def test_coin_delta_differs_from_a_converted_vanilla_delta():
    """The finding: the gap is the option's own value over spot squared."""
    coin, naive, diff = fiat_delta_mismatch("call", S0, K, T, R, Q, S_VOL)
    expected_gap = -bs_price("call", S0, K, T, R, Q, S_VOL) / S0 ** 2
    assert abs(diff - expected_gap) < 1e-12
    assert abs(diff / naive) > 0.01          # materially different, not a rounding


def test_mismatch_is_largest_where_the_option_is_most_valuable():
    spots = np.array([60.0, 110.0, 200.0])
    _, _, diff = fiat_delta_mismatch("call", spots, K, T, R, Q, S_VOL)
    value = bs_price("call", spots, K, T, R, Q, S_VOL)
    assert np.argmax(np.abs(diff)) == np.argmax(value / spots ** 2)


def test_deep_in_the_money_call_approaches_one_minus_strike_over_spot():
    deep = inverse_price("call", 10_000.0, K, 0.01, 0.0, 0.0, 0.2)
    assert abs(deep - (1.0 - K / 10_000.0)) < 1e-6


def test_put_call_parity_in_coin_terms():
    """call - put = (S - K e^{-rT} ... ) / S, inherited from the vanilla parity."""
    c = inverse_price("call", S0, K, T, R, Q, S_VOL)
    p = inverse_price("put", S0, K, T, R, Q, S_VOL)
    rhs = (S0 * np.exp(-Q * T) - K * np.exp(-R * T)) / S0
    assert abs((c - p) - rhs) < 1e-12
