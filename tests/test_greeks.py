import numpy as np
import pytest

from vollab.pricing.black_scholes import (
    bs_price, bs_delta, bs_gamma, bs_vega, bs_theta,
)

ARGS = (100.0, 105.0, 0.5, 0.03, 0.01, 0.35)


def _central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = _central(lambda x: bs_price(kind, x, K, T, r, q, s), S, 1e-4)
    assert abs(bs_delta(kind, *ARGS) - fd) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_gamma_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = _central(lambda x: bs_delta(kind, x, K, T, r, q, s), S, 1e-4)
    assert abs(bs_gamma(kind, *ARGS) - fd) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_vega_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = _central(lambda x: bs_price(kind, S, K, T, r, q, x), s, 1e-5)
    assert abs(bs_vega(kind, *ARGS) - fd) < 1e-5 * max(1.0, abs(fd))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_theta_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = -_central(lambda x: bs_price(kind, S, K, x, r, q, s), T, 1e-5)
    assert abs(bs_theta(kind, *ARGS) - fd) < 1e-5 * max(1.0, abs(fd))


def test_gamma_and_vega_are_kind_independent():
    assert bs_gamma("call", *ARGS) == pytest.approx(bs_gamma("put", *ARGS))
    assert bs_vega("call", *ARGS) == pytest.approx(bs_vega("put", *ARGS))


def test_greeks_vectorise_over_spot():
    S = np.array([80.0, 100.0, 120.0])
    _, K, T, r, q, s = ARGS
    d = bs_delta("call", S, K, T, r, q, s)
    assert d.shape == (3,)
    assert np.all(np.diff(d) > 0)
