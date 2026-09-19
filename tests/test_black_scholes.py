import numpy as np
from vollab.pricing.black_scholes import bs_price


def test_put_call_parity():
    S, K, T, r, q, s = 100.0, 105.0, 0.5, 0.03, 0.01, 0.35
    c = bs_price("call", S, K, T, r, q, s)
    p = bs_price("put", S, K, T, r, q, s)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-12


def test_known_value():
    from scipy.stats import norm
    expected = 100.0 * (2 * norm.cdf(0.1) - 1)
    got = bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.2)
    assert abs(got - expected) < 1e-12


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        bs_price("straddle", 100.0, 100.0, 1.0, 0.0, 0.0, 0.2)
