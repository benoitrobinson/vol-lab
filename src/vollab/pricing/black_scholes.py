"""Black-Scholes price, greeks and implied volatility.

All greeks here are long-option greeks. Callers holding a short position negate
them; the position-greek convention of the spec is applied at the call site.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, s):
    v = s * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * s * s) * T) / v
    return d1, d1 - v


def bs_price(kind, S, K, T, r, q, s):
    d1, d2 = _d1_d2(S, K, T, r, q, s)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    if kind == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_delta(kind, S, K, T, r, q, s):
    d1, _ = _d1_d2(S, K, T, r, q, s)
    df_q = np.exp(-q * T)
    if kind == "call":
        return df_q * norm.cdf(d1)
    if kind == "put":
        return df_q * (norm.cdf(d1) - 1.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_gamma(kind, S, K, T, r, q, s):
    d1, _ = _d1_d2(S, K, T, r, q, s)
    return np.exp(-q * T) * norm.pdf(d1) / (S * s * np.sqrt(T))


def bs_vega(kind, S, K, T, r, q, s):
    d1, _ = _d1_d2(S, K, T, r, q, s)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(kind, S, K, T, r, q, s):
    d1, d2 = _d1_d2(S, K, T, r, q, s)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    common = -S * df_q * norm.pdf(d1) * s / (2 * np.sqrt(T))
    if kind == "call":
        return common - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    if kind == "put":
        return common + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_implied_vol(kind, price, S, K, T, r, q, lo=1e-8, hi=6.0):
    def f(x):
        return bs_price(kind, S, K, T, r, q, x) - price

    if f(lo) > 0 or f(hi) < 0:
        raise ValueError(
            f"price {price} outside attainable range for s in [{lo}, {hi}]"
        )
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
