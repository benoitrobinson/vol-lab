"""Black-Scholes price, greeks and implied volatility.

All greeks here are long-option greeks. Callers holding a short position negate
them; the position-greek convention of the spec is applied at the call site.
"""

import numpy as np
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
