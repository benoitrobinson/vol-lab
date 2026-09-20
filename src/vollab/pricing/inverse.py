"""Coin-margined (inverse) options.

Crypto venues list options that settle in the underlying coin rather than in
fiat. A BTC call pays `max(S_T - K, 0) / S_T` bitcoin, and its premium, margin
and P&L are all denominated in coin.

Multiplying that payoff by `S_T` gives back `max(S_T - K, 0)`, the vanilla
dollar payoff exactly. So the *dollar* value of an inverse option is a vanilla
option's value, and its price quoted in coin is

    inverse_price_coin(S, K, T, r, q, s) = bs_price(S, K, T, r, q, s) / S

That is a change of units at inception and nothing more. What is not a change of
units is the hedge. Differentiating the coin price,

    d/dS [ C(S) / S ] = Delta(S)/S - C(S)/S^2

so the coin delta is the naive translation `Delta/S` minus a term `C/S^2` that
does not vanish. A desk hedging an inverse book with converted vanilla deltas
carries a systematic error of exactly the option's own value over spot squared,
which is largest where the option is most valuable.

A warning about what this is not. `E^Q[coin payoff]` under the dollar
risk-neutral measure is *not* the coin price; Jensen's inequality separates them
by a wide margin. The price in coin is recovered either by discounting the
dollar payoff and converting at today's spot, or by taking the coin payoff under
the share measure, where the drift is `r - q + s^2`, and discounting at `q`.
Both routes are checked in `tests/test_inverse.py`; they agree, and the naive
expectation does not.
"""

import numpy as np

from vollab.pricing.black_scholes import bs_delta, bs_gamma, bs_price


def inverse_price(kind, S, K, T, r, q, s):
    """Price of a coin-settled option, denominated in coin."""
    S = np.asarray(S, dtype=float)
    return bs_price(kind, S, K, T, r, q, s) / S


def inverse_payoff(kind, S_T, K):
    """Terminal payoff in coin."""
    S_T = np.asarray(S_T, dtype=float)
    intrinsic = (np.maximum(S_T - K, 0.0) if kind == "call"
                 else np.maximum(K - S_T, 0.0))
    return intrinsic / S_T


def inverse_delta(kind, S, K, T, r, q, s):
    """d(coin price)/dS, in closed form.

    Delta/S - C/S^2. The second term is the whole difference from a naively
    converted vanilla delta.
    """
    S = np.asarray(S, dtype=float)
    return (bs_delta(kind, S, K, T, r, q, s) / S
            - bs_price(kind, S, K, T, r, q, s) / (S * S))


def inverse_gamma(kind, S, K, T, r, q, s):
    """d2(coin price)/dS2 = Gamma/S - 2*Delta/S^2 + 2*C/S^3."""
    S = np.asarray(S, dtype=float)
    return (bs_gamma(kind, S, K, T, r, q, s) / S
            - 2.0 * bs_delta(kind, S, K, T, r, q, s) / (S * S)
            + 2.0 * bs_price(kind, S, K, T, r, q, s) / (S * S * S))


def fiat_delta_mismatch(kind, S, K, T, r, q, s):
    """What a desk loses by hedging an inverse book with converted vanilla deltas.

    Returns (coin_delta, naive_delta, difference). The naive translation divides
    the vanilla delta by spot, which omits the `C/S^2` term entirely.
    """
    S = np.asarray(S, dtype=float)
    coin = inverse_delta(kind, S, K, T, r, q, s)
    naive = bs_delta(kind, S, K, T, r, q, s) / S
    return coin, naive, coin - naive


def share_measure_drift(r, q, s):
    """Drift of the underlying under the coin (share) numeraire measure."""
    return r - q + s * s
