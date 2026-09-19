import pytest

from vollab.pricing.black_scholes import bs_price, bs_implied_vol, bs_vega

GRID = [
    (kind, K, T, s)
    for kind in ("call", "put")
    for K in (60.0, 100.0, 160.0)
    for T in (0.05, 1.0, 3.0)
    for s in (0.1, 0.45, 1.2)
]

# Below this vega the price is flat in volatility to within double precision, so
# the vol is not recoverable however good the solver is. Measured vega for the
# excluded cases is of order 1e-95 and smaller.
VEGA_FLOOR = 1e-3


@pytest.mark.parametrize("kind,K,T,s", GRID)
def test_price_roundtrips_exactly(kind, K, T, s):
    """The invariant that holds everywhere: the recovered vol reprices the option."""
    S, r, q = 100.0, 0.03, 0.01
    px = bs_price(kind, S, K, T, r, q, s)
    iv = bs_implied_vol(kind, px, S, K, T, r, q)
    assert abs(bs_price(kind, S, K, T, r, q, iv) - px) < 1e-10 * max(1.0, px)


@pytest.mark.parametrize("kind,K,T,s", GRID)
def test_vol_roundtrips_where_vega_is_meaningful(kind, K, T, s):
    S, r, q = 100.0, 0.03, 0.01
    if bs_vega(kind, S, K, T, r, q, s) < VEGA_FLOOR:
        pytest.skip("vega below the double-precision floor; vol is not identifiable")
    px = bs_price(kind, S, K, T, r, q, s)
    assert abs(bs_implied_vol(kind, px, S, K, T, r, q) - s) < 1e-8


def test_raises_below_intrinsic():
    with pytest.raises(ValueError):
        bs_implied_vol("call", 0.0, 100.0, 50.0, 1.0, 0.0, 0.0)
