import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import DeltaBand, FixedTime
from vollab.hedge.simulator import simulate


def cfg(**kw):
    base = dict(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(s_imp=0.3, s_hedge=0.3, s_real=0.3),
        schedule=FixedTime(1), n_mon=64, cost_bps=0.0,
        n_paths=2000, seed=1, chunk_paths=512,
    )
    base.update(kw)
    return HedgeConfig(**base)


def test_result_shapes_and_metadata():
    r = simulate(cfg())
    assert r.pnl.shape == (2000,)
    assert r.n_rehedges.dtype == np.int64
    assert r.engine_used == "numpy"
    assert r.rng_scheme_version == 1


def test_chunking_does_not_change_results():
    assert np.array_equal(simulate(cfg(chunk_paths=512)).pnl,
                          simulate(cfg(chunk_paths=2000)).pnl)


def test_static_hedge_trades_exactly_twice():
    r = simulate(cfg(schedule=FixedTime(10 ** 9)))
    assert np.isfinite(r.pnl).all()
    assert (r.n_rehedges == 2).all()


def test_perfect_hedge_mean_pnl_is_zero():
    r = simulate(cfg(n_mon=512, n_paths=40_000, chunk_paths=10_000))
    se = r.pnl.std(ddof=1) / np.sqrt(r.pnl.size)
    assert abs(r.pnl.mean()) < 3 * se


def test_more_frequent_hedging_reduces_dispersion():
    coarse = simulate(cfg(n_mon=512, schedule=FixedTime(64), n_paths=8000))
    fine = simulate(cfg(n_mon=512, schedule=FixedTime(1), n_paths=8000))
    assert fine.pnl.std(ddof=1) < coarse.pnl.std(ddof=1)


def test_costs_reduce_pnl():
    assert simulate(cfg(cost_bps=20.0)).pnl.mean() < simulate(cfg(cost_bps=0.0)).pnl.mean()


def test_band_trades_no_more_often_than_fixed_time_on_same_grid():
    ft = simulate(cfg(schedule=FixedTime(1))).n_rehedges
    bd = simulate(cfg(schedule=DeltaBand(0.02))).n_rehedges
    assert (bd <= ft).all()


def test_turnover_is_non_negative():
    assert (simulate(cfg()).turnover >= 0).all()


def test_cpp_engine_raises_rather_than_falling_back():
    with pytest.raises(NotImplementedError):
        simulate(cfg(), engine="cpp")


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        simulate(cfg(), engine="fortran")


def test_hedging_collapses_dispersion_versus_naked_short():
    """The test that catches a delta sign error.

    A sign error leaves the mean P&L near zero but roughly doubles the exposure
    instead of cancelling it, so dispersion is the discriminating statistic and
    the mean is not. Measured: sd 0.46 hedged against 21.0 naked, and 0.30 for
    the sign-flipped book, which is 91x the correct one.
    """
    from vollab.paths.gbm import gbm_paths
    from vollab.pricing.black_scholes import bs_price

    c = cfg(n_mon=512, n_paths=20_000, chunk_paths=5_000)
    hedged = simulate(c).pnl

    S = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, 512, c.seed, 0, c.n_paths)
    naked = bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.3) - np.maximum(
        S[:, -1] - 100.0, 0.0
    )

    assert hedged.std(ddof=1) < 0.05 * naked.std(ddof=1)
