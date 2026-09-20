import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import FixedTime, Leland, WhalleyWilmott
from vollab.hedge.simulator import simulate


def test_leland_raises_vol_for_a_written_option():
    s, k, dt = 0.3, 0.001, 1 / 252
    out = Leland(every=1, short=True).hedge_vol(s, k, dt)
    le = np.sqrt(2 / np.pi) * k / (s * np.sqrt(dt))
    assert abs(out - s * np.sqrt(1 + le)) < 1e-12
    assert out > s


def test_leland_lowers_vol_for_a_held_option():
    assert Leland(every=1, short=False).hedge_vol(0.3, 0.001, 1 / 252) < 0.3


def test_leland_is_identity_at_zero_cost():
    assert abs(Leland(1).hedge_vol(0.3, 0.0, 1 / 252) - 0.3) < 1e-15


def _band(k, gamma, S=100.0, gam_ra=1.0):
    return WhalleyWilmott(gam_ra=gam_ra).band_width(
        np.array([S]), np.array([gamma]), k
    )[0]


def test_band_scales_as_k_to_the_one_third():
    assert abs(_band(8e-4, 0.02) / _band(1e-4, 0.02) - 8 ** (1 / 3)) < 1e-9


def test_band_scales_as_gamma_to_the_two_thirds():
    assert abs(_band(1e-4, 0.08) / _band(1e-4, 0.02) - 4 ** (2 / 3)) < 1e-9


def test_band_scales_as_s_to_the_one_third():
    """S enters to the first power inside the cube root, not the second."""
    assert abs(_band(1e-4, 0.02, S=800.0) / _band(1e-4, 0.02, S=100.0)
               - 8 ** (1 / 3)) < 1e-9


def test_band_widens_with_risk_tolerance():
    assert _band(1e-4, 0.02, gam_ra=0.1) > _band(1e-4, 0.02, gam_ra=10.0)


def _cfg(schedule, cost_bps=10.0):
    return HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=schedule, n_mon=256,
        cost_bps=cost_bps, n_paths=4000, seed=41, chunk_paths=2000,
    )


def test_whalley_wilmott_trades_less_than_fixed_time():
    ft = simulate(_cfg(FixedTime(1))).n_rehedges
    ww = simulate(_cfg(WhalleyWilmott(gam_ra=1.0))).n_rehedges
    assert ww.mean() < ft.mean()
    assert (ww <= ft).all()


def test_whalley_wilmott_beats_naive_fixed_time_under_costs():
    """The point of a band: less turnover for comparable risk."""
    ft = simulate(_cfg(FixedTime(1)))
    ww = simulate(_cfg(WhalleyWilmott(gam_ra=1.0)))
    assert ww.turnover.mean() < ft.turnover.mean()
    assert ww.pnl.mean() > ft.pnl.mean()


def test_leland_changes_the_hedge_relative_to_plain_fixed_time():
    plain = simulate(_cfg(FixedTime(4))).pnl
    lel = simulate(_cfg(Leland(every=4, short=True))).pnl
    assert not np.array_equal(plain, lel)
