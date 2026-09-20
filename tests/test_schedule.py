import numpy as np

from vollab.hedge.schedule import FixedTime, DeltaBand

Z = np.zeros(3)


def _call(sched, step, target, held):
    return sched.should_trade(
        step, 8, np.asarray(target, dtype=float), np.asarray(held, dtype=float),
        Z + 100.0, Z, 0.125,
    )


def test_fixed_time_trades_on_multiples():
    s = FixedTime(every=2)
    assert _call(s, 0, Z, Z).all()
    assert not _call(s, 1, Z, Z).any()
    assert _call(s, 2, Z, Z).all()


def test_fixed_time_always_trades_final_step():
    assert _call(FixedTime(every=3), 8, Z, Z).all()


def test_delta_band_triggers_only_outside_band():
    m = _call(DeltaBand(h=0.05), 3, [0.50, 0.56, 0.44], [0.50, 0.50, 0.50])
    assert list(m) == [False, True, True]


def test_delta_band_always_trades_first_and_final_step():
    s = DeltaBand(h=0.5)
    assert _call(s, 0, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).all()
    assert _call(s, 8, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).all()


def test_hedge_vol_is_identity_for_both():
    assert FixedTime(2).hedge_vol(0.3, 0.0005, 0.01) == 0.3
    assert DeltaBand(0.05).hedge_vol(0.3, 0.0005, 0.01) == 0.3


def test_mask_shape_matches_paths():
    assert _call(FixedTime(1), 1, Z, Z).shape == (3,)
    assert _call(DeltaBand(0.1), 1, Z, Z).shape == (3,)
