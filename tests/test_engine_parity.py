"""Tiered parity between the NumPy reference and the C++ engine.

Bitwise equality is unattainable by design, not by sloppiness. Roughly 16% of
inverse-CDF draws differ in the last bits because the Cephes tail branch goes
through log(), which is not correctly rounded across libms. Those ULP gaps then
flip discrete decisions: where delta saturates at exactly 0 or 1, one engine
books a trade and the other does not.

Measured divergence: P&L agrees to 4e-13 relative, while 8% to 22% of paths
differ in rehedge count, always by at most two trades out of hundreds.
"""

import numpy as np
import pytest

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import DeltaBand, FixedTime
from vollab.hedge.simulator import cpp_available, simulate
from vollab.paths.base import Merton

_core = pytest.importorskip("vollab._core", reason="C++ extension not built")


def _cfg(**kw):
    base = dict(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(4),
        n_mon=512, cost_bps=5.0, n_paths=4000, seed=1234, chunk_paths=4000,
    )
    base.update(kw)
    return HedgeConfig(**base)


# Tier 1: exact, on the raw bit stream.

def test_uniform_stream_is_bitwise_identical():
    for seed, path, n in [(7, 0, 64), (0, 0, 32), (42, 17, 16)]:
        ref = np.random.Generator(
            np.random.Philox(key=[np.uint64(seed), np.uint64(path)])).random(n)
        assert np.array_equal(_core.uniforms(seed, path, n), ref)


def test_uniforms_match_the_shift_and_scale_identity():
    raw = np.random.Philox(key=[np.uint64(3), np.uint64(9)]).random_raw(16)
    assert np.array_equal(_core.uniforms(3, 9, 16),
                          (raw >> np.uint64(11)) * 2.0 ** -53)


# Tier 2: tolerance, on transcendental output.

def test_normals_agree_to_a_few_ulp():
    from scipy.special import ndtri
    cpp = _core.normals(7, 0, 5000)
    ref = ndtri(_core.uniforms(7, 0, 5000))
    assert np.abs(cpp - ref).max() < 1e-14
    assert np.mean(cpp == ref) > 0.75      # measured 84% exactly equal


# Tier 3: tolerance on P&L, bounded divergence on decisions.

@pytest.mark.parametrize("every,bps", [(1, 0.0), (4, 5.0), (8, 10.0)])
def test_pnl_agrees_to_floating_point_tolerance(every, bps):
    cfg = _cfg(schedule=FixedTime(every), cost_bps=bps)
    ref, cpp = simulate(cfg), simulate(cfg, engine="cpp")
    scale = max(np.abs(ref.pnl).max(), 1.0)
    assert np.abs(cpp.pnl - ref.pnl).max() / scale < 1e-11


@pytest.mark.parametrize("every", [1, 4, 8])
def test_rehedge_counts_diverge_only_marginally(every):
    """Not an exact match: a 1-ULP delta gap at saturation flips a trade."""
    cfg = _cfg(schedule=FixedTime(every))
    ref, cpp = simulate(cfg), simulate(cfg, engine="cpp")
    dn = np.abs(cpp.n_rehedges - ref.n_rehedges)
    assert dn.max() <= 3
    assert dn.mean() < 0.5


def test_engine_used_is_recorded_honestly():
    assert simulate(_cfg()).engine_used == "numpy"
    assert simulate(_cfg(), engine="cpp").engine_used == "cpp"


# Refusals: the engine must never silently substitute the other one.

def test_cpp_refuses_an_unsupported_model():
    with pytest.raises(NotImplementedError, match="GBM only"):
        simulate(_cfg(model=Merton(lam=1.0, mu_J=-0.1, s_J=0.15)), engine="cpp")


def test_cpp_refuses_an_unsupported_schedule():
    with pytest.raises(NotImplementedError, match="FixedTime only"):
        simulate(_cfg(schedule=DeltaBand(0.05)), engine="cpp")


def test_cpp_refuses_a_put():
    with pytest.raises(NotImplementedError, match="calls only"):
        simulate(_cfg(contract=Contract("put", 100.0, 100.0, 1.0, 0.0, 0.0)),
                 engine="cpp")


def test_availability_probe_agrees_with_the_import():
    assert cpp_available() is True
