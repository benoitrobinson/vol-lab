import json
from pathlib import Path

import numpy as np
import pytest

from vollab.hedge.registry import build_config
from vollab.hedge.simulator import simulate

GOLDEN = json.loads((Path(__file__).parent / "golden" / "discretisation.json").read_text())


# Floating-point results are not bit-portable across architectures: numpy's
# pairwise summation, SIMD width and libm all differ between arm64 and x86-64.
# CI caught this immediately, with the golden mean differing in the last bit
# between a macOS arm64 developer machine and a Linux x86-64 runner. The same
# reasoning already governs the C++ parity tiers, so the same resolution applies
# here: assert a tolerance far tighter than any real engine change, and reserve
# exact equality for integers.
FLOAT_TOL = 1e-12


def test_golden_run_is_reproducible():
    """Any engine change that moves these numbers needs a reason.

    A relative tolerance of 1e-12 is roughly four orders of magnitude tighter
    than the smallest change a genuine bug would produce, and several orders
    looser than cross-platform last-bit noise.
    """
    r = simulate(build_config(GOLDEN["config"]))
    assert r.pnl.mean() == pytest.approx(GOLDEN["pnl_mean"], rel=FLOAT_TOL)
    assert r.pnl.std(ddof=1) == pytest.approx(GOLDEN["pnl_sd"], rel=FLOAT_TOL)
    assert r.turnover.mean() == pytest.approx(GOLDEN["turnover_mean"], rel=FLOAT_TOL)
    # Integers must match exactly: a differing trade count is a real difference,
    # never a rounding artifact.
    assert int(r.n_rehedges.sum()) == GOLDEN["n_rehedges_total"]
    assert int(r.rehedge_mask_hash[0]) == GOLDEN["mask_hash_xor"]


def test_golden_attribution_is_reproducible():
    a = simulate(build_config(GOLDEN["config"])).attribution
    assert a.gamma.mean() == pytest.approx(GOLDEN["attr_gamma_mean"], rel=FLOAT_TOL)
    assert a.theta.mean() == pytest.approx(GOLDEN["attr_theta_mean"], rel=FLOAT_TOL)
    assert a.cost.mean() == pytest.approx(GOLDEN["attr_cost_mean"], rel=FLOAT_TOL)


def test_golden_signs_are_economically_right():
    """Sold vol rich (0.35 implied against 0.28 realized) and hedged: a profit,
    short convexity, collected decay, paid costs."""
    assert GOLDEN["pnl_mean"] > 0
    assert GOLDEN["attr_gamma_mean"] < 0
    assert GOLDEN["attr_theta_mean"] > 0
    assert GOLDEN["attr_cost_mean"] < 0
