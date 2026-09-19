import json
from pathlib import Path

import numpy as np

from vollab.hedge.registry import build_config
from vollab.hedge.simulator import simulate

GOLDEN = json.loads((Path(__file__).parent / "golden" / "discretisation.json").read_text())


def test_golden_run_is_bit_reproducible():
    """Any engine change that moves these numbers needs a reason."""
    r = simulate(build_config(GOLDEN["config"]))
    assert r.pnl.mean() == GOLDEN["pnl_mean"]
    assert r.pnl.std(ddof=1) == GOLDEN["pnl_sd"]
    assert int(r.n_rehedges.sum()) == GOLDEN["n_rehedges_total"]
    assert r.turnover.mean() == GOLDEN["turnover_mean"]
    assert int(r.rehedge_mask_hash[0]) == GOLDEN["mask_hash_xor"]


def test_golden_attribution_is_reproducible():
    a = simulate(build_config(GOLDEN["config"])).attribution
    assert a.gamma.mean() == GOLDEN["attr_gamma_mean"]
    assert a.theta.mean() == GOLDEN["attr_theta_mean"]
    assert a.cost.mean() == GOLDEN["attr_cost_mean"]


def test_golden_signs_are_economically_right():
    """Sold vol rich (0.35 implied against 0.28 realized) and hedged: a profit,
    short convexity, collected decay, paid costs."""
    assert GOLDEN["pnl_mean"] > 0
    assert GOLDEN["attr_gamma_mean"] < 0
    assert GOLDEN["attr_theta_mean"] > 0
    assert GOLDEN["attr_cost_mean"] < 0
