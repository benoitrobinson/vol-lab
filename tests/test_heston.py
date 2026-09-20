import numpy as np
import pytest

from vollab.paths.heston import heston_paths
from vollab.pricing.heston_cf import feller_ratio, heston_price

FELLER_OK = dict(v0=0.04, kap_h=3.0, th_h=0.04, xi=0.3, rho=-0.7)
FELLER_VIOLATED = dict(v0=0.09, kap_h=1.5, th_h=0.09, xi=0.9, rho=-0.6)
MKT = dict(S=100.0, K=100.0, T=1.0, r=0.02, q=0.0)


def test_feller_ratio_classifies_both_parameter_sets():
    assert feller_ratio(FELLER_OK["kap_h"], FELLER_OK["th_h"], FELLER_OK["xi"]) > 1.0
    assert feller_ratio(FELLER_VIOLATED["kap_h"], FELLER_VIOLATED["th_h"],
                        FELLER_VIOLATED["xi"]) < 1.0


def test_zero_vol_of_vol_approaches_black_scholes():
    from vollab.pricing.black_scholes import bs_price
    flat = dict(v0=0.04, kap_h=2.0, th_h=0.04, xi=1e-6, rho=0.0)
    got = heston_price("call", **MKT, **flat)
    ref = bs_price("call", MKT["S"], MKT["K"], MKT["T"], MKT["r"], MKT["q"], 0.2)
    assert abs(got - ref) < 1e-4


def test_put_call_parity():
    c = heston_price("call", **MKT, **FELLER_OK)
    p = heston_price("put", **MKT, **FELLER_OK)
    rhs = MKT["S"] * np.exp(-MKT["q"] * MKT["T"]) - MKT["K"] * np.exp(-MKT["r"] * MKT["T"])
    assert abs((c - p) - rhs) < 1e-8


@pytest.mark.parametrize("params,label", [(FELLER_OK, "satisfied"),
                                          (FELLER_VIOLATED, "violated")])
def test_monte_carlo_matches_the_characteristic_function(params, label):
    """QE must hold up where Feller is violated; that is the case it exists for."""
    n = 40_000
    P = heston_paths(MKT["S"], MKT["r"], MKT["q"], MKT["T"], 256, 3, 0, n,
                     params["v0"], params["kap_h"], params["th_h"],
                     params["xi"], params["rho"])
    disc = np.exp(-MKT["r"] * MKT["T"]) * np.maximum(P[:, -1] - MKT["K"], 0.0)
    se = disc.std(ddof=1) / np.sqrt(n)
    ref = heston_price("call", **MKT, **params)
    assert abs(disc.mean() - ref) < 3 * se, f"{label}: {disc.mean():.4f} vs {ref:.4f}"


def test_variance_stays_non_negative_under_violated_feller():
    P = heston_paths(100.0, 0.0, 0.0, 1.0, 256, 5, 0, 2000, **FELLER_VIOLATED)
    assert np.all(np.isfinite(P)) and np.all(P > 0)


def test_negative_correlation_produces_a_left_skew():
    from scipy.stats import skew
    P = heston_paths(100.0, 0.0, 0.0, 1.0, 256, 9, 0, 20_000, **FELLER_OK)
    assert skew(np.log(P[:, -1])) < 0
