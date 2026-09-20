import numpy as np
import pytest

from vollab.surface.calibrate import calibrate_svi
from vollab.surface.svi import SVIParams, implied_vol

T = 1.0
TRUE = SVIParams(a=0.035, b=0.35, rho=-0.45, m=0.02, sigma=0.12)
K = np.linspace(-0.6, 0.6, 21)


@pytest.fixture(scope="module")
def clean_fit():
    return calibrate_svi(K, implied_vol(TRUE, K, T), T)


def test_noise_free_recovery_is_exact(clean_fit):
    p, rmse, _ = clean_fit
    assert rmse < 1e-6
    assert np.allclose(p.as_array(), TRUE.as_array(), atol=1e-3)


def test_reported_rmse_matches_the_residuals(clean_fit):
    p, rmse, diag = clean_fit
    fitted = implied_vol(p, K, T)
    assert np.sqrt(np.mean((fitted - implied_vol(TRUE, K, T)) ** 2)) == pytest.approx(rmse)
    assert diag["rmse_vol_points"] == pytest.approx(rmse * 100)


def test_fit_to_a_noisy_smile_stays_arbitrage_free():
    rng = np.random.default_rng(0)
    iv = implied_vol(TRUE, K, T) + rng.normal(0, 0.004, K.size)
    _, _, diag = calibrate_svi(K, iv, T, arb_free=True)
    assert diag["min_durrleman_g"] >= -1e-8
    assert diag["rmse_vol_points"] < 1.0


def test_constrained_fit_respects_the_lee_wing_bound():
    rng = np.random.default_rng(3)
    iv = implied_vol(TRUE, K, T) + rng.normal(0, 0.01, K.size)
    _, _, diag = calibrate_svi(K, iv, T, arb_free=True)
    assert diag["wing_left"] <= 2.0 + 1e-6
    assert diag["wing_right"] <= 2.0 + 1e-6


def test_constrained_fit_keeps_total_variance_positive():
    _, _, diag = calibrate_svi(K, implied_vol(TRUE, K, T), T)
    assert diag["min_total_variance"] > 0


def test_weights_move_the_fit_toward_the_weighted_points():
    """Weighting the at-the-money point heavily must tighten the fit there."""
    rng = np.random.default_rng(5)
    iv = implied_vol(TRUE, K, T) + rng.normal(0, 0.01, K.size)
    atm = np.argmin(np.abs(K))
    w = np.ones(K.size)
    w[atm] = 50.0
    p_flat, _, _ = calibrate_svi(K, iv, T)
    p_w, _, _ = calibrate_svi(K, iv, T, weights=w)
    err_flat = abs(implied_vol(p_flat, K[atm], T) - iv[atm])
    err_w = abs(implied_vol(p_w, K[atm], T) - iv[atm])
    assert err_w <= err_flat + 1e-9


def test_calibration_recovers_a_heston_smile():
    """A realistic target: Heston produces a genuine smile, and SVI should fit
    it to well under a vol point."""
    from vollab.pricing.black_scholes import bs_implied_vol
    from vollab.pricing.heston_cf import heston_price

    S, r, q, Tm = 100.0, 0.0, 0.0, 1.0
    par = dict(v0=0.06, kap_h=2.0, th_h=0.05, xi=0.5, rho=-0.6)
    ks = np.linspace(-0.4, 0.4, 15)
    strikes = S * np.exp(ks)
    iv = np.array([
        bs_implied_vol("call", heston_price("call", S, k, Tm, r, q, **par),
                       S, k, Tm, r, q)
        for k in strikes
    ])
    _, rmse, diag = calibrate_svi(ks, iv, Tm, arb_free=True)
    assert diag["rmse_vol_points"] < 0.5
    assert diag["min_durrleman_g"] >= -1e-8


def test_calibration_raises_rather_than_returning_a_bad_fit():
    with pytest.raises(RuntimeError):
        calibrate_svi(np.array([0.0]), np.array([np.nan]), T)


def test_quasi_explicit_recovers_a_short_dated_slice_exactly():
    """The case a general 5D optimiser fails on: total variance of order 0.01,
    where an unscaled local method misses a perfect fit by vol points."""
    short = SVIParams(a=0.008, b=0.25, rho=-0.7, m=0.0, sigma=0.1)
    Ts = 0.10
    ks = np.linspace(-0.45, 0.45, 9)
    p, rmse, diag = calibrate_svi(ks, implied_vol(short, ks, Ts), Ts)
    assert diag["rmse_vol_points"] < 1e-3
    assert np.allclose(p.as_array(), short.as_array(), atol=5e-3)


def test_calibration_refuses_an_arbitrageable_target():
    """A slice whose own density is negative admits no arbitrage-free fit, and
    saying so is better than returning a slice that silently is not one."""
    from vollab.surface.svi import durrleman_g

    bad = SVIParams(a=0.004, b=0.28, rho=-0.75, m=0.03, sigma=0.035)
    Ts = 0.10
    kg = np.linspace(-0.7, 0.7, 401)
    assert durrleman_g(bad, kg).min() < 0       # the target really is inadmissible

    ks = np.linspace(-0.45, 0.45, 9)
    with pytest.raises(RuntimeError, match="arbitrage-free"):
        calibrate_svi(ks, implied_vol(bad, ks, Ts), Ts, arb_free=True)


def test_unconstrained_fit_reproduces_an_arbitrageable_target():
    """Without the constraint the fit follows the data wherever it goes."""
    bad = SVIParams(a=0.004, b=0.28, rho=-0.75, m=0.03, sigma=0.035)
    Ts = 0.10
    ks = np.linspace(-0.45, 0.45, 9)
    _, _, diag = calibrate_svi(ks, implied_vol(bad, ks, Ts), Ts, arb_free=False)
    assert diag["min_durrleman_g"] < 0
