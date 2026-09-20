import numpy as np
import pytest

from vollab.surface.svi import (
    SVIParams, butterfly_violation, calendar_violation, durrleman_g, _derivatives,
    implied_vol, min_total_variance, risk_neutral_density, total_variance,
    wing_slopes,
)

CLEAN = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
ARBED = SVIParams(a=0.01, b=0.9, rho=-0.85, m=0.0, sigma=0.03)
K = np.linspace(-1.5, 1.5, 601)


def test_roundtrip_through_array():
    assert SVIParams.from_array(CLEAN.as_array()) == CLEAN


def test_total_variance_is_positive_and_convex_at_the_money():
    w = total_variance(CLEAN, K)
    assert np.all(w > 0)
    assert w[len(w) // 2] == pytest.approx(total_variance(CLEAN, 0.0))


def test_closed_form_first_derivative_matches_finite_difference():
    h = 1e-5
    _, dw, _ = _derivatives(CLEAN, K)
    fd = (total_variance(CLEAN, K + h) - total_variance(CLEAN, K - h)) / (2 * h)
    assert np.abs(dw - fd).max() < 1e-8


def test_closed_form_second_derivative_matches_finite_difference():
    h = 1e-4
    _, _, d2w = _derivatives(CLEAN, K)
    fd = (total_variance(CLEAN, K + h) - 2 * total_variance(CLEAN, K)
          + total_variance(CLEAN, K - h)) / h ** 2
    assert np.abs(d2w - fd).max() < 1e-5


def test_zero_wing_slope_gives_a_flat_smile():
    flat = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    iv = implied_vol(flat, K, 1.0)
    assert np.allclose(iv, iv[0])


def test_min_total_variance_matches_a_grid_search():
    fine = np.linspace(-6.0, 6.0, 20001)
    assert min_total_variance(CLEAN) == pytest.approx(
        total_variance(CLEAN, fine).min(), abs=1e-6)


def test_wing_slopes_are_the_asymptotic_gradients():
    """Both slopes are reported as positive magnitudes: total variance rises in
    both directions away from the money, so the left wing's gradient in k is
    negative while its slope is b*(1 - rho)."""
    left, right = wing_slopes(CLEAN)
    far = 500.0
    num_right = total_variance(CLEAN, far + 1) - total_variance(CLEAN, far)
    num_left = total_variance(CLEAN, -far - 1) - total_variance(CLEAN, -far)
    assert num_right == pytest.approx(right, rel=1e-3)
    assert num_left == pytest.approx(left, rel=1e-3)


def test_clean_slice_has_no_butterfly_arbitrage():
    assert butterfly_violation(CLEAN, K) == 0.0
    assert np.all(durrleman_g(CLEAN, K) > 0)


def test_arbitraged_slice_is_detected():
    """A steep, low-variance, highly skewed slice implies a negative density."""
    assert butterfly_violation(ARBED, K) > 0.1
    assert risk_neutral_density(ARBED, K, 1.0).min() < 0


def test_density_is_positive_for_a_clean_slice():
    assert risk_neutral_density(CLEAN, K, 1.0).min() > 0


def test_calendar_violation_is_directional():
    near = SVIParams(0.02, 0.3, -0.3, 0.0, 0.1)
    far = SVIParams(0.05, 0.4, -0.3, 0.0, 0.1)
    assert calendar_violation(near, far, K) == 0.0
    assert calendar_violation(far, near, K) > 0.0


def test_implied_vol_scales_with_maturity():
    """Same total variance over twice the time is a lower volatility."""
    assert implied_vol(CLEAN, 0.0, 2.0) < implied_vol(CLEAN, 0.0, 1.0)
