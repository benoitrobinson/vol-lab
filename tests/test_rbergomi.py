"""Rough Bergomi, gated on what is known in closed form.

There is no closed-form price for this model, so every check here is an
identity the simulation must satisfy: the variance of the Volterra process, the
flat-kernel case where it degenerates to Brownian motion, the zero vol-of-vol
case where it degenerates to Black-Scholes, and the two martingale properties.
A scheme that passes all five is not being compared against itself.
"""

import numpy as np
import pytest

from vollab.paths.rbergomi import rbergomi_paths, variance_paths, volterra_paths
from vollab.pricing.black_scholes import bs_price
from vollab.rng.scheme import normals_block

SEED = 20260923
MKT = dict(S0=100.0, r=0.02, q=0.0, T=1.0)


def _volterra(H, n_mon, n_paths, T=1.0, seed=SEED):
    dt = T / n_mon
    z = normals_block(seed, 0, n_paths, 3 * n_mon)
    return volterra_paths(H, n_mon, dt, z[:, 0::3], z[:, 1::3])


def test_volterra_variance_matches_the_kernel():
    """Var(Y_t) = t^(2H) / (2H). This is what gates the hybrid scheme."""
    H, n_mon, T = 0.1, 256, 1.0
    y, _ = _volterra(H, n_mon, 20_000, T)
    t = np.arange(n_mon + 1) * (T / n_mon)
    for i in (16, 64, 128, 256):
        expected = t[i] ** (2 * H) / (2 * H)
        assert abs(y[:, i].var(ddof=1) / expected - 1.0) < 0.05


def test_the_flat_kernel_is_brownian_motion():
    """H = 1/2 makes the kernel identically one, so Y is the driving Brownian
    motion itself. Off by a weight and this fails on every path."""
    y, dW = _volterra(0.5, 64, 200)
    assert np.allclose(y[:, 1:], np.cumsum(dW, axis=1), atol=1e-9)


def test_the_hurst_exponent_is_recovered_from_the_paths():
    """Regressing log variance on log time returns 2H, which is the whole
    content of the word rough."""
    for H in (0.1, 0.3):
        y, _ = _volterra(H, 512, 8_000)
        t = np.arange(1, 513) * (1.0 / 512)
        v = y[:, 1:].var(axis=0, ddof=1)
        slope = np.polyfit(np.log(t), np.log(v), 1)[0]
        assert abs(slope - 2 * H) < 0.02


def test_the_variance_process_has_the_forward_variance_as_its_mean():
    """E[V_t] = xi0 by construction. A missing convexity term shows up here."""
    xi0 = 0.04
    _, v = variance_paths(1.0, 256, SEED, 0, 40_000, xi0, H=0.1, eta=0.5)
    for i in (64, 128, 256):
        assert abs(v[:, i].mean() / xi0 - 1.0) < 0.03


def test_zero_vol_of_vol_recovers_black_scholes():
    xi0, n_paths = 0.04, 60_000
    paths = rbergomi_paths(**MKT, n_mon=256, seed=SEED, path_start=0, n_paths=n_paths,
                           xi0=xi0, H=0.1, eta=0.0, rho=-0.7)
    payoff = np.maximum(paths[:, -1] - 100.0, 0.0) * np.exp(-MKT["r"] * MKT["T"])
    ref = bs_price("call", 100.0, 100.0, 1.0, MKT["r"], MKT["q"], np.sqrt(xi0))
    se = payoff.std(ddof=1) / np.sqrt(n_paths)
    assert abs(payoff.mean() - ref) < 3 * se


def test_the_spot_is_a_martingale_under_the_drift():
    n_paths = 60_000
    paths = rbergomi_paths(**MKT, n_mon=256, seed=SEED, path_start=0, n_paths=n_paths,
                           xi0=0.04, H=0.1, eta=1.5, rho=-0.7)
    ratio = paths[:, -1] / (100.0 * np.exp(MKT["r"] * MKT["T"]))
    se = ratio.std(ddof=1) / np.sqrt(n_paths)
    assert abs(ratio.mean() - 1.0) < 3 * se


def test_paths_are_reproducible_and_path_keyed():
    args = dict(**MKT, n_mon=64, seed=SEED, n_paths=8, xi0=0.04, H=0.1, eta=1.5, rho=-0.7)
    first = rbergomi_paths(path_start=0, **args)
    assert np.array_equal(first, rbergomi_paths(path_start=0, **args))
    assert not np.allclose(first, rbergomi_paths(path_start=8, **args))


def test_rough_variance_is_rougher_than_a_diffusion():
    """The sample paths of V have unbounded variation of a higher order than a
    diffusion: the total absolute increment grows as the grid refines, and it
    grows faster for smaller H."""
    def wiggle(H, n_mon):
        _, v = variance_paths(1.0, n_mon, SEED, 0, 200, 0.04, H=H, eta=1.5)
        return np.abs(np.diff(v, axis=1)).sum(axis=1).mean()

    rough = wiggle(0.1, 1024) / wiggle(0.1, 128)
    smooth = wiggle(0.45, 1024) / wiggle(0.45, 128)
    assert rough > smooth > 1.0


@pytest.mark.slow
def test_the_atm_skew_follows_a_power_law_in_maturity():
    """The reason rough volatility exists: the at-the-money skew explodes as
    T^(H - 1/2) at short maturity, which no diffusive stochastic volatility
    model can produce. Measured by pricing two strikes either side of the money
    at each maturity and differencing the implied volatilities."""
    from vollab.pricing.black_scholes import bs_implied_vol

    H, eta, rho, xi0 = 0.1, 1.9, -0.9, 0.04
    maturities = [0.02, 0.05, 0.1, 0.25, 0.5]
    n_paths, steps_per_year = 80_000, 2_000
    skews = []
    for T in maturities:
        n_mon = max(32, int(steps_per_year * T))
        paths = rbergomi_paths(100.0, 0.0, 0.0, T, n_mon, SEED, 0, n_paths,
                               xi0, H, eta, rho)
        st = paths[:, -1]
        vols = []
        for k in (-0.02, 0.02):
            strike = 100.0 * np.exp(k)
            price = np.maximum(st - strike, 0.0).mean()
            vols.append(bs_implied_vol("call", price, 100.0, strike, T, 0.0, 0.0))
        skews.append(abs(vols[1] - vols[0]) / 0.04)

    slope = np.polyfit(np.log(maturities), np.log(skews), 1)[0]
    # The theoretical exponent is H - 1/2 = -0.4. Monte Carlo on five maturities
    # will not pin it exactly, but a diffusive model would give a slope near
    # zero at these maturities, so the sign and the magnitude are the finding.
    assert -0.55 < slope < -0.20
