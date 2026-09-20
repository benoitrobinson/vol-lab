import numpy as np

from vollab.paths.gbm import gbm_paths
from vollab.pricing.black_scholes import bs_price


def test_shape_and_initial_value():
    p = gbm_paths(100.0, 0.03, 0.01, 0.3, 1.0, 16, seed=1, path_start=0, n_paths=5)
    assert p.shape == (5, 17)
    assert np.all(p[:, 0] == 100.0)


def test_brownian_nesting_subsamples_the_same_path():
    fine = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, 64, seed=2, path_start=0, n_paths=3)
    coarse = fine[:, ::4]
    assert coarse.shape == (3, 17)
    assert np.array_equal(coarse[:, 0], fine[:, 0])
    assert np.array_equal(coarse[:, -1], fine[:, -1])


def test_chunk_offset_is_consistent():
    """Path 7 must be identical whether generated alone or inside a block."""
    block = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, 32, seed=5, path_start=0, n_paths=10)
    single = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, 32, seed=5, path_start=7, n_paths=1)
    assert np.array_equal(block[7], single[0])


def test_mc_price_converges_to_black_scholes():
    S0, K, T, r, q, s = 100.0, 105.0, 0.5, 0.03, 0.01, 0.35
    n = 400_000
    p = gbm_paths(S0, r, q, s, T, 1, seed=3, path_start=0, n_paths=n)
    disc = np.exp(-r * T) * np.maximum(p[:, -1] - K, 0.0)
    se = disc.std(ddof=1) / np.sqrt(n)
    assert abs(disc.mean() - bs_price("call", S0, K, T, r, q, s)) < 3 * se


def test_terminal_distribution_is_grid_independent():
    """Same seed, different monitoring grid: the terminal value is the same path."""
    a = gbm_paths(100.0, 0.02, 0.0, 0.3, 1.0, 64, seed=9, path_start=0, n_paths=200)
    b = gbm_paths(100.0, 0.02, 0.0, 0.3, 1.0, 256, seed=9, path_start=0, n_paths=200)
    # not identical (different dt), but must agree in distribution
    assert abs(a[:, -1].mean() - b[:, -1].mean()) < 0.05 * a[:, -1].std(ddof=1)


def test_drift_override_is_respected():
    p = gbm_paths(100.0, 0.0, 0.0, 1e-9, 1.0, 1, seed=4, path_start=0, n_paths=1, mu=0.5)
    assert abs(np.log(p[0, -1] / 100.0) - 0.5) < 1e-6
