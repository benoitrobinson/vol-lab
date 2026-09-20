import numpy as np

from vollab.metrics.bootstrap import bootstrap_sd, paired_bootstrap


def test_paired_bootstrap_interval_contains_known_difference():
    base = np.random.default_rng(0).standard_normal(5000)
    mean, lo, hi = paired_bootstrap(base + 0.5, base, n_boot=2000, seed=1)
    assert abs(mean - 0.5) < 1e-12
    assert lo <= 0.5 <= hi
    assert hi - lo < 1e-6          # pairing removes essentially all variance


def test_unpaired_data_gives_a_much_wider_interval():
    rng = np.random.default_rng(0)
    _, lo, hi = paired_bootstrap(rng.standard_normal(5000) + 0.5,
                                 rng.standard_normal(5000), n_boot=2000, seed=1)
    assert hi - lo > 0.01


def test_paired_bootstrap_straddles_zero_for_no_real_difference():
    base = np.random.default_rng(2).standard_normal(5000)
    _, lo, hi = paired_bootstrap(base, base.copy(), n_boot=500, seed=3)
    assert lo <= 0.0 <= hi


def test_bootstrap_sd_recovers_known_sd_with_a_sane_error():
    x = np.random.default_rng(3).normal(0.0, 2.0, 20_000)
    sd, se = bootstrap_sd(x, n_boot=500, seed=2)
    assert abs(sd - 2.0) < 0.05
    assert 0.0 < se < 0.05
    assert abs(sd - 2.0) < 4 * se


def test_bootstrap_sd_error_shrinks_with_sample_size():
    rng = np.random.default_rng(4)
    _, se_small = bootstrap_sd(rng.normal(0, 1, 2_000), n_boot=300, seed=5)
    _, se_big = bootstrap_sd(rng.normal(0, 1, 20_000), n_boot=300, seed=5)
    assert se_big < se_small
