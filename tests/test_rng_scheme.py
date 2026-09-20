import numpy as np
import pytest

from vollab.rng.scheme import (
    RNG_SCHEME_VERSION, DRAWS_PER_STEP, normals, normals_block,
)


def test_scheme_version_is_pinned():
    assert RNG_SCHEME_VERSION == 1
    assert DRAWS_PER_STEP == {"gbm": 1, "heston": 3, "merton": 1}


def test_same_key_gives_same_stream():
    assert np.array_equal(normals(7, 3, 64), normals(7, 3, 64))


def test_different_path_gives_different_stream():
    assert not np.array_equal(normals(7, 3, 64), normals(7, 4, 64))


def test_different_seed_gives_different_stream():
    assert not np.array_equal(normals(7, 3, 64), normals(8, 3, 64))


def test_prefix_property():
    """A longer draw extends a shorter one, so Brownian nesting is well defined."""
    assert np.array_equal(normals(7, 3, 16), normals(7, 3, 64)[:16])


def test_block_matches_per_path_calls():
    blk = normals_block(11, 100, 4, 32)
    for i in range(4):
        assert np.array_equal(blk[i], normals(11, 100 + i, 32))


def test_key_must_be_two_words():
    """The constraint that forces the step index into the counter, not the key."""
    with pytest.raises(ValueError):
        np.random.Philox(key=[1, 2, 3])


def test_uniforms_match_the_raw_bit_stream_exactly():
    """This identity is what lets the future C++ engine reproduce the uniform
    stream bit for bit from one line."""
    from vollab.rng.scheme import _uniforms
    u = _uniforms(7, 0, 8)
    raw = np.random.Philox(key=[np.uint64(7), np.uint64(0)]).random_raw(8)
    assert np.array_equal(u, (raw >> np.uint64(11)) * 2.0 ** -53)


def test_distribution_is_standard_normal():
    x = normals_block(0, 0, 2000, 500).ravel()
    assert abs(x.mean()) < 5 * x.std() / np.sqrt(x.size)
    assert abs(x.std(ddof=1) - 1.0) < 0.01
