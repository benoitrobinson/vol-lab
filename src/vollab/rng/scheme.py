"""Counter-based random number scheme.

Keyed per path rather than per draw. The Philox key is exactly two uint64 words,
so a (seed, path, step) key is impossible and the step index lives in the
counter, which advances implicitly as the path's stream is consumed. Per-draw
keying was measured at 355x slower than this and is rejected.

Never call Philox.advance(): it moves whole four-output blocks and discards the
buffer, which silently strides the stream and can overlap paths.
"""

import numpy as np
from scipy.special import ndtri

RNG_SCHEME_VERSION = 1

# Fixed per-step draw budget. Both engines always consume the full budget and
# discard what the taken branch did not use, so a data-dependent branch cannot
# desynchronise the two streams. Merton is 1 because its jump times are keyed
# per path and are therefore independent of the step grid.
DRAWS_PER_STEP = {"gbm": 1, "heston": 3, "merton": 1}


def _uniforms(seed, path_index, n):
    bg = np.random.Philox(key=[np.uint64(seed), np.uint64(path_index)])
    return np.random.Generator(bg).random(n)


def normals(seed, path_index, n):
    """Inverse-CDF normals for one path.

    Prefix stable: asking for more draws extends the same stream rather than
    producing a different one, which is what makes Brownian nesting exact.
    """
    return ndtri(_uniforms(seed, path_index, n))


def normals_block(seed, path_start, n_paths, n):
    out = np.empty((n_paths, n), dtype=np.float64)
    for i in range(n_paths):
        out[i] = normals(seed, path_start + i, n)
    return out
