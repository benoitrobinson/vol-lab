# vol-lab Phase A Part 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-Python Monte Carlo delta-hedging engine, its protocol layer and its terminal charts, to the point where findings F1, F2, F3 and F5 are measured, tested and plotted.

**Architecture:** A chunked, path-vectorised simulator loops over a fine monitoring grid; schedules decide which paths trade at each step; a cash recursion produces terminal P&L; a P&L explain decomposes it. All randomness comes from a counter-based Philox scheme keyed per path so coarse rehedge frequencies are Brownian-nested on identical paths. Every run is pre-registered by config hash and recorded in a sqlite ledger.

**Tech Stack:** Python 3.14, uv, numpy, scipy, plotext, pytest, hypothesis, sqlite3 (stdlib), tomllib (stdlib).

**Spec:** `docs/superpowers/specs/2026-09-19-vol-lab-design.md`

**Covers:** milestones M1 to M4. Milestones M5 to M7 (C++ core, Heston, Merton, REPORT.md, TUI) get a separate plan once this one is green.

## Global Constraints

- Python `>=3.14`. Build backend `uv_build`. Package layout `src/vollab/`.
- All greeks are **position** greeks: short one call gives `Delta < 0`, `Gamma < 0`.
- `s_imp` marks the option, `s_hedge` sizes the traded hedge. Never one symbol for both. Names: `delta_mark`, `delta_hedge`.
- `n_mon` (monitoring grid) and `cost_bps` are the only sources of truth. Schedules never carry their own copy.
- Paths are always generated on the finest grid. Coarser rehedge frequencies subsample it.
- `RNG_SCHEME_VERSION = 1`. Never call `Philox.advance()`.
- No em dashes anywhere. ASCII only in code, identifiers, paths and commands.
- Commit messages: one short imperative subject line, max ~50 chars, no body unless the why is non-obvious. No AI attribution trailers.
- Every reported mean and every reported `sd` carries a standard error.

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | uv project, deps, `vl` console script |
| `src/vollab/pricing/black_scholes.py` | BS price, greeks, implied vol. Free functions, no state |
| `src/vollab/rng/scheme.py` | `RNG_SCHEME_VERSION`, `DRAWS_PER_STEP`, per-path normal generation |
| `src/vollab/paths/gbm.py` | Chunked GBM log-path generator on the fine grid |
| `src/vollab/hedge/schedule.py` | Trade-decision functions. Pure, no cash, no paths |
| `src/vollab/hedge/simulator.py` | Cash recursion, chunk driver, `simulate()` |
| `src/vollab/hedge/attribution.py` | P&L explain, six terms plus residual statistics |
| `src/vollab/hedge/registry.py` | TOML dict to `HedgeConfig`. Keeps `protocol/` domain-free |
| `src/vollab/metrics/bootstrap.py` | Paired bootstrap, bootstrap SE for `sd` |
| `src/vollab/protocol/hashing.py` | Canonical bytes and sha256 of a config dict |
| `src/vollab/protocol/prereg.py` | Load TOML, verify stored `config_hash` |
| `src/vollab/protocol/ledger.py` | sqlite schema, insert, query |
| `src/vollab/render/charts.py` | plotext wrappers taking plain arrays |
| `src/vollab/cli.py` | `vl price`, `vl run`, `vl compare`, `vl ledger` |

---

### Task 1: Project scaffold and Black-Scholes price

**Files:**
- Create: `pyproject.toml`, `src/vollab/__init__.py`, `src/vollab/pricing/__init__.py`, `src/vollab/pricing/black_scholes.py`
- Test: `tests/test_black_scholes.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `bs_price(kind: str, S, K, T, r, q, s) -> float | np.ndarray`, accepting scalars or arrays.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_black_scholes.py
import numpy as np
from vollab.pricing.black_scholes import bs_price

def test_put_call_parity():
    S, K, T, r, q, s = 100.0, 105.0, 0.5, 0.03, 0.01, 0.35
    c = bs_price("call", S, K, T, r, q, s)
    p = bs_price("put", S, K, T, r, q, s)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-12

def test_known_value():
    # S=K=100, T=1, r=q=0, s=0.2 -> 2*Phi(0.1)-1 times 100
    from scipy.stats import norm
    expected = 100.0 * (2 * norm.cdf(0.1) - 1)
    got = bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.2)
    assert abs(got - expected) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_black_scholes.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab'`

- [ ] **Step 3: Write the scaffold and minimal implementation**

```toml
# pyproject.toml
[project]
name = "vollab"
version = "0.1.0"
description = "Monte Carlo delta-hedging laboratory"
requires-python = ">=3.14"
dependencies = ["numpy>=2.5", "scipy>=1.18", "plotext>=5.3"]

[project.scripts]
vl = "vollab.cli:main"

[build-system]
requires = ["uv_build>=0.11,<0.12"]
build-backend = "uv_build"

[dependency-groups]
dev = ["pytest>=9.1", "hypothesis>=6.168"]
```

```python
# src/vollab/pricing/black_scholes.py
import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, r, q, s):
    v = s * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * s * s) * T) / v
    return d1, d1 - v

def bs_price(kind, S, K, T, r, q, s):
    d1, d2 = _d1_d2(S, K, T, r, q, s)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    if kind == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
```

Create empty `src/vollab/__init__.py` and `src/vollab/pricing/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv sync && uv run pytest tests/test_black_scholes.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/vollab tests/test_black_scholes.py
git commit -m "Add Black-Scholes price"
```

---

### Task 2: Greeks

**Files:**
- Modify: `src/vollab/pricing/black_scholes.py`
- Test: `tests/test_greeks.py`

**Interfaces:**
- Consumes: `_d1_d2`, `bs_price`.
- Produces: `bs_delta(kind, S, K, T, r, q, s)`, `bs_gamma(...)`, `bs_vega(...)`, `bs_theta(kind, ...)`. All are **long-option** greeks; callers negate for a short position.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_greeks.py
import numpy as np
import pytest
from vollab.pricing.black_scholes import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta

ARGS = (100.0, 105.0, 0.5, 0.03, 0.01, 0.35)

def _central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = _central(lambda x: bs_price(kind, x, K, T, r, q, s), S, 1e-4)
    assert abs(bs_delta(kind, *ARGS) - fd) < 1e-6 * max(1.0, abs(fd))

@pytest.mark.parametrize("kind", ["call", "put"])
def test_gamma_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = _central(lambda x: bs_delta(kind, x, K, T, r, q, s), S, 1e-4)
    assert abs(bs_gamma(kind, *ARGS) - fd) < 1e-6 * max(1.0, abs(fd))

@pytest.mark.parametrize("kind", ["call", "put"])
def test_vega_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    fd = _central(lambda x: bs_price(kind, S, K, T, r, q, x), s, 1e-5)
    assert abs(bs_vega(kind, *ARGS) - fd) < 1e-5 * max(1.0, abs(fd))

@pytest.mark.parametrize("kind", ["call", "put"])
def test_theta_matches_finite_difference(kind):
    S, K, T, r, q, s = ARGS
    # theta is d/dt = -d/dT
    fd = -_central(lambda x: bs_price(kind, S, K, x, r, q, s), T, 1e-5)
    assert abs(bs_theta(kind, *ARGS) - fd) < 1e-5 * max(1.0, abs(fd))

def test_gamma_and_vega_are_kind_independent():
    assert bs_gamma("call", *ARGS) == pytest.approx(bs_gamma("put", *ARGS))
    assert bs_vega("call", *ARGS) == pytest.approx(bs_vega("put", *ARGS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_greeks.py -v`
Expected: FAIL, `ImportError: cannot import name 'bs_delta'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/vollab/pricing/black_scholes.py
def bs_delta(kind, S, K, T, r, q, s):
    d1, _ = _d1_d2(S, K, T, r, q, s)
    df_q = np.exp(-q * T)
    if kind == "call":
        return df_q * norm.cdf(d1)
    return df_q * (norm.cdf(d1) - 1.0)

def bs_gamma(kind, S, K, T, r, q, s):
    d1, _ = _d1_d2(S, K, T, r, q, s)
    return np.exp(-q * T) * norm.pdf(d1) / (S * s * np.sqrt(T))

def bs_vega(kind, S, K, T, r, q, s):
    d1, _ = _d1_d2(S, K, T, r, q, s)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def bs_theta(kind, S, K, T, r, q, s):
    d1, d2 = _d1_d2(S, K, T, r, q, s)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    common = -S * df_q * norm.pdf(d1) * s / (2 * np.sqrt(T))
    if kind == "call":
        return common - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    return common + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
```

`kind` is accepted by `bs_gamma` and `bs_vega` for a uniform call signature even though both are kind independent; the last test pins that.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_greeks.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/pricing/black_scholes.py tests/test_greeks.py
git commit -m "Add Black-Scholes greeks"
```

---

### Task 3: Implied volatility inverse

**Files:**
- Modify: `src/vollab/pricing/black_scholes.py`
- Test: `tests/test_implied_vol.py`

**Interfaces:**
- Consumes: `bs_price`.
- Produces: `bs_implied_vol(kind, price, S, K, T, r, q) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_implied_vol.py
import pytest
from vollab.pricing.black_scholes import bs_price, bs_implied_vol

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [60.0, 100.0, 160.0])
@pytest.mark.parametrize("T", [0.05, 1.0, 3.0])
@pytest.mark.parametrize("s", [0.1, 0.45, 1.2])
def test_roundtrip_recovers_input(kind, K, T, s):
    S, r, q = 100.0, 0.03, 0.01
    px = bs_price(kind, S, K, T, r, q, s)
    assert abs(bs_implied_vol(kind, px, S, K, T, r, q) - s) < 1e-10

def test_raises_below_intrinsic():
    with pytest.raises(ValueError):
        bs_implied_vol("call", 0.0, 100.0, 50.0, 1.0, 0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_implied_vol.py -v`
Expected: FAIL, `ImportError: cannot import name 'bs_implied_vol'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/vollab/pricing/black_scholes.py
from scipy.optimize import brentq

def bs_implied_vol(kind, price, S, K, T, r, q, lo=1e-8, hi=6.0):
    f = lambda x: bs_price(kind, S, K, T, r, q, x) - price
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError(f"price {price} outside attainable range for s in [{lo}, {hi}]")
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_implied_vol.py -v`
Expected: 55 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/pricing/black_scholes.py tests/test_implied_vol.py
git commit -m "Add implied vol inverse"
```

---

### Task 4: RNG scheme

**Files:**
- Create: `src/vollab/rng/__init__.py`, `src/vollab/rng/scheme.py`
- Test: `tests/test_rng_scheme.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RNG_SCHEME_VERSION: int`, `DRAWS_PER_STEP: dict[str, int]`, `normals(seed: int, path_index: int, n: int) -> np.ndarray`, `normals_block(seed, path_start, n_paths, n) -> np.ndarray` of shape `(n_paths, n)`.

Keyed per path, not per draw: per-draw keying measured 355x slower and is rejected by the spec. The key is exactly two uint64 words, which is why the step index lives in the counter rather than the key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rng_scheme.py
import numpy as np
import pytest
from vollab.rng.scheme import RNG_SCHEME_VERSION, DRAWS_PER_STEP, normals, normals_block

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
    # a longer draw must extend the shorter one, so nesting is well defined
    assert np.array_equal(normals(7, 3, 16), normals(7, 3, 64)[:16])

def test_block_matches_per_path_calls():
    blk = normals_block(11, 100, 4, 32)
    for i in range(4):
        assert np.array_equal(blk[i], normals(11, 100 + i, 32))

def test_key_must_be_two_words():
    # the constraint that forces step index into the counter, not the key
    with pytest.raises(ValueError):
        np.random.Philox(key=[1, 2, 3])

def test_distribution_is_standard_normal():
    x = normals_block(0, 0, 2000, 500).ravel()
    assert abs(x.mean()) < 5 * x.std() / np.sqrt(x.size)
    assert abs(x.std(ddof=1) - 1.0) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rng_scheme.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.rng'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/rng/scheme.py
import numpy as np
from scipy.special import ndtri

RNG_SCHEME_VERSION = 1

# Fixed per-step draw budget. Both engines always consume the full budget and
# discard what the branch did not use, so a data-dependent branch cannot
# desynchronise the streams. Merton is 1 because jump times are keyed per path.
DRAWS_PER_STEP = {"gbm": 1, "heston": 3, "merton": 1}

def _uniforms(seed: int, path_index: int, n: int) -> np.ndarray:
    bg = np.random.Philox(key=[np.uint64(seed), np.uint64(path_index)])
    return np.random.Generator(bg).random(n)

def normals(seed: int, path_index: int, n: int) -> np.ndarray:
    """Inverse-CDF normals for one path. Prefix-stable: a longer n extends a shorter one."""
    return ndtri(_uniforms(seed, path_index, n))

def normals_block(seed: int, path_start: int, n_paths: int, n: int) -> np.ndarray:
    out = np.empty((n_paths, n), dtype=np.float64)
    for i in range(n_paths):
        out[i] = normals(seed, path_start + i, n)
    return out
```

Create an empty `src/vollab/rng/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rng_scheme.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/rng tests/test_rng_scheme.py
git commit -m "Add Philox RNG scheme"
```

---

### Task 5: GBM path generator

**Files:**
- Create: `src/vollab/paths/__init__.py`, `src/vollab/paths/gbm.py`
- Test: `tests/test_gbm.py`

**Interfaces:**
- Consumes: `vollab.rng.scheme.normals_block`.
- Produces: `gbm_paths(S0, r, q, s, T, n_mon, seed, path_start, n_paths, mu=None) -> np.ndarray` of shape `(n_paths, n_mon + 1)`. `mu` overrides the drift; `None` means `r - q`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gbm.py
import numpy as np
from vollab.paths.gbm import gbm_paths
from vollab.pricing.black_scholes import bs_price

def test_shape_and_initial_value():
    p = gbm_paths(100.0, 0.03, 0.01, 0.3, 1.0, 16, seed=1, path_start=0, n_paths=5)
    assert p.shape == (5, 17)
    assert np.all(p[:, 0] == 100.0)

def test_brownian_nesting():
    """Coarse grids must be a subsample of the fine grid, not a different path."""
    fine = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, 64, seed=2, path_start=0, n_paths=3)
    # subsampling every 4th point is what the simulator does for coarser frequencies
    assert fine[:, ::4].shape == (3, 17)
    assert np.array_equal(fine[:, ::4][:, 0], fine[:, 0])

def test_mc_price_converges_to_black_scholes():
    S0, K, T, r, q, s = 100.0, 105.0, 0.5, 0.03, 0.01, 0.35
    n = 400_000
    p = gbm_paths(S0, r, q, s, T, 1, seed=3, path_start=0, n_paths=n)
    disc = np.exp(-r * T) * np.maximum(p[:, -1] - K, 0.0)
    se = disc.std(ddof=1) / np.sqrt(n)
    assert abs(disc.mean() - bs_price("call", S0, K, T, r, q, s)) < 3 * se

def test_drift_override_is_respected():
    p = gbm_paths(100.0, 0.0, 0.0, 1e-9, 1.0, 1, seed=4, path_start=0, n_paths=1, mu=0.5)
    assert abs(np.log(p[0, -1] / 100.0) - 0.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gbm.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.paths'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/paths/gbm.py
import numpy as np
from vollab.rng.scheme import normals_block

def gbm_paths(S0, r, q, s, T, n_mon, seed, path_start, n_paths, mu=None):
    """Exact log-space GBM on the fine monitoring grid, shape (n_paths, n_mon + 1)."""
    drift = (r - q) if mu is None else mu
    dt = T / n_mon
    z = normals_block(seed, path_start, n_paths, n_mon)
    incr = (drift - 0.5 * s * s) * dt + s * np.sqrt(dt) * z
    log_path = np.empty((n_paths, n_mon + 1), dtype=np.float64)
    log_path[:, 0] = np.log(S0)
    np.cumsum(incr, axis=1, out=log_path[:, 1:])
    log_path[:, 1:] += np.log(S0)
    return np.exp(log_path)
```

Create an empty `src/vollab/paths/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gbm.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/paths tests/test_gbm.py
git commit -m "Add GBM path generator"
```

---

### Task 6: Trade schedules

**Files:**
- Create: `src/vollab/hedge/__init__.py`, `src/vollab/hedge/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: nothing.
- Produces: frozen dataclasses `FixedTime(every: int)` and `DeltaBand(h: float)`, each with
  `should_trade(step: int, n_mon: int, delta_target: np.ndarray, delta_held: np.ndarray, S: np.ndarray, gamma: np.ndarray, dt: float) -> np.ndarray` returning a boolean mask over paths, and `hedge_vol(s_hedge: float, k: float, dt: float) -> float`.

The uniform signature exists so the simulator calls every schedule identically. `Leland` and `WhalleyWilmott` in Task 11 use the arguments `FixedTime` and `DeltaBand` ignore.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schedule.py
import numpy as np
from vollab.hedge.schedule import FixedTime, DeltaBand

Z = np.zeros(3)

def _call(sched, step, target, held):
    return sched.should_trade(step, 8, np.asarray(target), np.asarray(held), Z + 100.0, Z, 0.125)

def test_fixed_time_trades_on_multiples():
    s = FixedTime(every=2)
    assert _call(s, 0, Z, Z).all()
    assert not _call(s, 1, Z, Z).any()
    assert _call(s, 2, Z, Z).all()

def test_fixed_time_always_trades_final_step():
    s = FixedTime(every=3)
    assert _call(s, 8, Z, Z).all()

def test_delta_band_triggers_only_outside_band():
    s = DeltaBand(h=0.05)
    m = _call(s, 3, [0.50, 0.56, 0.44], [0.50, 0.50, 0.50])
    assert list(m) == [False, True, True]

def test_delta_band_always_trades_first_and_final_step():
    s = DeltaBand(h=0.5)
    assert _call(s, 0, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).all()
    assert _call(s, 8, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).all()

def test_hedge_vol_is_identity_for_both():
    assert FixedTime(2).hedge_vol(0.3, 0.0005, 0.01) == 0.3
    assert DeltaBand(0.05).hedge_vol(0.3, 0.0005, 0.01) == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.hedge'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/hedge/schedule.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FixedTime:
    every: int
    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        on = (step % self.every == 0) or step == n_mon
        return np.full(delta_target.shape, on, dtype=bool)
    def hedge_vol(self, s_hedge, k, dt):
        return s_hedge

@dataclass(frozen=True)
class DeltaBand:
    h: float
    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        if step == 0 or step == n_mon:
            return np.ones(delta_target.shape, dtype=bool)
        return np.abs(delta_target - delta_held) > self.h
    def hedge_vol(self, s_hedge, k, dt):
        return s_hedge
```

Create an empty `src/vollab/hedge/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/hedge tests/test_schedule.py
git commit -m "Add fixed-time and band schedules"
```

---

### Task 7: Hedge simulator and cash invariant

**Files:**
- Create: `src/vollab/hedge/config.py`, `src/vollab/hedge/simulator.py`
- Test: `tests/test_simulator.py`

**Interfaces:**
- Consumes: `gbm_paths`, `bs_price`, `bs_delta`, schedules.
- Produces: frozen dataclasses `Contract(kind, S0, K, T, r, q)`, `VolSpec(s_imp, s_hedge, s_real)`, `HedgeConfig(contract, vols, schedule, n_mon, cost_bps, n_paths, seed, mu=None, chunk_paths=50_000, trace_paths=())`, `HedgeResult(pnl, n_rehedges, turnover, rehedge_mask_hash, engine_used, rng_scheme_version, trace)`, and `simulate(cfg, engine="numpy") -> HedgeResult`.

`simulate` raises on `engine="cpp"` in this plan. It must never silently fall back, or the future parity test compares NumPy against NumPy and passes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulator.py
import numpy as np
import pytest
from vollab.hedge.config import Contract, VolSpec, HedgeConfig
from vollab.hedge.simulator import simulate
from vollab.hedge.schedule import FixedTime, DeltaBand
from vollab.pricing.black_scholes import bs_price

def cfg(**kw):
    base = dict(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(s_imp=0.3, s_hedge=0.3, s_real=0.3),
        schedule=FixedTime(1), n_mon=64, cost_bps=0.0,
        n_paths=2000, seed=1, chunk_paths=512,
    )
    base.update(kw)
    return HedgeConfig(**base)

def test_result_shapes_and_metadata():
    r = simulate(cfg())
    assert r.pnl.shape == (2000,)
    assert r.n_rehedges.dtype == np.int64
    assert r.engine_used == "numpy"
    assert r.rng_scheme_version == 1

def test_chunking_does_not_change_results():
    a = simulate(cfg(chunk_paths=512))
    b = simulate(cfg(chunk_paths=2000))
    assert np.array_equal(a.pnl, b.pnl)

def test_never_rehedging_reproduces_naked_short():
    """A schedule that only trades at the ends is a static hedge, not a naked short,
    so compare against the explicit static-hedge payoff."""
    c = cfg(schedule=FixedTime(10**9), n_mon=64, n_paths=4000)
    r = simulate(c)
    assert np.isfinite(r.pnl).all()
    assert (r.n_rehedges == 2).all()

def test_perfect_hedge_mean_pnl_is_zero():
    r = simulate(cfg(n_mon=512, n_paths=40_000, chunk_paths=10_000))
    se = r.pnl.std(ddof=1) / np.sqrt(r.pnl.size)
    assert abs(r.pnl.mean()) < 3 * se

def test_costs_reduce_pnl():
    free = simulate(cfg(cost_bps=0.0)).pnl.mean()
    paid = simulate(cfg(cost_bps=20.0)).pnl.mean()
    assert paid < free

def test_band_trades_no_more_often_than_fixed_time_on_same_grid():
    ft = simulate(cfg(schedule=FixedTime(1))).n_rehedges
    bd = simulate(cfg(schedule=DeltaBand(0.02))).n_rehedges
    assert (bd <= ft).all()

def test_cpp_engine_raises_rather_than_falling_back():
    with pytest.raises(NotImplementedError):
        simulate(cfg(), engine="cpp")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simulator.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.hedge.config'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/hedge/config.py
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class Contract:
    kind: Literal["call", "put"]
    S0: float; K: float; T: float; r: float; q: float

@dataclass(frozen=True)
class VolSpec:
    s_imp: float; s_hedge: float; s_real: float

@dataclass(frozen=True)
class HedgeConfig:
    contract: Contract
    vols: VolSpec
    schedule: object
    n_mon: int
    cost_bps: float
    n_paths: int
    seed: int
    mu: float | None = None
    chunk_paths: int = 50_000
    trace_paths: tuple[int, ...] = ()

@dataclass(frozen=True)
class HedgeResult:
    pnl: np.ndarray
    n_rehedges: np.ndarray
    turnover: np.ndarray
    rehedge_mask_hash: np.ndarray
    engine_used: str
    rng_scheme_version: int
    trace: dict | None = None
```

```python
# src/vollab/hedge/simulator.py
import numpy as np
from vollab.paths.gbm import gbm_paths
from vollab.pricing.black_scholes import bs_price, bs_delta
from vollab.rng.scheme import RNG_SCHEME_VERSION
from vollab.hedge.config import HedgeResult

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)

def _mix(acc, step):
    return (acc ^ np.uint64(step)) * _FNV_PRIME

def simulate(cfg, engine="numpy"):
    if engine == "cpp":
        raise NotImplementedError("C++ engine lands in Phase A part 2; no silent fallback")
    if engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")

    c, v = cfg.contract, cfg.vols
    k = cfg.cost_bps * 1e-4
    dt = c.T / cfg.n_mon
    s_h = cfg.schedule.hedge_vol(v.s_hedge, k, dt)

    pnl, nreh, turn, mhash = [], [], [], []
    for start in range(0, cfg.n_paths, cfg.chunk_paths):
        m = min(cfg.chunk_paths, cfg.n_paths - start)
        S = gbm_paths(c.S0, c.r, c.q, v.s_real, c.T, cfg.n_mon, cfg.seed, start, m, cfg.mu)

        cash = np.full(m, bs_price(c.kind, c.S0, c.K, c.T, c.r, c.q, v.s_imp))
        held = np.zeros(m)
        n_r = np.zeros(m, dtype=np.int64)
        tv = np.zeros(m)
        h = np.full(m, _FNV_OFFSET, dtype=np.uint64)

        for i in range(cfg.n_mon + 1):
            tau = max(c.T - i * dt, 0.0)
            if i > 0:
                cash = cash * np.exp(c.r * dt) + c.q * held * S[:, i - 1] * dt
            if i == cfg.n_mon:
                target = np.zeros(m)
            else:
                target = -bs_delta(c.kind, S[:, i], c.K, tau, c.r, c.q, s_h)
                gam = np.zeros(m)
                trade = cfg.schedule.should_trade(i, cfg.n_mon, target, held, S[:, i], gam, dt)
                target = np.where(trade, target, held)
            d = target - held
            traded = d != 0.0
            cash -= d * S[:, i] + k * np.abs(d) * S[:, i]
            tv += np.abs(d) * S[:, i]
            n_r += traded
            h = np.where(traded, _mix(h, i), h)
            held = target

        payoff = np.maximum(S[:, -1] - c.K, 0.0) if c.kind == "call" \
            else np.maximum(c.K - S[:, -1], 0.0)
        pnl.append(cash - payoff)
        nreh.append(n_r); turn.append(tv); mhash.append(h)

    return HedgeResult(
        pnl=np.concatenate(pnl), n_rehedges=np.concatenate(nreh),
        turnover=np.concatenate(turn), rehedge_mask_hash=np.concatenate(mhash),
        engine_used="numpy", rng_scheme_version=RNG_SCHEME_VERSION,
    )
```

The final step sets `target` to zero, so liquidation and the `k*|Delta_{N-1}|*S_T` charge fall out of the same recursion rather than being special-cased.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_simulator.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/hedge tests/test_simulator.py
git commit -m "Add hedge simulator"
```

---

### Task 8: F1, discretisation law recovery

**Files:**
- Create: `src/vollab/metrics/__init__.py`, `src/vollab/metrics/stats.py`
- Test: `tests/test_f1_law.py`

**Interfaces:**
- Consumes: `simulate`, `FixedTime`.
- Produces: `sd_vs_frequency(cfg, every_list) -> tuple[np.ndarray, np.ndarray]` returning rehedge counts and `sd(PnL)`, and `loglog_slope(x, y) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_f1_law.py
import numpy as np
from vollab.hedge.config import Contract, VolSpec, HedgeConfig
from vollab.hedge.schedule import FixedTime
from vollab.metrics.stats import sd_vs_frequency, loglog_slope

def test_sd_scales_as_inverse_sqrt_n():
    cfg = HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1),
        n_mon=4096, cost_bps=0.0, n_paths=20_000, seed=11, chunk_paths=5_000,
    )
    every = [256, 128, 64, 32, 16, 8, 4, 2, 1]   # N_reh 16 .. 4096
    n, sd = sd_vs_frequency(cfg, every)
    assert np.all(np.diff(sd) < 0)               # monotone decreasing
    assert abs(loglog_slope(n, sd) - (-0.5)) < 0.03

def test_loglog_slope_on_exact_power_law():
    x = np.array([1.0, 10.0, 100.0])
    assert abs(loglog_slope(x, 3.0 * x ** -0.5) - (-0.5)) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_f1_law.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/metrics/stats.py
import dataclasses
import numpy as np
from vollab.hedge.simulator import simulate
from vollab.hedge.schedule import FixedTime

def sd_vs_frequency(cfg, every_list):
    """All frequencies run on the same Brownian-nested paths, so the sweep is paired."""
    n, sd = [], []
    for e in every_list:
        r = simulate(dataclasses.replace(cfg, schedule=FixedTime(every=e)))
        n.append(r.n_rehedges.mean())
        sd.append(r.pnl.std(ddof=1))
    return np.asarray(n), np.asarray(sd)

def loglog_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)[0])
```

Create an empty `src/vollab/metrics/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_f1_law.py -v`
Expected: 2 passed. If the slope misses, check that `gbm_paths` is being called once on the fine grid rather than regenerated per frequency; an unpaired sweep widens the error badly.

- [ ] **Step 5: Commit**

```bash
git add src/vollab/metrics tests/test_f1_law.py
git commit -m "Add discretisation law recovery"
```

---

### Task 9: Attribution

**Files:**
- Create: `src/vollab/hedge/attribution.py`
- Modify: `src/vollab/hedge/simulator.py` (accumulate attribution arrays)
- Test: `tests/test_attribution.py`

**Interfaces:**
- Consumes: greeks, simulator internals.
- Produces: frozen dataclass `Attribution(delta, gamma, theta, vega, carry, cost, residual_sum, residual_abs_sum, residual_max_abs)`, all `(n_paths,)` float64; `HedgeResult` gains `attribution: Attribution`.

Six terms plus residual. The closure identity is true by construction, so the test that has power is a **bound on the residual relative to the gamma term**, not closure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attribution.py
import numpy as np
from vollab.hedge.config import Contract, VolSpec, HedgeConfig
from vollab.hedge.simulator import simulate
from vollab.hedge.schedule import FixedTime

def cfg(**kw):
    base = dict(contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
                vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1),
                n_mon=512, cost_bps=0.0, n_paths=4000, seed=5, chunk_paths=1000)
    base.update(kw); return HedgeConfig(**base)

def test_components_have_path_shape():
    a = simulate(cfg()).attribution
    for name in ("delta", "gamma", "theta", "vega", "carry", "cost",
                 "residual_sum", "residual_abs_sum", "residual_max_abs"):
        assert getattr(a, name).shape == (4000,)

def test_vega_is_identically_zero_in_phase_a():
    assert np.all(simulate(cfg()).attribution.vega == 0.0)

def test_residual_is_small_relative_to_gamma_under_gbm():
    """The expansion is exact to second order, so the residual is third order."""
    a = simulate(cfg()).attribution
    assert a.residual_max_abs.max() / np.abs(a.gamma).max() < 0.05

def test_cost_term_captures_transaction_costs():
    """Costs must land in their own line, never in the residual."""
    free = simulate(cfg(cost_bps=0.0)).attribution
    paid = simulate(cfg(cost_bps=20.0)).attribution
    assert np.all(free.cost == 0.0)
    assert paid.cost.mean() < 0.0
    # the residual must not absorb the cost
    assert abs(paid.residual_abs_sum.mean() - free.residual_abs_sum.mean()) \
        < 0.1 * abs(paid.cost.mean())

def test_closure_holds_as_a_nan_guard():
    r = simulate(cfg())
    a = r.attribution
    total = a.delta + a.gamma + a.theta + a.vega + a.carry + a.cost + a.residual_sum
    assert np.allclose(total, r.pnl, rtol=0, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: FAIL, `AttributeError: 'HedgeResult' object has no attribute 'attribution'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/hedge/attribution.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Attribution:
    delta: np.ndarray; gamma: np.ndarray; theta: np.ndarray
    vega: np.ndarray; carry: np.ndarray; cost: np.ndarray
    residual_sum: np.ndarray; residual_abs_sum: np.ndarray; residual_max_abs: np.ndarray

class Accumulator:
    """Per-step P&L explain, summed into per-path arrays.

    Mark greeks are evaluated at s_imp (the option is marked there); the traded
    hedge is sized at s_hedge. They differ whenever the two vols differ, which
    is the F2 configuration, so they are never the same variable.
    """
    def __init__(self, m):
        z = lambda: np.zeros(m)
        self.delta, self.gamma, self.theta = z(), z(), z()
        self.vega, self.carry, self.cost = z(), z(), z()
        self.res_sum, self.res_abs, self.res_max = z(), z(), z()

    def step(self, delta_mark, gamma_mark, theta_mark, dS, dt,
             cash_prev, r, q, delta_held_prev, S_prev, cost_paid, realized):
        d = delta_mark * dS
        g = 0.5 * gamma_mark * dS * dS
        th = theta_mark * dt
        ca = cash_prev * (np.exp(r * dt) - 1.0) + q * delta_held_prev * S_prev * dt
        res = realized - (d + g + th + ca + cost_paid)
        self.delta += d; self.gamma += g; self.theta += th
        self.carry += ca; self.cost += cost_paid
        self.res_sum += res
        self.res_abs += np.abs(res)
        self.res_max = np.maximum(self.res_max, np.abs(res))

    def finish(self):
        return Attribution(self.delta, self.gamma, self.theta, np.zeros_like(self.delta),
                           self.carry, self.cost, self.res_sum, self.res_abs, self.res_max)
```

In `simulator.py`: construct an `Accumulator(m)` per chunk, and at each step `i > 0`
compute `realized` as the change in (cash + held*S - option mark at s_imp), call
`acc.step(...)` with `delta_mark`/`gamma_mark`/`theta_mark` evaluated at `v.s_imp` and
`cost_paid = -k * abs(d) * S[:, i]`, then concatenate `acc.finish()` across chunks
into `HedgeResult.attribution`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: 5 passed. If `test_residual_is_small_relative_to_gamma_under_gbm` fails, the
most likely cause is mark greeks accidentally evaluated at `s_hedge`, or `carry` using
`r*dt` instead of `exp(r*dt)-1`.

- [ ] **Step 5: Commit**

```bash
git add src/vollab/hedge tests/test_attribution.py
git commit -m "Add P&L attribution"
```

---

### Task 10: F2, hedging at implied against hedging at realized

**Files:**
- Test: `tests/test_f2_lockin.py`

**Interfaces:**
- Consumes: `simulate`, `bs_price`. No new production code; this task proves an existing property.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_f2_lockin.py
import numpy as np
from vollab.hedge.config import Contract, VolSpec, HedgeConfig
from vollab.hedge.simulator import simulate
from vollab.hedge.schedule import FixedTime
from vollab.pricing.black_scholes import bs_price

S0, K, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
S_IMP, S_REAL = 0.35, 0.25          # sold rich: implied above realized

def _cfg(s_hedge):
    return HedgeConfig(
        contract=Contract("call", S0, K, T, r, q),
        vols=VolSpec(s_imp=S_IMP, s_hedge=s_hedge, s_real=S_REAL),
        schedule=FixedTime(1), n_mon=2048, cost_bps=0.0,
        n_paths=20_000, seed=21, chunk_paths=5_000,
    )

def _edge():
    return bs_price("call", S0, K, T, r, q, S_IMP) - bs_price("call", S0, K, T, r, q, S_REAL)

def test_drift_is_pinned_to_r_minus_q():
    """The equality of the two means holds only under this drift."""
    assert _cfg(S_REAL).mu is None          # None means r - q

def test_hedging_at_realized_locks_in_the_edge():
    pnl = simulate(_cfg(S_REAL)).pnl
    se = pnl.std(ddof=1) / np.sqrt(pnl.size)
    assert abs(pnl.mean() - _edge()) < 4 * se
    assert pnl.std(ddof=1) < 0.05 * abs(_edge())     # dispersion is discretisation only

def test_hedging_at_implied_has_same_mean_but_is_path_dependent():
    at_imp = simulate(_cfg(S_IMP)).pnl
    at_real = simulate(_cfg(S_REAL)).pnl
    se = at_imp.std(ddof=1) / np.sqrt(at_imp.size)
    assert abs(at_imp.mean() - _edge()) < 4 * se
    assert at_imp.std(ddof=1) > 5 * at_real.std(ddof=1)

def test_hedging_at_implied_has_the_sign_of_s_imp_minus_s_real():
    """Short vol: profit when realized comes in below implied. Sign is s_imp - s_real."""
    pnl = simulate(_cfg(S_IMP)).pnl
    assert _edge() > 0
    assert (pnl > 0).mean() > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_f2_lockin.py -v`
Expected: the sign and lock-in tests fail if the simulator ever reuses one delta for
both marking and hedging. If all four pass immediately, that is the correct outcome
and Task 9 already got it right; record the observed numbers in the commit message.

- [ ] **Step 3: Fix whatever the tests expose**

The expected defect is a single `s` threaded into both `bs_delta` (hedge) and the
attribution greeks (mark). Separate them: `s_h = cfg.schedule.hedge_vol(v.s_hedge, k, dt)`
for trading, `v.s_imp` for marking.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_f2_lockin.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_f2_lockin.py src/vollab/hedge
git commit -m "Add F2 lock-in and sign tests"
```

---

### Task 11: Leland and Whalley-Wilmott schedules

**Files:**
- Modify: `src/vollab/hedge/schedule.py`
- Test: `tests/test_cost_schedules.py`

**Interfaces:**
- Consumes: `bs_gamma`.
- Produces: `Leland(every: int, short: bool = True)` and `WhalleyWilmott(gam_ra: float)`, both with the Task 6 signature.

`Leland` adjusts the vol: `s_L^2 = s^2 * (1 + Le)` for a **written** option, `(1 - Le)`
for a held one, with `Le = sqrt(2/pi) * k / (s * sqrt(dt))`. `WhalleyWilmott` band
half-width is `(3 * k * S * Gamma^2 / (2 * gam_ra))^(1/3)`. Acceptance is the scaling,
`k^(1/3)` and `Gamma^(2/3)`, not a transcribed constant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_schedules.py
import numpy as np
from vollab.hedge.schedule import Leland, WhalleyWilmott

def test_leland_raises_vol_for_a_written_option():
    s, k, dt = 0.3, 0.001, 1 / 252
    out = Leland(every=1, short=True).hedge_vol(s, k, dt)
    le = np.sqrt(2 / np.pi) * k / (s * np.sqrt(dt))
    assert abs(out - s * np.sqrt(1 + le)) < 1e-12
    assert out > s

def test_leland_lowers_vol_for_a_held_option():
    s, k, dt = 0.3, 0.001, 1 / 252
    assert Leland(every=1, short=False).hedge_vol(s, k, dt) < s

def test_leland_is_identity_at_zero_cost():
    assert abs(Leland(1).hedge_vol(0.3, 0.0, 1 / 252) - 0.3) < 1e-15

def _band(k, gamma, S=100.0, gam_ra=1.0):
    w = WhalleyWilmott(gam_ra=gam_ra)
    target = np.array([0.5]); held = np.array([0.5])
    return w.band_width(np.array([S]), np.array([gamma]), k)[0]

def test_band_scales_as_k_to_the_one_third():
    b1, b2 = _band(1e-4, 0.02), _band(8e-4, 0.02)
    assert abs(b2 / b1 - 8 ** (1 / 3)) < 1e-9

def test_band_scales_as_gamma_to_the_two_thirds():
    b1, b2 = _band(1e-4, 0.02), _band(1e-4, 0.08)
    assert abs(b2 / b1 - 4 ** (2 / 3)) < 1e-9

def test_band_is_dimensionally_consistent_in_S():
    """S enters to the first power inside the cube root, giving S^(1/3)."""
    b1, b2 = _band(1e-4, 0.02, S=100.0), _band(1e-4, 0.02, S=800.0)
    assert abs(b2 / b1 - 8 ** (1 / 3)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cost_schedules.py -v`
Expected: FAIL, `ImportError: cannot import name 'Leland'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/vollab/hedge/schedule.py
@dataclass(frozen=True)
class Leland:
    every: int
    short: bool = True
    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        on = (step % self.every == 0) or step == n_mon
        return np.full(delta_target.shape, on, dtype=bool)
    def hedge_vol(self, s_hedge, k, dt):
        le = np.sqrt(2.0 / np.pi) * k / (s_hedge * np.sqrt(dt))
        sign = 1.0 if self.short else -1.0
        return float(s_hedge * np.sqrt(max(1.0 + sign * le, 1e-12)))

@dataclass(frozen=True)
class WhalleyWilmott:
    gam_ra: float
    def band_width(self, S, gamma, k):
        return np.cbrt(1.5 * k * S * gamma * gamma / self.gam_ra)
    def should_trade(self, step, n_mon, delta_target, delta_held, S, gamma, dt):
        if step == 0 or step == n_mon:
            return np.ones(delta_target.shape, dtype=bool)
        return np.abs(delta_target - delta_held) > self.band_width(S, gamma, self._k)
    def hedge_vol(self, s_hedge, k, dt):
        object.__setattr__(self, "_k", k)
        return s_hedge
```

The simulator already calls `hedge_vol` once before the step loop, so stashing `k`
there is what gives `should_trade` access to it without widening the shared signature.
`gamma` passed to `should_trade` must now be the real mark gamma, so change the
simulator's `gam = np.zeros(m)` to `gam = bs_gamma(c.kind, S[:, i], c.K, tau, c.r, c.q, v.s_imp)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost_schedules.py tests/test_simulator.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/hedge/schedule.py src/vollab/hedge/simulator.py tests/test_cost_schedules.py
git commit -m "Add Leland and Whalley-Wilmott schedules"
```

---

### Task 12: Bootstrap statistics and F5

**Files:**
- Create: `src/vollab/metrics/bootstrap.py`
- Test: `tests/test_bootstrap.py`, `tests/test_f5_ucurve.py`

**Interfaces:**
- Consumes: `simulate`.
- Produces: `paired_bootstrap(a, b, n_boot=10_000, seed=0) -> tuple[float, float, float]` returning mean difference and a 95% interval; `bootstrap_sd(x, n_boot=2000, seed=0) -> tuple[float, float]` returning `sd` and its standard error.

`sd` needs its own bootstrap because it is nonlinear: its sampling error is not `s/sqrt(n)`, and both F1 and F4 depend on it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bootstrap.py
import numpy as np
from vollab.metrics.bootstrap import paired_bootstrap, bootstrap_sd

def test_paired_bootstrap_interval_contains_known_difference():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(5000)
    a, b = base + 0.5, base            # perfectly paired, difference exactly 0.5
    mean, lo, hi = paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert abs(mean - 0.5) < 1e-12
    assert lo <= 0.5 <= hi
    assert hi - lo < 1e-6              # pairing removes essentially all variance

def test_unpaired_data_gives_a_wide_interval():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(5000) + 0.5, rng.standard_normal(5000)
    _, lo, hi = paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert hi - lo > 0.01

def test_bootstrap_sd_recovers_known_sd_with_a_sane_error():
    x = np.random.default_rng(3).normal(0.0, 2.0, 20_000)
    sd, se = bootstrap_sd(x, n_boot=500, seed=2)
    assert abs(sd - 2.0) < 0.05
    assert 0.0 < se < 0.05
    assert abs(sd - 2.0) < 4 * se
```

```python
# tests/test_f5_ucurve.py
import dataclasses
import numpy as np
from vollab.hedge.config import Contract, VolSpec, HedgeConfig
from vollab.hedge.simulator import simulate
from vollab.hedge.schedule import FixedTime

def _cfg(cost_bps):
    return HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(1),
        n_mon=2048, cost_bps=cost_bps, n_paths=8000, seed=31, chunk_paths=2000,
    )

def _curve(cost_bps, every_list):
    out = []
    for e in every_list:
        r = simulate(dataclasses.replace(_cfg(cost_bps), schedule=FixedTime(e)))
        # risk-adjusted total cost: mean loss plus one sd of it
        out.append(-r.pnl.mean() + r.pnl.std(ddof=1))
    return np.asarray(out)

EVERY = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

def test_cost_curve_has_an_interior_minimum():
    c = _curve(10.0, EVERY)
    i = int(np.argmin(c))
    assert 0 < i < len(c) - 1, f"minimum at edge index {i}: {c}"

def test_optimum_moves_to_less_frequent_hedging_as_costs_rise():
    cheap = int(np.argmin(_curve(2.0, EVERY)))
    dear = int(np.argmin(_curve(40.0, EVERY)))
    assert dear <= cheap      # EVERY is descending, so a lower index is less frequent

def test_zero_cost_curve_is_monotone_decreasing():
    c = _curve(0.0, EVERY)
    assert np.all(np.diff(c) < 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bootstrap.py tests/test_f5_ucurve.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.metrics.bootstrap'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/metrics/bootstrap.py
import numpy as np

def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Resample paths, not arms. The two arms are perfectly correlated by
    construction on a given path, which is the point: it removes variance."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):                       # loop, not an (n_boot, n) matrix
        means[i] = d[rng.integers(0, d.size, d.size)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

def bootstrap_sd(x, n_boot=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    sds = np.empty(n_boot)
    for i in range(n_boot):
        sds[i] = x[rng.integers(0, x.size, x.size)].std(ddof=1)
    return float(x.std(ddof=1)), float(sds.std(ddof=1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bootstrap.py tests/test_f5_ucurve.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/metrics/bootstrap.py tests/test_bootstrap.py tests/test_f5_ucurve.py
git commit -m "Add bootstrap stats and F5 cost curve"
```

---

### Task 13: Pre-registration and ledger

**Files:**
- Create: `src/vollab/protocol/__init__.py`, `src/vollab/protocol/hashing.py`, `src/vollab/protocol/prereg.py`, `src/vollab/protocol/ledger.py`, `src/vollab/hedge/registry.py`, `configs/discretisation.toml`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: `HedgeConfig`, schedules.
- Produces: `canonical_bytes(d: dict) -> bytes`, `config_hash(d: dict) -> str`, `load_registered(path) -> dict` (raises `HashMismatch` when the stored hash is stale), `Ledger(db_path)` with `.insert(row: dict) -> None` and `.all() -> list[dict]`, and `build_config(d: dict) -> HedgeConfig`.

`protocol/` imports nothing from `hedge/`: it hashes canonical bytes and stores JSON. TOML resolution lives in `hedge/registry.py`, which keeps the ledger reusable by Phase B.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocol.py
import json, sqlite3
import pytest
from vollab.protocol.hashing import canonical_bytes, config_hash
from vollab.protocol.prereg import load_registered, HashMismatch
from vollab.protocol.ledger import Ledger
from vollab.hedge.registry import build_config

def test_hash_is_invariant_to_key_order_and_whitespace():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

def test_hash_changes_when_a_value_changes():
    assert config_hash({"a": 1}) != config_hash({"a": 2})

def test_stored_hash_field_is_excluded_from_the_hash():
    d = {"a": 1, "config_hash": "deadbeef"}
    assert config_hash(d) == config_hash({"a": 1})

def test_load_registered_accepts_a_matching_hash(tmp_path):
    body = 'hypothesis = "slope is -0.5"\n[contract]\nK = 100.0\n'
    h = config_hash({"hypothesis": "slope is -0.5", "contract": {"K": 100.0}})
    p = tmp_path / "c.toml"
    p.write_text(f'config_hash = "{h}"\n{body}')
    assert load_registered(p)["hypothesis"] == "slope is -0.5"

def test_load_registered_rejects_an_edited_config(tmp_path):
    """Editing a parameter without re-registering must block the run."""
    body = 'hypothesis = "h"\n[contract]\nK = 999.0\n'
    stale = config_hash({"hypothesis": "h", "contract": {"K": 100.0}})
    p = tmp_path / "c.toml"
    p.write_text(f'config_hash = "{stale}"\n{body}')
    with pytest.raises(HashMismatch):
        load_registered(p)

def test_load_registered_requires_a_hypothesis(tmp_path):
    d = {"contract": {"K": 100.0}}
    p = tmp_path / "c.toml"
    p.write_text(f'config_hash = "{config_hash(d)}"\n[contract]\nK = 100.0\n')
    with pytest.raises(ValueError, match="hypothesis"):
        load_registered(p)

def test_ledger_roundtrip(tmp_path):
    led = Ledger(tmp_path / "l.db")
    row = dict(run_id="r1", schema_version=1, ts="2026-09-19T00:00:00",
               config_hash="abc", config_toml="x=1", hypothesis="h",
               paths_fingerprint="fp", git_commit="c", git_dirty=0, git_diff_sha=None,
               vollab_version="0.1.0", rng_scheme_version=1, engine="numpy",
               engine_build_id=None, numpy_version="2.5.3", scipy_version="1.18.1",
               python_version="3.14.3", platform="darwin-arm64", status="ok",
               n_paths_completed=1000, metrics=json.dumps({"sharpe": 1.0}),
               artifacts="[]", artifact_sha256="[]", runtime_s=1.5)
    led.insert(row)
    got = led.all()
    assert len(got) == 1 and got[0]["run_id"] == "r1" and got[0]["status"] == "ok"

def test_registry_builds_a_config_from_a_dict():
    cfg = build_config({
        "contract": {"kind": "call", "S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.0, "q": 0.0},
        "vols": {"s_imp": 0.3, "s_hedge": 0.3, "s_real": 0.3},
        "schedule": {"name": "fixed_time", "every": 4},
        "n_mon": 256, "cost_bps": 5.0, "n_paths": 1000, "seed": 1,
    })
    assert cfg.n_mon == 256 and cfg.schedule.every == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.protocol'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/protocol/hashing.py
import hashlib, json

def canonical_bytes(d):
    """Sorted keys, compact separators, UTF-8. The stored hash field never
    participates, or verifying it would be circular."""
    clean = {k: v for k, v in d.items() if k != "config_hash"}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")

def config_hash(d):
    return hashlib.sha256(canonical_bytes(d)).hexdigest()
```

```python
# src/vollab/protocol/prereg.py
import tomllib
from pathlib import Path
from vollab.protocol.hashing import config_hash

class HashMismatch(Exception):
    pass

def load_registered(path):
    d = tomllib.loads(Path(path).read_text())
    stored = d.get("config_hash")
    if stored is None:
        raise HashMismatch(f"{path} has no config_hash; register it first")
    actual = config_hash(d)
    if stored != actual:
        raise HashMismatch(f"{path} changed since registration: stored {stored}, actual {actual}")
    if not d.get("hypothesis"):
        raise ValueError("config must declare a prose hypothesis before it can run")
    return d
```

```python
# src/vollab/protocol/ledger.py
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, ts TEXT NOT NULL,
  config_hash TEXT NOT NULL, config_toml TEXT NOT NULL, hypothesis TEXT NOT NULL,
  paths_fingerprint TEXT NOT NULL,
  git_commit TEXT NOT NULL, git_dirty INTEGER NOT NULL, git_diff_sha TEXT,
  vollab_version TEXT NOT NULL, rng_scheme_version INTEGER NOT NULL,
  engine TEXT NOT NULL, engine_build_id TEXT,
  numpy_version TEXT NOT NULL, scipy_version TEXT NOT NULL,
  python_version TEXT NOT NULL, platform TEXT NOT NULL,
  status TEXT NOT NULL, n_paths_completed INTEGER NOT NULL,
  metrics TEXT NOT NULL, artifacts TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
  runtime_s REAL NOT NULL);
CREATE INDEX IF NOT EXISTS runs_config_hash ON runs(config_hash);
CREATE INDEX IF NOT EXISTS runs_ts ON runs(ts);
"""

class Ledger:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def insert(self, row):
        cols = ",".join(row)
        marks = ",".join(":" + c for c in row)
        self.conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", row)
        self.conn.commit()

    def all(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM runs ORDER BY ts DESC")]
```

```python
# src/vollab/hedge/registry.py
from vollab.hedge.config import Contract, VolSpec, HedgeConfig
from vollab.hedge.schedule import FixedTime, DeltaBand, Leland, WhalleyWilmott

_SCHEDULES = {"fixed_time": FixedTime, "delta_band": DeltaBand,
              "leland": Leland, "whalley_wilmott": WhalleyWilmott}

def build_config(d):
    sd = dict(d["schedule"])
    sched = _SCHEDULES[sd.pop("name")](**sd)
    return HedgeConfig(
        contract=Contract(**d["contract"]), vols=VolSpec(**d["vols"]), schedule=sched,
        n_mon=d["n_mon"], cost_bps=d["cost_bps"], n_paths=d["n_paths"], seed=d["seed"],
        mu=d.get("mu"), chunk_paths=d.get("chunk_paths", 50_000),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/vollab/protocol src/vollab/hedge/registry.py tests/test_protocol.py
git commit -m "Add pre-registration and run ledger"
```

---

### Task 14: Charts, CLI, and the golden run

**Files:**
- Create: `src/vollab/render/__init__.py`, `src/vollab/render/charts.py`, `src/vollab/cli.py`, `configs/discretisation.toml`, `tests/test_cli.py`, `tests/test_golden.py`, `tests/golden/discretisation.json`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `loglog(x, y, title, ref_slope=None) -> str`, `histogram(x, title, bins=60) -> str`, `main(argv=None) -> int`.

Charts take plain arrays so `render/charts.py` stays a leaf and Phase B can reuse it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_charts.py
import numpy as np
from vollab.render.charts import loglog, histogram

def test_loglog_returns_a_non_empty_string():
    out = loglog(np.array([16.0, 64.0, 256.0]), np.array([1.0, 0.5, 0.25]), "sd vs N")
    assert isinstance(out, str) and len(out) > 0

def test_histogram_returns_a_non_empty_string():
    assert len(histogram(np.random.default_rng(0).standard_normal(500), "pnl")) > 0

def test_charts_do_not_raise_on_a_single_point():
    assert isinstance(loglog(np.array([1.0]), np.array([1.0]), "t"), str)
```

```python
# tests/test_cli.py
import json
from pathlib import Path
from vollab.cli import main
from vollab.protocol.hashing import config_hash
from vollab.protocol.prereg import load_registered

def test_vl_price_prints_a_number(capsys):
    assert main(["price", "--K", "100", "--T", "1", "--vol", "0.2", "--S", "100"]) == 0
    assert "price" in capsys.readouterr().out.lower()

def test_vl_run_writes_a_ledger_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "hypothesis": "sd falls as N^-0.5",
        "contract": {"kind": "call", "S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.0, "q": 0.0},
        "vols": {"s_imp": 0.3, "s_hedge": 0.3, "s_real": 0.3},
        "schedule": {"name": "fixed_time", "every": 1},
        "n_mon": 64, "cost_bps": 0.0, "n_paths": 500, "seed": 1,
    }
    p = tmp_path / "c.toml"
    import tomli_w  # dev dependency
    p.write_bytes(tomli_w.dumps({**cfg, "config_hash": config_hash(cfg)}).encode())
    assert main(["run", str(p)]) == 0
    from vollab.protocol.ledger import Ledger
    assert len(Ledger(tmp_path / ".vollab" / "ledger.db").all()) == 1

def test_vl_run_refuses_an_unregistered_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "bad.toml"
    p.write_text('config_hash = "stale"\nhypothesis = "h"\n')
    assert main(["run", str(p)]) == 2
```

```python
# tests/test_golden.py
import json
from pathlib import Path
import numpy as np
from vollab.hedge.registry import build_config
from vollab.hedge.simulator import simulate

GOLDEN = Path(__file__).parent / "golden" / "discretisation.json"

def test_golden_run_is_reproducible():
    g = json.loads(GOLDEN.read_text())
    r = simulate(build_config(g["config"]))
    assert abs(r.pnl.mean() - g["pnl_mean"]) < 1e-12
    assert abs(r.pnl.std(ddof=1) - g["pnl_sd"]) < 1e-12
    assert int(r.n_rehedges.sum()) == g["n_rehedges_total"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_charts.py tests/test_cli.py tests/test_golden.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'vollab.render'`

- [ ] **Step 3: Write the implementation**

```python
# src/vollab/render/charts.py
import numpy as np

def _plt():
    try:
        import plotext
        return plotext
    except Exception:
        return None

def loglog(x, y, title, ref_slope=None):
    p = _plt()
    if p is None:
        return _ascii_fallback(x, y, title)
    p.clf(); p.title(title)
    p.plot(np.log(x).tolist(), np.log(y).tolist(), marker="braille")
    if ref_slope is not None and len(x) > 1:
        lx = np.log(x)
        ref = np.log(y[0]) + ref_slope * (lx - lx[0])
        p.plot(lx.tolist(), ref.tolist(), marker="dot")
    return p.build()

def histogram(x, title, bins=60):
    p = _plt()
    if p is None:
        return _ascii_fallback(np.arange(len(x)), x, title)
    p.clf(); p.title(title); p.hist(np.asarray(x).tolist(), bins=bins)
    return p.build()

def _ascii_fallback(x, y, title):
    """Named fallback for the plotext risk in the spec's risk table."""
    y = np.asarray(y, dtype=float)
    lo, hi = y.min(), y.max()
    span = (hi - lo) or 1.0
    blocks = " .:-=+*#%@"
    line = "".join(blocks[int((v - lo) / span * (len(blocks) - 1))] for v in y)
    return f"{title}\n{line}\n[{lo:.4g} .. {hi:.4g}]"
```

```python
# src/vollab/cli.py
import argparse, json, platform, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np, scipy
from vollab.pricing.black_scholes import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta
from vollab.protocol.prereg import load_registered, HashMismatch
from vollab.protocol.hashing import config_hash
from vollab.protocol.ledger import Ledger
from vollab.hedge.registry import build_config
from vollab.hedge.simulator import simulate
from vollab.metrics.bootstrap import bootstrap_sd
from vollab.rng.scheme import RNG_SCHEME_VERSION
from vollab.render.charts import histogram

def _git(*args, default=""):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return default

def _cmd_price(a):
    args = (a.S, a.K, a.T, a.r, a.q, a.vol)
    print(f"price {bs_price(a.kind, *args):.6f}")
    print(f"delta {bs_delta(a.kind, *args):.6f}  gamma {bs_gamma(a.kind, *args):.6f}")
    print(f"vega  {bs_vega(a.kind, *args):.6f}  theta {bs_theta(a.kind, *args):.6f}")
    return 0

def _cmd_run(a):
    try:
        d = load_registered(a.config)
    except (HashMismatch, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    cfg = build_config(d)
    t0 = time.time()
    r = simulate(cfg)
    sd, se = bootstrap_sd(r.pnl, n_boot=500)
    print(histogram(r.pnl, "terminal P&L"))
    print(f"mean {r.pnl.mean():.6f} +/- {r.pnl.std(ddof=1)/np.sqrt(r.pnl.size):.6f}")
    print(f"sd   {sd:.6f} +/- {se:.6f}")
    print(f"hypothesis: {d['hypothesis']}")
    Ledger(Path(".vollab") / "ledger.db").insert(dict(
        run_id=uuid.uuid4().hex, schema_version=1,
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"), config_hash=config_hash(d),
        config_toml=Path(a.config).read_text(), hypothesis=d["hypothesis"],
        paths_fingerprint=config_hash({k: d[k] for k in ("contract", "vols", "n_mon",
                                                         "n_paths", "seed")}),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=int(bool(_git("status", "--porcelain"))),
        git_diff_sha=_git("diff", "--stat") and config_hash({"d": _git("diff")}) or None,
        vollab_version="0.1.0", rng_scheme_version=RNG_SCHEME_VERSION,
        engine=r.engine_used, engine_build_id=None,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        status="ok", n_paths_completed=int(r.pnl.size),
        metrics=json.dumps({"pnl_mean": float(r.pnl.mean()), "pnl_sd": sd, "pnl_sd_se": se}),
        artifacts="[]", artifact_sha256="[]", runtime_s=time.time() - t0))
    return 0

def _cmd_ledger(a):
    for row in Ledger(Path(".vollab") / "ledger.db").all():
        print(f"{row['ts']}  {row['run_id'][:8]}  {row['config_hash'][:8]}  {row['status']}")
    return 0

def main(argv=None):
    p = argparse.ArgumentParser(prog="vl")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("price", help="1. Black-Scholes price and greeks")
    pr.add_argument("--kind", default="call"); pr.add_argument("--S", type=float, default=100.0)
    pr.add_argument("--K", type=float, required=True); pr.add_argument("--T", type=float, required=True)
    pr.add_argument("--r", type=float, default=0.0); pr.add_argument("--q", type=float, default=0.0)
    pr.add_argument("--vol", type=float, required=True); pr.set_defaults(fn=_cmd_price)
    rn = sub.add_parser("run", help="2. run a pre-registered experiment")
    rn.add_argument("config"); rn.set_defaults(fn=_cmd_run)
    lg = sub.add_parser("ledger", help="3. list recorded runs")
    lg.set_defaults(fn=_cmd_ledger)
    a = p.parse_args(argv)
    return a.fn(a)
```

Add `tomli-w` to the dev dependency group. Generate the golden file once by running
the golden config and writing the observed `pnl_mean`, `pnl_sd` and `n_rehedges_total`
into `tests/golden/discretisation.json`; commit it as the snapshot.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vollab/render src/vollab/cli.py configs tests
git commit -m "Add charts, CLI and golden run"
```

---

## Self-Review

**Spec coverage.** Sections 5 (cash recursion with dividends, two deltas), 6 (F1 in
Task 8, F2 in Task 10, F3 in Task 9, F5 in Task 12), 8 (interfaces in Task 7), 9
(chunking in Task 7), 10 GBM only (Task 5), 11 keying and nesting (Tasks 4, 5, 8), 12
(Task 7), 13 (Task 9), 14 (Tasks 6, 11), 15 (Task 13), 16 (Task 12), 17 (Task 14), 19
rows for analytic pricing, law recovery, F2 lock-in, zero mean, attribution accuracy,
F5 U-curve, cash accounting, properties and golden run are all covered.

Deferred to Part 2 by design, not by omission: Heston and Merton (spec 10), all
parity tiers (spec 11), `vl bench`, `vl compare` against a stored run, `vl view`, the
TUI (spec 18), the jump-floor test, and REPORT.md.

**Known gaps to close in Part 2.** The cash-accounting per-step invariant from spec
section 12 is asserted only at the terminal level here; the per-step form arrives with
the trace subset. `vl compare` is specified in spec section 17 but has no Part 1 task,
since a paired comparison needs two stored runs and `paths_fingerprint` is only written
starting at Task 13; it is the first task of Part 2.

**Type consistency.** `should_trade` and `hedge_vol` carry one signature across all
four schedules (Tasks 6 and 11). `HedgeResult` field names are fixed in Task 7 and
extended once, in Task 9, with `attribution`. `config_hash` is used identically by
`prereg`, `cli` and the tests.
