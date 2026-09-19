# vol-lab: design

Date: 2026-09-19
Status: approved, revised after four independent reviews
Phase: A (module `hedge`)

Revision note. v1 was audited by four independent reviewers covering quantitative
correctness, software architecture, and internal consistency. v2 incorporates their
findings. Where a review was rejected, section 14 records why.

## 1. Purpose

A terminal-native Monte Carlo laboratory for the question an options market maker
answers every day: what is my P&L if I sell an option and delta hedge it, and how
does that P&L change with hedging frequency, transaction costs, and the process the
underlying actually follows?

The deliverable is not a library. It is five reproducible findings, each backed by a
test that fails if the engine is wrong, and each rendered as a chart in the terminal.

## 2. Context and constraints

Author profile: quantitative finance major, coursework in stochastic calculus,
pricing and market risk, numerical methods, numerical optimisation. Paid experience
as a quantitative research intern at a crypto options market maker. C++ claimed as a
skill with no public evidence behind it.

Constraint that shapes scope: internship applications are open now. A partial system
demonstrable in three weeks beats a complete system finished in three months.
Milestones are ordered so each ends at a state worth showing.

Phase A covers the `hedge` module only. Phase B (`surface`) and Phase C (`mm`) get
their own spec cycles and reuse this phase's protocol, rendering and C++ build.

## 3. Non-goals

American options, exotics, real market data, multi-asset books, portfolio-level
hedging, live trading, broker connectivity, a web interface, a plugin system.

## 4. Notation

| Symbol | Meaning |
|--------|---------|
| `S_t`, `K`, `T` | underlying, strike, maturity |
| `r`, `q` | risk-free rate, continuous dividend yield |
| `s_imp` | implied volatility at which the option is sold and subsequently marked |
| `s_hedge` | volatility at which the hedge delta is sized |
| `s_real` | volatility of the true process (for Heston, the initial `sqrt(v_0)`) |
| `s` | a generic volatility where the distinction does not matter |
| `N_mon` | monitoring grid steps: where the process is simulated and schedules evaluated |
| `N_reh` | realized number of rehedges on a path, an output not an input |
| `dt` | `T / N_mon` |
| `k` | proportional transaction cost rate (quoted in bps) |
| `h` | delta band half-width |
| `Delta, Gamma, Theta, Vega` | **position** greeks (short one call gives `Delta < 0`, `Gamma < 0`) |
| `v, kap_h, th_h, xi, rho` | Heston variance, mean-reversion speed, long-run variance, vol-of-vol, correlation |
| `lam` | Merton jump intensity |
| `mu_J, s_J` | mean and standard deviation of log jump size |
| `kap` | Merton compensator, `exp(mu_J + 0.5*s_J^2) - 1` |
| `gam_ra` | Whalley-Wilmott risk aversion (units of inverse currency) |
| `Le` | Leland number |

All greeks in this document are position greeks. Every formula is written in that
convention and the implementation carries it, so no sign flip occurs downstream.

## 5. The experiment, formally

At `t_0` the desk sells one European call struck `K` expiring `T` and receives the
Black-Scholes premium at `s_imp`. It holds `Delta` units of the underlying,
resized on a schedule using `s_hedge`. The underlying evolves under a true process
whose parameters are set independently of `s_imp`.

```
B_0 = BS(S_0, s_imp) - Delta_0 * S_0 - k * |Delta_0| * S_0
B_i = B_{i-1} * exp(r * dt)
      + q * Delta_{i-1} * S_{i-1} * dt            dividends on the held shares
      - (Delta_i - Delta_{i-1}) * S_i
      - k * |Delta_i - Delta_{i-1}| * S_i
PnL = B_{N-1} * exp(r * dt) + q * Delta_{N-1} * S_{N-1} * dt
      + Delta_{N-1} * S_T - payoff(S_T) - k * |Delta_{N-1}| * S_T
```

The dividend leg is explicit. v1 omitted it from the recursion while attributing it
in the P&L explain, which would have produced a systematic residual for any `q != 0`.

**Two distinct deltas.** `Delta_i` above is the traded hedge, sized at `s_hedge` (or
at the Leland-adjusted volatility under that schedule). The greeks used in the P&L
explain of section 11 mark the short option and are evaluated at `s_imp`. These are
different numbers whenever `s_hedge != s_imp`, which is the entire subject of F2.
The implementation names them `delta_hedge` and `delta_mark` and never reuses one
symbol for both.

### One-step hedging error

Freezing `Gamma` over a step and writing `dS/S = s * sqrt(dt) * z` with `z ~ N(0,1)`:

```
dPnL = 0.5 * Gamma * S^2 * s^2 * dt * (z^2 - 1)
```

`Var(z^2 - 1) = 2` since `E[z^4] = 3`. Per-step terms are not literally independent,
because `Gamma` is a function of the realized path. They are asymptotically
uncorrelated under the per-step freezing approximation, and the quadratic-variation
argument gives total variance proportional to `1/N_reh` as `dt -> 0`. Hence
`sd(PnL) ~ N_reh^(-1/2)`, the Boyle and Emanuel (1980) result, and the engine's
primary correctness test.

## 6. Findings

| ID | Setup | Expected result |
|----|-------|-----------------|
| F1 | GBM, `s_hedge = s_real = s_imp`, no costs | `E[PnL]` indistinguishable from zero; log-log slope of `sd(PnL)` against `N_reh` is -0.5 |
| F2 | GBM, `s_imp != s_real` | Hedging at `s_real`: terminal P&L converges to the deterministic `BS(s_imp) - BS(s_real)`, known at inception, with a random mark-to-market path. Hedging at `s_imp`: P&L is path dependent, always of the sign of `s_imp - s_real`, with mean equal to `BS(s_imp) - BS(s_real)` |
| F3 | Any model | Realized P&L decomposes into delta, gamma, theta, vega, carry and cost terms. The residual is third order under GBM and large under jumps. The size of the residual measures the risk delta hedging cannot see |
| F4 | Merton jumps | `sd(PnL)` stops decreasing in `N_reh` and reaches a floor. Fat left tail |
| F5 | GBM with costs | Total cost against rehedge frequency is U-shaped. The optimum is compared against Leland and a Whalley-Wilmott band |

**F2 sign.** The short position profits when realized volatility comes in below
implied, so the pathwise sign follows `s_imp - s_real`, matching the sign of the mean
`BS(s_imp) - BS(s_real)`. v1 stated the opposite sign in the same sentence as the
mean, which was self-contradictory.

**F2 drift condition.** The equality of the two means holds only when the simulated
drift equals `r - q`. The F2 configuration pins the drift and its test asserts it.

**F2 dispersion.** The hedge-at-realized case is not dispersion-free at finite
`N_reh`. Its dispersion is exactly F1's discretisation error and vanishes as
`N_reh^(-1/2)`.

**F4 mechanism.** Between jumps the diffusive error hedges away as `N_reh^(-1/2)`.
Each Poisson jump delivers an unhedgeable convexity P&L that no rehedge frequency can
pre-empt, with variance set by `lam`, the jump-size law and `T`, all independent of
`N_reh`. Total variance therefore converges to that floor rather than to zero.

## 7. Architecture

```
vol-lab/
  src/vollab/
    rng/        philox.py scheme.py        # keying scheme, RNG_SCHEME_VERSION
    pricing/    black_scholes.py greeks.py merton.py heston_cf.py
    metrics/    stats.py bootstrap.py
    paths/      base.py gbm.py heston.py merton.py
    hedge/      simulator.py schedule.py attribution.py costs.py registry.py
    protocol/   prereg.py ledger.py hashing.py
    render/     charts.py report.py
    tui/        app.py
    cli.py
  cpp/          include/vollab/{philox,paths,hedge}.hpp src/ bindings.cpp CMakeLists.txt
  configs/      *.toml
  tests/  REPORT.md  docs/superpowers/specs/
```

Dependency graph, corrected from v1:

```
rng, pricing, metrics          leaves
paths      -> rng
hedge      -> paths, pricing, rng
render.charts                  leaf (plain arrays only)
protocol                       leaf: hashes canonical TOML bytes, stores JSON,
                               imports nothing from vollab
cli, tui, render.report        -> everything
```

v1 drew `hedge -> metrics -> protocol`, which was wrong twice. `metrics` operates on
plain arrays and imports nothing from `hedge`, so it is a leaf. And `protocol` would
have had to import `hedge` to turn TOML into a `HedgeConfig`, contradicting the
section 22 reuse seam. TOML resolution therefore lives in `hedge/registry.py`, and
`protocol/hashing.py` hashes canonicalised TOML bytes without knowing what they mean.

Stack: uv, Python 3.14, numpy, scipy, nanobind, **scikit-build-core** (the PEP 517
backend uv needs to build a CMake project; absent from v1), plotext, textual, pytest,
hypothesis. Editable builds require `tool.scikit-build.build-dir` and
`editable.rebuild = true`, or every C++ edit needs a manual reinstall.

## 8. Interfaces

```python
@dataclass(frozen=True)
class Contract:
    kind: Literal["call", "put"]
    S0: float; K: float; T: float; r: float; q: float

@dataclass(frozen=True)
class VolSpec:
    s_imp: float          # sale and mark
    s_hedge: float        # hedge sizing
    s_real: float         # true process; for Heston, sqrt(v_0)

@dataclass(frozen=True)
class HedgeConfig:
    contract: Contract
    vols: VolSpec
    model: PathModel          # GBM | Heston | Merton; carries optional mu override
    schedule: Schedule        # FixedTime | DeltaBand | Leland | WhalleyWilmott
    n_mon: int                # monitoring grid, the single source of truth
    cost_bps: float           # the single source of truth for k
    n_paths: int
    seed: int
    chunk_paths: int = 50_000
    trace_paths: tuple[int, ...] = ()

@dataclass(frozen=True)
class Attribution:            # named (n_paths,) arrays, not an (n_paths, m) block
    delta: np.ndarray; gamma: np.ndarray; theta: np.ndarray
    vega:  np.ndarray; carry: np.ndarray; cost:  np.ndarray
    residual_sum: np.ndarray; residual_abs_sum: np.ndarray; residual_max_abs: np.ndarray

@dataclass(frozen=True)
class HedgeResult:
    pnl: np.ndarray               # (n_paths,) float64
    attribution: Attribution
    n_rehedges: np.ndarray        # (n_paths,) int64
    turnover: np.ndarray          # (n_paths,) float64
    rehedge_mask_hash: np.ndarray # (n_paths,) uint64, digest of the decision sequence
    trace: PathTrace | None       # per-step detail for trace_paths only
    engine_used: Literal["numpy", "cpp"]
    rng_scheme_version: int

def simulate(cfg: HedgeConfig, engine: Literal["numpy", "cpp"]) -> HedgeResult: ...
```

Five interface decisions, each fixing a v1 defect:

- **`simulate(cfg, "cpp")` raises when the extension is missing.** It never silently
  falls back. v1's fallback would have made the parity test compare NumPy against
  NumPy and pass. The graceful fallback lives in the CLI layer only, and
  `engine_used` records what actually ran.
- **`n_mon` and `cost_bps` are the only sources of truth.** Schedules read `k` rather
  than carrying their own, and `FixedTime(every)` rehedges every `every`-th
  monitoring step rather than declaring its own step count.
- **`VolSpec` gives `s_real` a home.** v1 buried realized volatility inside the model
  object, leaving F2 undefined for Heston and Merton.
- **`PathModel` carries an optional `mu` override**, so F2's drift condition is
  configurable and therefore assertable. v1 hardcoded `r - q` and then promised a
  test that pinned the drift, which was unwritable.
- **Attribution is a dataclass of named arrays** carrying three residual statistics.
  A signed sum alone cancels two-signed jump residuals across steps, destroying
  exactly the F4 measurement it exists to support.

## 9. Memory and chunking

Simulating 1e6 paths on a 4096-step grid is 33 GB as one float64 array, and section
11 requires always simulating on the finest grid, so this is the normal case rather
than the extreme one. `simulate` therefore loops over path chunks of `chunk_paths`,
accumulating per-path summary arrays and retaining full per-step detail only for
`trace_paths`. Counter-based keying makes an individual path independently
addressable, which is what makes a cheap trace subset possible.

This is decided before M2 because retrofitting chunking through the cash-accounting
assertions is a rewrite.

## 10. Path models

Each model exposes a chunked generator and an independent closed-form or
semi-analytic price used as ground truth. Having an independent price for every model
means a broken path generator is caught by a pricing test before it reaches the
hedging logic.

**GBM.** `dS = (r - q) S dt + s S dW`, simulated exactly in log space with the Ito
correction `-0.5*s^2*dt`. Ground truth: Black-Scholes.

**Heston.** `dS = (r - q) S dt + sqrt(v) S dW1`,
`dv = kap_h (th_h - v) dt + xi sqrt(v) dW2`, `corr(dW1, dW2) = rho`. Andersen
quadratic-exponential discretisation, with full-truncation Euler as a small-`dt`
cross-check. Ground truth: the Heston (1993) characteristic-function price.

The Feller condition is `2 * kap_h * th_h > xi^2`. Equity-like calibrations routinely
violate it, which is precisely where QE and Euler agreement is most `dt`-sensitive
and where the characteristic-function integration needs care near the pole. The test
grid includes one Feller-satisfying and one Feller-violating parameter set, and the
spec records which is which.

**Merton.** `dS/S = (r - q - lam*kap) dt + s dW + (J - 1) dN`, `ln J ~ N(mu_J, s_J^2)`,
`kap = exp(mu_J + 0.5*s_J^2) - 1`.

Ground truth is the Merton series, specified completely because a naive
implementation reusing one rate and one volatility per term is wrong:

```
C = sum_n  w_n * BS(S, K, T, r_n, q, s_n)
w_n = exp(-lam' T) (lam' T)^n / n!      with lam' = lam * (1 + kap)
s_n = sqrt(s^2 + n * s_J^2 / T)
r_n = r - lam*kap + n * ln(1 + kap) / T
```

Truncated when the weight tail falls below 1e-12. This formula was verified
numerically against an independent Monte Carlo of the risk-neutral jump diffusion
before being written here: series 10.040100 against MC 10.044299 +/- 0.009470, a
separation of 0.44 standard errors.

**Jump times are generated per path, not per step.** Each path draws its jump times
and sizes from a dedicated counter stream keyed by path alone. The jump set is
therefore identical across every `N_mon` in a sweep, which is what makes F4's
frequency sweep paired rather than unpaired.

## 11. Random numbers and engine parity

This is the highest-risk section of the project. Three decisions, each verified
against numpy 2.5.3 by running code rather than from memory.

### Keying scheme

```
key     = (seed_u64, path_index_u64)        # Philox key is exactly 128 bits
counter = (block_index_u64, 0, 0, 0)        # block_index = draw_index // 4
draw_index = fine_step_index * DRAWS_PER_STEP + factor_index
```

Verified constraints:

- `numpy.random.Philox(key=...)` accepts **exactly two** uint64 words.
  `key=[1,2,3]` raises `ValueError: key must have 2 elements when using array form`.
  v1's proposed `(seed, path_index, step_index)` key is therefore impossible; the
  step index must live in the counter.
- **`numpy.random.Philox(key=k, counter=c)` emits the Philox4x64-10 block for counter
  `c+1`, not `c`.** A C++ Random123 call with `ctr=c` is off by one block and
  produces a stream that is correct in distribution and wrong in parity.
- **`advance(n)` moves `n` blocks of four outputs and discards the buffer**, despite
  its docstring saying draws. Verified: `advance(1)` leaves counter `[1,0,0,0]` with
  `buffer_pos=4`, while one draw leaves the same counter with `buffer_pos=1`.
  Implementing step offsets with `advance` silently strides 4x and can overlap
  streams between paths. The scheme never uses it.
- `Generator.random()` equals `(u64 >> 11) * 2**-53` exactly, so the uniform stream is
  bit-reproducible in C++ from one line.

Per-draw keying was measured at 0.139 Mdraw/s against 20.8 for per-path keying and
49.2 for a single vectorised stream. Per-draw keying is 355x too slow and is
rejected; per-path keying costs 2.4x and is affordable.

`rng/scheme.py` defines `RNG_SCHEME_VERSION`, which both languages read and which is
written to every ledger row.

### Fixed per-step draw budget

Heston QE branches on `psi <= psi_c`, consuming a normal on one branch and a uniform
on the other. Vectorised NumPy draws both and selects; scalar C++ draws one. The
streams then desynchronise structurally, not approximately. Merton has the same
problem through a random jump count.

`DRAWS_PER_STEP` is therefore fixed per model and asserted in both engines: GBM 1,
Heston 3 (variance normal, variance uniform, price normal, all always drawn),
Merton 1 (diffusion only, since jumps are keyed per path). Both engines always
consume the full budget and discard what the branch did not use.

### Brownian nesting

F1 and F5 sweep `N_reh` and section 16 requires paired comparison. Keying by step
index alone would give the `N=16` and `N=4096` runs completely different paths,
making the `sd` against `N` curve an unpaired comparison with far larger Monte Carlo
error than assumed.

Paths are therefore always generated on the finest grid `N_mon`, and coarser rehedge
frequencies are obtained by trading every `2^j`-th monitoring step. All frequencies
in a sweep are then Brownian-nested on identical paths.

This works exactly for GBM, and for Merton because jump times are grid-independent by
section 10. It does not work for Heston, whose variance path is grid-dependent. No
`N`-sweep finding depends on Heston, and section 19's test for Heston states that its
comparisons are unpaired.

### Tiered parity acceptance

v1 demanded bitwise identity between engines. That is unattainable, for a reason that
has nothing to do with care in implementation: `DeltaBand` branches on a float
comparison, so a 1-ULP difference in delta flips a boolean on some path at some step
and the engines then differ by **a whole trade**, not by a rounding error. Heston QE
and Merton branch similarly, and 27% of inverse-CDF draws land on a `log` branch that
is not bit-portable across libms. A bitwise gate would either never go green or go
green on GBM and break at M6.

Four separate assertions replace it:

| Level | Scope | Criterion |
|-------|-------|-----------|
| Exact | raw Philox uint64 stream | bitwise identical (pure integer arithmetic) |
| Exact | uniform doubles | bitwise identical via `(u64 >> 11) * 2**-53` |
| Exact | decision sequence | `n_rehedges` and `rehedge_mask_hash` identical per path |
| Tolerance | normals, paths, P&L | 2 ULP on normals, 1e-13 relative on log-paths, `n_mon * eps * scale` on P&L |

The decision-sequence assertion is the one that actually catches an engine bug: it
fails loudly on a real defect and survives ULP noise. A branch-free
GBM + FixedTime + zero-cost configuration additionally carries a stricter tolerance.

## 12. Hedge simulator

Vectorised over paths, looping over monitoring steps, chunked per section 9. At each
step it computes `delta_hedge` for all paths, applies the schedule as a boolean mask,
updates cash, accrues financing and dividends, and charges costs on traded notional.
Paths not triggering carry their position forward.

**Monitoring grid against rehedge grid.** The schedule condition is evaluated at
every monitoring step; trading happens on the subset where it triggers. v1 conflated
the two, which would have let two engineers implement `DeltaBand` on different grids
and obtain different F5 curves. Band schedules reduce trading cost, not compute:
delta is still computed for every path at every monitoring step.

Asserted by construction: cash plus share position minus option mark equals running
P&L at every step of every path, and a path that never rehedges reproduces a naked
short.

## 13. Attribution

Per monitoring step, per path, then summed into the section 8 arrays:

```
delta_pnl = delta_mark * dS
gamma_pnl = 0.5 * Gamma  * dS^2
theta_pnl = Theta * dt
vega_pnl  = Vega * d(s_imp)                       # identically zero in Phase A
carry_pnl = cash * (exp(r*dt) - 1) + q * delta_hedge * S * dt
cost_pnl  = -k * |d(delta_hedge)| * S
residual  = realized_step_pnl - sum(the above)
```

Four corrections to v1:

- **Costs have their own line.** v1 left `k*|dDelta|*S` unattributed, so under F5,
  the cost sweep, the residual would have been dominated by transaction costs and
  F3's interpretation of the residual would have been false.
- **Carry uses `exp(r*dt) - 1`, not `r*dt`.** v1's linearisation disagreed with
  section 5's `exp(r*dt)` accrual by `O((r dt)^2)` with the same sign every step,
  which accumulates into a systematic residual.
- **Mark greeks use `s_imp`; the traded hedge uses `s_hedge`.** `delta_mark` and
  `delta_hedge` are different numbers whenever the two volatilities differ, which is
  the F2 configuration. v1 used one symbol for both.
- **`vega_pnl` is an all-zeros column in Phase A**, since implied volatility is held
  fixed. It is kept as the Phase B seam and stated to be zero so a reader does not
  hunt for it in the stacked-bar chart.

The interpretation that makes the residual a finding: the expansion is exact to
second order in `dS`, so under GBM the residual is third order and negligible, while
a jump is not small and produces a large residual. Its magnitude directly measures
how much risk delta hedging cannot see.

## 14. Costs and schedules

Cost model: proportional, `k * |d(delta_hedge)| * S`, with `k` in basis points, plus
an optional per-trade fee defaulting to zero.

- **FixedTime(every)**: trade every `every`-th monitoring step.
- **DeltaBand(h)**: trade when `|delta_target - delta_held| > h`.
- **Leland(every, k)**: fixed time with delta computed at `s_L^2 = s^2 * (1 +/- Le)`,
  `Le = sqrt(2/pi) * k / (s * sqrt(dt))`. **Plus for a written option, minus for a
  held one.** v1 hardcoded the plus branch with no statement, which is wrong for a
  long-option Leland schedule.
- **WhalleyWilmott(k, gam_ra)**: half-width proportional to
  `(k * S * Gamma^2 / gam_ra)^(1/3)`, per the 1997 paper.

Acceptance for WhalleyWilmott is the scaling, not a transcribed constant: the fitted
band must vary as `k^(1/3)` and `Gamma^(2/3)`.

**Rejected review finding.** One reviewer ranked as its top correction that the band
should contain `S^2 * Gamma^2`. This is dimensionally impossible. The band is a delta
band and must be dimensionless; `Delta` is dimensionless so `Gamma ~ 1/currency`, `k`
is dimensionless, and exponential-utility risk aversion `gam_ra ~ 1/currency`. Then
`k * S * Gamma^2 / gam_ra` is dimensionless while `k * S^2 * Gamma^2 / gam_ra` has
units of currency, whose cube root cannot be a delta band. The formula stands. The
likely confusion is between `S * Gamma^2` and cash gamma `S^2 * Gamma`.

## 15. Protocol layer

**Pre-registration.** An experiment is a TOML file declaring model, parameters, the
frequency grid, costs, seeds, and a mandatory prose hypothesis. The file carries a
`config_hash` field. `vl run` canonicalises the resolved configuration, recomputes
the hash, and refuses to run when the two differ.

v1 said the hash was computed from the file and compared against the file, which is
circular and checks nothing. The stored-field form is the intended semantics: editing
a parameter without re-registering invalidates the hash and blocks the run.

Canonicalisation is defined explicitly (sorted keys, normalised floats, no comments,
UTF-8, LF) or the hash is unstable across TOML writers.

**Ledger.** sqlite at `.vollab/ledger.db`:

```sql
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, ts TEXT NOT NULL,
  config_hash TEXT NOT NULL, config_toml TEXT NOT NULL, hypothesis TEXT NOT NULL,
  paths_fingerprint TEXT NOT NULL,   -- model, params, seed, n_paths, fine grid
  git_commit TEXT NOT NULL, git_dirty INTEGER NOT NULL, git_diff_sha TEXT,
  vollab_version TEXT NOT NULL, rng_scheme_version INTEGER NOT NULL,
  engine TEXT NOT NULL, engine_build_id TEXT,   -- commit, compiler, -O, -ffp-contract
  numpy_version TEXT NOT NULL, scipy_version TEXT NOT NULL,
  python_version TEXT NOT NULL, platform TEXT NOT NULL,
  status TEXT NOT NULL, n_paths_completed INTEGER NOT NULL,
  metrics TEXT NOT NULL, artifacts TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
  runtime_s REAL NOT NULL
);
CREATE INDEX runs_config_hash ON runs(config_hash);
CREATE INDEX runs_ts ON runs(ts);
```

`config_toml` alongside `config_hash` is not redundant: the hash is over the
canonical resolved form and survives cosmetic edits, the text is the human record.

`paths_fingerprint` exists so `vl compare` can **refuse** a paired bootstrap between
two runs that were not run on the same paths, which would otherwise be silently
wrong. Floating-point flags are recorded because section 11 makes them load-bearing.
`status` and `n_paths_completed` exist so a crashed run leaves an honest row rather
than none.

## 16. Statistics

Comparisons are paired: both arms run on identical paths from identical seeds, the
difference is taken per path, and the result is a mean difference with a bootstrap
confidence interval over paths. Pairing across perfectly correlated arms is sound
because resampling is across paths, which are independent.

**`sd(PnL)` gets its own bootstrap standard error.** Section 16 of v1 promised
standard errors on every mean, but the two headline tests, law recovery and the jump
floor, both depend on `sd`, a nonlinear statistic whose sampling error is not
`s/sqrt(n)`. Every reported `sd` carries a bootstrap interval.

An estimate without its standard error is a defect.

## 17. Rendering and CLI

```
vl price   --K 100 --T 0.25 --vol 0.6       Black-Scholes price and greeks
vl paths   show --model merton --seed 7     inspect sample paths
vl run     configs/discretisation.toml      pre-registered experiment
vl compare <run_a> <run_b>                  paired bootstrap, refuses on fingerprint mismatch
vl bench                                    NumPy against C++
vl view    <run_id>                         textual viewer
vl ledger                                   runs with config hashes
```

Charts: `sd(PnL)` against `N_reh` log-log with a -0.5 reference; P&L histograms by
model; attribution stacked bars; the cost against frequency U curve; delta and
underlying for a traced path.

**`vl bench` reports two speedups.** Section 11 mandates inverse-CDF on the NumPy
side, and `ndtri(random())` measured 3.4x slower than `standard_normal()` (49.2
against 165.8 Mdraw/s). Reporting only against the reference engine would compare C++
to a deliberately handicapped NumPy. Both numbers are printed, labelled "against
reference NumPy" and "against idiomatic NumPy".

`plotext` is a single-maintainer dependency gating every chart in M4. If it does not
work on Python 3.14, the fallback is a small ANSI braille renderer, which section 21
carries as a named risk.

## 18. TUI

`vl view <run_id>` opens a Textual application over saved artifacts only. It never
runs a simulation, which is what keeps the engine free of interactive code paths and
fully testable headless.

## 19. Testing

| Test | Asserts |
|------|---------|
| Analytic pricing | Put-call parity to 1e-12; greeks against central differences to 1e-6 relative; implied-vol inverse recovers input to 1e-10 |
| Model ground truth | MC price converges to Black-Scholes, the Merton series, and the Heston characteristic-function price, each within 3 standard errors at a stated minimum `n_paths`, raised for deep-OTM and jump configurations where payoff skew slows CLT convergence |
| RNG stream parity | C++ and NumPy raw uint64 and uniform doubles bitwise identical |
| Decision parity | `n_rehedges` and `rehedge_mask_hash` identical per path, per model, asserted only for models that exist at that milestone |
| Numeric parity | Normals within 2 ULP, log-paths 1e-13 relative, P&L within `n_mon * eps * scale` |
| Law recovery | Fitted slope of `log sd(PnL)` against `log N_reh` is -0.5 within a tolerance derived from the bootstrap SE of each `sd`, over `N_reh` from 2^4 to 2^12 on Brownian-nested paths |
| F2 lock-in | Hedging at `s_real` gives terminal P&L within tolerance of `BS(s_imp) - BS(s_real)`; hedging at `s_imp` has that same mean and a pathwise sign matching `s_imp - s_real`; the configured drift is asserted equal to `r - q` |
| Zero mean | Perfect hedge, zero cost: mean P&L not distinguishable from zero by a t-test |
| Attribution accuracy | Under GBM with `q=0, k=0`: `max|residual| / max|gamma_pnl| < tol`. A literal closure check runs only as a NaN guard |
| Jump floor | Under Merton, `sd(PnL)` at `N=2^12` is at least a calibrated fraction of its value at `N=2^8`, with the fraction derived from the analytic jump-variance floor and a tolerance from the bootstrap SE of each `sd` |
| F5 U-curve | The cost-against-frequency curve has an interior minimum, and its location moves with `k` in the direction Leland predicts |
| Cash accounting | Cash plus shares minus option mark equals running P&L at every step, every path |
| Properties (hypothesis) | P&L is non-increasing in `k`; band schedules trade no more often than `FixedTime(1)` on the same monitoring grid |
| Fixed-seed ladder | On a fixed coarse `N` ladder, zero-cost GBM variance is non-increasing in frequency, within a stated multiple of the SE |
| Golden run | Fixed seed and config reproduce a committed metrics snapshot |

Five test corrections carried from review:

- **Attribution closure was vacuous.** Since `residual` is *defined* as the realized
  P&L minus the other terms, "the components sum to realized P&L" is true by
  construction for any hedger, including a broken one. The test that has power is a
  bound on the residual relative to the gamma term.
- **Engine parity must skip, not pass, when the extension is absent**
  (`pytest.importorskip`), and CI fails if it skipped where the build was expected.
  Combined with v1's silent fallback this was a green test proving nothing.
- **"Costs are monotonically non-increasing in P&L"** was garbled. The testable
  statement is that P&L is non-increasing in `k`.
- **"Payoffs are non-negative"** is a tautology for vanillas and is dropped.
- **"More frequent hedging does not increase variance"** is a statistical claim, and
  as a hypothesis property it would find a Monte Carlo corner where noise inverts it.
  It becomes a fixed-seed ladder with an explicit SE tolerance.

## 20. Milestones

| ID | Days | Deliverable | Done when |
|----|------|-------------|-----------|
| M1 | 1-3 | uv project, Black-Scholes price and greeks, CLI skeleton | Analytic pricing tests green, `vl price` correct |
| M2 | 4-7 | `rng/` scheme, NumPy GBM on the nested grid, simulator, chunking, cash accounting | Zero-mean, cash-accounting, law-recovery green. F1 exists |
| M3 | 8-11 | Attribution with cost and carry lines, the four schedules | Attribution-accuracy green. F3 and F5 measurable |
| M4 | 12-14 | Charts, `vl run`, `vl compare`, pre-registration, ledger | A registered config yields a chart, a ledger row, and a refused mismatched compare. F2 test green |
| M5 | 15-26 | C++ core, nanobind, scikit-build-core, tiered parity, `vl bench` | Stream, decision and numeric parity green on GBM; both speedups reported |
| M6 | 27-30 | Heston and Merton with their ground-truth prices | Jump-floor green, parity extended to both models. F4 exists |
| M7 | 31-34 | REPORT.md with F1 to F5, README, Textual viewer | Every finding has a chart, a number with a standard error, and a command that regenerates it |

**M5 was re-estimated from 4 days to 12.** It comprises a CMake project, nanobind,
a uv editable build, Philox4x64-10 matching numpy's key, counter and buffer semantics
including the `c+1` offset, a Cephes `ndtri` port, GBM paths, the hedge loop, parity
debugging and `vl bench`. Four evenings is the figure for someone who has done the
toolchain before, and section 2 records that this author has not.

**M5 degraded deliverable.** If it overruns, it reduces to a C++ GBM path generator
only, benched against NumPy with parity asserted on the uniform bit stream. That
still discharges the section 2 purpose at roughly 40% of the cost.

**Early warning signs, in order.** First, end of M5 day 1 without
`python -c "import vollab._core"` working from a clean `uv sync`: if the toolchain is
not proven on day 1 with a trivial function, timebox the build separately. Second, a
first parity run differing only in low bits, which looks nearly working and is not.
Third, M3 running long, since the Whalley-Wilmott scaling fit is a day on its own.

## 21. Risks

| Risk | Mitigation |
|------|------------|
| Engines diverge untraceably | Counter-based keying makes any disagreement reproduce on one path; tiered acceptance separates ULP noise from real defects |
| Build fails on the target machine | Python engine is complete alone; `simulate(cfg, "cpp")` raises rather than falling back, and only the CLI degrades |
| Heston discretisation bias mistaken for a hedging result | The characteristic-function price gates every Heston finding; no Heston chart ships before it is green |
| `plotext` unusable on Python 3.14 | Named fallback: a small ANSI braille renderer. M4 charts are the gate for M7 |
| M5 overruns the application window | Degraded deliverable defined above; M2, M4 and M7 are independently presentable |
| Scope creeps into Phase B | `surface` and `mm` are named non-goals requiring their own spec cycles |

## 22. Seams for later phases

Reused unchanged by Phase B and C: `protocol/` (a leaf that hashes canonical TOML and
stores JSON, importing nothing from `hedge`), `render/charts.py` (plain arrays only),
`pricing/`, `rng/`, and the C++ build.

What Phase A must expose: pricing and greeks as free functions independent of hedging;
a ledger keyed by an opaque configuration hash; chart helpers taking plain arrays; and
a TOML registry that lives in `hedge/` so `protocol/` stays domain-free.

## References

- Boyle, P. and Emanuel, D. (1980). Discretely adjusted option hedges.
- Leland, H. (1985). Option pricing and replication with transactions costs.
- Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Merton, R. (1976). Option pricing when underlying stock returns are discontinuous.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model.
- Ahmad, R. and Wilmott, P. (2005). Which free lunch would you like today, sir?
