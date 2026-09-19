# vol-lab: design

Date: 2026-09-19
Status: approved, pending implementation plan
Phase: A (module `hedge`)

## 1. Purpose

A terminal-native Monte Carlo laboratory for the question an options market maker
answers every day: what is my P&L if I sell an option and delta hedge it, and how
does that P&L change with hedging frequency, transaction costs, and the process the
underlying actually follows?

The deliverable is not a library. It is a set of five reproducible findings, each
backed by a test that fails if the engine is wrong, and each rendered as a chart in
the terminal.

## 2. Context and constraints

Author profile: quantitative finance major, coursework in stochastic calculus,
pricing and market risk, numerical methods, numerical optimisation. Paid experience
as a quantitative research intern at a crypto options market maker (Deribit implied
volatility, digital-option fair value, live pricer, order-book data). C++ claimed as
a skill with no public evidence behind it.

Constraint that shapes scope: internship applications are open now. A partial system
that is demonstrable in three weeks is worth more than a complete system finished in
three months. Milestones are therefore ordered so that each one ends at a state
worth showing.

Phase A covers the `hedge` module only. Phase B (`surface`, SVI/SSVI calibration
with no-arbitrage constraints) and Phase C (`mm`, quoting with inventory risk) get
their own spec cycles and reuse this phase's protocol, rendering and C++ build.

## 3. Non-goals

American options, exotics, real market data, multi-asset books, portfolio-level
hedging, live trading, broker connectivity, a web interface, a plugin system.

Real market data enters in Phase B, where it is the point. In Phase A it would
remove the one thing that makes the engine provable: a closed-form answer to compare
against.

## 4. The experiment, formally

At `t_0` the desk sells one European call struck at `K` expiring at `T`, and
receives the Black-Scholes premium computed at implied volatility `s_imp`. It hedges
by holding `Delta` units of the underlying, recomputed at each rehedge time using
hedging volatility `s_hedge`. The underlying evolves under a true process whose
parameters are set independently of `s_imp`.

Cash account recursion, with `k` the proportional transaction cost rate:

```
B_0    = BS(S_0, s_imp) - Delta_0 * S_0 - k * |Delta_0| * S_0
B_i    = B_{i-1} * exp(r * dt) - (Delta_i - Delta_{i-1}) * S_i - k * |Delta_i - Delta_{i-1}| * S_i
PnL    = B_{N-1} * exp(r * dt) + Delta_{N-1} * S_T - payoff(S_T) - k * |Delta_{N-1}| * S_T
```

Liquidation of the final share position is charged at the same cost rate. The three
volatilities `s_imp`, `s_hedge` and the realized process volatility are separate
inputs, because the difference between them is the entire subject of findings F2 and
F3.

### One-step hedging error

All greeks in this document are **position** greeks: selling one call gives
`Delta < 0` and `Gamma < 0`. Every formula below is written in that convention, and
the implementation carries it, so no sign flip is applied anywhere downstream.

Holding `Gamma` fixed over a step and writing `dS/S = s * sqrt(dt) * z` with
`z ~ N(0, 1)`, the P&L contributed by one rehedge interval is

```
dPnL = 0.5 * Gamma * S^2 * s^2 * dt * (z^2 - 1)
```

Since `Var(z^2 - 1) = 2` and steps are independent, summing `N = T / dt` steps gives
a total variance proportional to `1 / N`, so `sd(PnL)` scales as `N^(-1/2)`. This is
the Boyle and Emanuel (1980) result and it is the engine's primary correctness test:
the fitted slope of `log sd(PnL)` against `log N` must be -0.5.

## 5. Findings the system must produce

| ID | Setup | Expected result |
|----|-------|-----------------|
| F1 | GBM, `s_hedge = s_real = s_imp`, no costs | `E[PnL]` indistinguishable from zero; `sd(PnL)` slope against `N` in log-log is -0.5 |
| F2 | GBM, `s_imp != s_real` | Hedging at `s_real`: terminal P&L converges to the deterministic `BS(s_imp) - BS(s_real)`, known at inception, while the mark-to-market path is random. Hedging at `s_imp`: P&L is path dependent, always of the sign of `s_real - s_imp`, with mean equal to that same quantity. Ahmad and Wilmott (2005) |
| F3 | Any model | Realized P&L equals the sum of attributed gamma, theta, vega and residual terms to floating-point tolerance. Residual is small under GBM and large under jumps |
| F4 | Merton jump diffusion | `sd(PnL)` stops decreasing in `N` and reaches a floor. Left tail is fat. Delta hedging cannot remove gap risk |
| F5 | GBM with transaction costs | Total cost against rehedge frequency is U-shaped. The located optimum is compared against Leland (1985) adjusted volatility and a Whalley and Wilmott (1997) band |

F4 is the result that matters for a crypto derivatives book. F5 is the result a
market maker pays for.

Two conditions on F2 that the implementation must respect. The equality of the two
means holds only when the simulated drift equals `r - q`, which is the case under
section 8; a configuration with a different drift breaks it, so the F2 experiment
pins the drift and the test asserts it. And the dispersion of the hedge-at-realized
case is not zero at finite `N`, it is the discretisation error of F1 and vanishes as
`N^(-1/2)`.

## 6. Architecture

```
vol-lab/
  src/vollab/
    paths/      base.py gbm.py heston.py merton.py
    pricing/    black_scholes.py merton.py heston_cf.py greeks.py
    hedge/      simulator.py schedule.py attribution.py costs.py
    metrics/    stats.py bootstrap.py
    protocol/   prereg.py ledger.py hashing.py
    render/     charts.py report.py
    tui/        app.py
    cli.py
  cpp/
    include/vollab/  philox.hpp paths.hpp hedge.hpp
    src/             paths.cpp hedge.cpp
    bindings.cpp
    CMakeLists.txt
  configs/      *.toml
  tests/
  REPORT.md
```

Dependency direction is one way:

```
paths, pricing  ->  hedge  ->  metrics  ->  protocol  ->  render, tui, cli
```

The simulator never imports the ledger. It is a pure function from a configuration
to an array of P&L outcomes, which is what makes it testable without any of the
surrounding machinery.

Stack: uv, Python 3.14, numpy, scipy, nanobind with CMake, plotext, textual, pytest,
hypothesis. Versions are pinned when added, not fixed by this document.

## 7. Interfaces

```python
@dataclass(frozen=True)
class Contract:
    kind: Literal["call", "put"]
    S0: float; K: float; T: float; r: float; q: float
    s_imp: float

@dataclass(frozen=True)
class HedgeConfig:
    contract: Contract
    model: PathModel            # GBM | Heston | Merton
    schedule: Schedule          # FixedTime | DeltaBand | Leland | WhalleyWilmott
    s_hedge: float
    cost_bps: float
    n_paths: int
    n_steps: int
    seed: int

@dataclass(frozen=True)
class HedgeResult:
    pnl: np.ndarray             # (n_paths,)
    attribution: Attribution    # (n_paths, 6): delta, gamma, theta, vega, carry, residual
    n_rehedges: np.ndarray      # (n_paths,)
    turnover: np.ndarray        # (n_paths,)

def simulate(cfg: HedgeConfig, engine: Literal["numpy", "cpp"]) -> HedgeResult: ...
```

`engine` is a parameter rather than a global, so the parity test is a direct call to
the same function twice.

## 8. Path models

Each model exposes `generate(n_paths, n_steps, T, seed) -> ndarray (n_paths, n_steps+1)`
and a closed-form or semi-analytic price used as its ground truth.

**GBM.** `dS = (r - q) S dt + s S dW`, simulated exactly in log space. Ground truth:
Black-Scholes.

**Heston.** `dS = (r - q) S dt + sqrt(v) S dW1`,
`dv = kappa (theta - v) dt + xi sqrt(v) dW2`, `corr(dW1, dW2) = rho`. Discretised
with the Andersen quadratic-exponential scheme; full-truncation Euler is kept as a
cross-check at small `dt`. Ground truth: the Heston (1993) characteristic-function
price by numerical integration.

**Merton jump diffusion.** `dS/S = (r - q - lam * kap) dt + s dW + (J - 1) dN`, with
`ln J ~ N(mu_J, s_J^2)`, `kap = exp(mu_J + 0.5 * s_J^2) - 1`, and `N` a Poisson
process of intensity `lam`. Ground truth: the Merton series price, a Poisson-weighted
sum of Black-Scholes prices, truncated when the weight tail falls below 1e-12.

Every model having an independent analytic price is deliberate. It means a broken
path generator is caught by a pricing test before it ever reaches the hedging logic.

## 9. Pricing and greeks

Black-Scholes price, delta, gamma, vega, theta and rho in closed form, for calls and
puts, with continuous dividend yield `q`. Implemented once and reused by the hedger,
the attribution module and the schedules.

Acceptance: put-call parity holds to 1e-12; every analytic greek matches a central
finite difference of the price to 1e-6 relative; the implied-volatility inverse
recovers the input volatility to 1e-10 over a wide strike and maturity grid.

## 10. Hedge simulator

Vectorised over paths. At each step it computes delta for all live paths, applies the
schedule to decide which paths rehedge, updates the cash account, accrues financing
at `r`, and charges costs on traded notional. Paths that have not triggered a rehedge
carry their previous position forward.

Two properties hold by construction and are asserted: cash accounting is exact (the
cash account plus the share position plus the short option equals the running P&L at
every step), and a path that never rehedges reproduces a naked short position.

## 11. Attribution

Per step, per path:

```
delta_pnl = Delta * dS
gamma_pnl = 0.5 * Gamma * dS^2
theta_pnl = Theta * dt
vega_pnl  = Vega * d(s_imp)                    # zero when implied vol is held fixed
carry_pnl = r * cash * dt - q * Delta * S * dt # financing and dividends
residual  = realized_step_pnl - sum(the above)
```

This is the P&L explain a desk actually produces: a second-order Taylor expansion of
the position value plus financing. For a delta-hedged book `delta_pnl` nets against
the hedge, leaving the gamma against theta trade-off that the charts display.

The decomposition must close: the six components sum to the realized step P&L to
floating-point tolerance. That is finding F3, and it is also the test that catches a
sign error in the hedger, because a sign error surfaces as a residual the size of the
gamma term rather than as a plausible-looking number.

Why the residual is a finding rather than a nuisance: the expansion is exact to
second order in `dS`, so under GBM the residual is third order and negligible. A jump
is not small, so under Merton the residual is large. The size of the residual is
therefore a direct measurement of how much of the risk delta hedging cannot see.

## 12. Costs and schedules

Cost model: proportional, `k * |dDelta| * S`, with `k` given in basis points. A
fixed per-trade fee is available and defaults to zero.

Schedules:

- **FixedTime(n)**: rehedge at `n` equally spaced times.
- **DeltaBand(h)**: rehedge when `|Delta_target - Delta_held| > h`.
- **Leland(n, k)**: fixed time, but delta computed at the Leland adjusted volatility
  `s_L^2 = s^2 * (1 + Le)` with `Le = sqrt(2 / pi) * k / (s * sqrt(dt))`.
- **WhalleyWilmott(k, lam)**: asymptotic band whose half-width is proportional to
  `(k * S * Gamma^2 / lam)^(1/3)`, implemented following the 1997 paper.

For WhalleyWilmott the acceptance criterion is the scaling, not a hard-coded
constant: the fitted band width must vary as `k^(1/3)` and `Gamma^(2/3)` across a
parameter sweep. This tests the result that matters and does not depend on
transcribing a constant correctly.

## 13. C++ core and RNG parity

The NumPy engine is the readable reference. The C++ engine is the fast path. They
must agree path for path, not merely in distribution, or the reference proves
nothing.

This requires a counter-based pseudo-random generator on both sides: `numpy.random.Philox`
in Python and Philox4x64-10 in C++. Random draws are keyed by the tuple
`(seed, path_index, step_index)` rather than drawn from a running stream. Two
consequences, both wanted:

1. The engines produce bitwise-identical normal draws, so parity is an exact
   assertion rather than a statistical one.
2. Any single path is independently addressable, so when the engines do disagree the
   failure can be reproduced on one path instead of ten million.

Normal variates are produced by the inverse-CDF method on both sides, using the same
rational approximation, because Box-Muller and Ziggurat consume raw draws at
different rates and would break parity even with an identical bit stream.

Build: CMake plus nanobind, exposed through `uv` as an editable build. If the build
is unavailable the Python engine remains fully functional and the CLI falls back with
a warning, so a broken toolchain never blocks the rest of the project.

## 14. Protocol layer

**Pre-registration.** An experiment is a TOML file declaring the model, parameters,
the `N` grid, costs, seeds, and the hypothesis in prose. `vl run` hashes the resolved
configuration and refuses to execute a configuration that differs from the file on
disk. The hypothesis field is mandatory and is copied into the run record, so the
claim is fixed before the number exists.

**Ledger.** sqlite at `.vollab/ledger.db`:

```sql
CREATE TABLE runs (
  run_id      TEXT PRIMARY KEY,
  ts          TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  config_toml TEXT NOT NULL,
  hypothesis  TEXT NOT NULL,
  git_commit  TEXT NOT NULL,
  git_dirty   INTEGER NOT NULL,
  engine      TEXT NOT NULL,
  metrics     TEXT NOT NULL,
  artifacts   TEXT NOT NULL,
  runtime_s   REAL NOT NULL
);
```

Arrays go to parquet under `.vollab/runs/<run_id>/`. `git_dirty` is recorded rather
than enforced: a run from a dirty tree is legal but permanently marked as such.

Deliberately not carried over from the earlier backtester design: the out-of-sample
lockbox and the deflated Sharpe ratio. Both exist to police repeated testing against
one fixed sample. Here the truth is analytic and data is generated on demand, so they
would be ceremony rather than protection. They return in Phase B, where the data is a
finite set of real Deribit chains.

## 15. Statistics

Comparisons between two schedules are paired: both are run on identical paths from
identical seeds, and the difference is taken per path. Reported as a mean difference
with a bootstrap confidence interval over paths, not as two separate point estimates.
A comparison whose interval straddles zero is reported as straddling zero.

Monte Carlo standard errors accompany every mean in every chart and table. An
estimate without its standard error is treated as a defect.

## 16. CLI

```
vl price   --K 100 --T 0.25 --vol 0.6          Black-Scholes price and greeks
vl paths   show --model merton --seed 7        inspect sample paths
vl run     configs/discretisation.toml         pre-registered experiment
vl compare <run_a> <run_b>                     paired bootstrap on the difference
vl bench                                       NumPy against C++ speedup table
vl view    <run_id>                            textual viewer
vl ledger                                      list runs with config hashes
```

`vl run` prints its report into scrollback as plotext charts and writes the run
record. Commands are numbered in run order in `--help`.

Charts: `sd(PnL)` against `N` in log-log with a -0.5 reference line; P&L histograms
overlaid by model; attribution as stacked bars; the cost against frequency U curve;
delta and underlying paths for a single sample path.

## 17. TUI

`vl view <run_id>` opens a Textual application that reads saved artifacts only. It
never runs a simulation. Panels: the charts above, the configuration, and the
hypothesis as recorded. A compare mode places two runs side by side.

Keeping the viewer read-only is what allows the engine to have no interactive code
paths and therefore to be fully testable headless.

## 18. Testing strategy

| Test | Asserts |
|------|---------|
| Analytic pricing | Put-call parity to 1e-12; greeks against central differences to 1e-6 relative; implied volatility inverse recovers input to 1e-10 |
| Model ground truth | MC price converges to Black-Scholes (GBM), the Merton series (jumps), and the characteristic-function price (Heston), each within three Monte Carlo standard errors |
| Engine parity | C++ and NumPy produce bitwise-identical paths and P&L for the same seed |
| Law recovery | Fitted slope of `log sd(PnL)` against `log N` is -0.5 within tolerance, over `N` from 2^4 to 2^12 |
| Zero mean | Perfect hedge with no costs has mean P&L not distinguishable from zero by a t-test |
| Attribution closure | Components sum to realized P&L to 1e-10 relative |
| Jump floor | Under Merton, `sd(PnL)` at `N = 2^12` is not below a fixed fraction of its value at `N = 2^8` |
| Cash accounting | Cash plus shares minus option value equals running P&L at every step, every path |
| Properties (hypothesis) | Costs are monotonically non-increasing in P&L; more frequent hedging does not increase variance in the zero-cost GBM case; payoffs are non-negative; band schedules never trade more than fixed-time at the same grid |
| Golden run | Fixed seed and config reproduce a committed metrics snapshot |

The law-recovery and attribution-closure tests are the two that make the engine
trustworthy. If either regresses, no result in REPORT.md can be believed.

## 19. Milestones

| ID | Days | Deliverable | Done when |
|----|------|-------------|-----------|
| M1 | 1-3 | uv project, Black-Scholes price and greeks, CLI skeleton | Analytic pricing tests green, `vl price` prints a correct chain |
| M2 | 4-7 | NumPy GBM paths, hedge simulator, cash accounting | Zero-mean, cash-accounting and -0.5 slope tests green. F1 exists |
| M3 | 8-10 | Attribution, costs, the four schedules | Attribution closure green, F3 and F5 measurable as numbers; their charts land at M4 |
| M4 | 11-13 | plotext charts, `vl run`, pre-registration, ledger | A pre-registered config produces a chart and a ledger row |
| M5 | 14-17 | C++ core, nanobind, Philox parity, `vl bench` | Parity test green, speedup table in the README |
| M6 | 18-20 | Heston and Merton paths and their ground-truth prices | Jump-floor test green. F4 exists |
| M7 | 21-24 | REPORT.md with F1 to F5, README, Textual viewer | Every finding has a chart, a number with a standard error, and a command that regenerates it |

M2 is the first state worth showing. M5 is the state at which the C++ claim has
evidence behind it. Each milestone leaves the repository in a demonstrable condition.

## 20. Risks

| Risk | Mitigation |
|------|------------|
| C++ and NumPy diverge and the cause is untraceable | Counter-based RNG keyed by path and step, so any disagreement reproduces on one path |
| nanobind or CMake build fails on the target machine | Python engine is complete on its own; the C++ path is opt-in and the CLI falls back with a warning |
| Heston discretisation bias is mistaken for a hedging result | The characteristic-function price gates every Heston finding; no Heston chart ships until that test is green |
| Scope creeps toward Phase B before Phase A ships | `surface` and `mm` are named non-goals in this document and require their own spec cycles |
| Milestones slip past the application window | Milestone order is chosen so M2, M4, M5 and M7 are each independently presentable |

## 21. Seams for later phases

Phase B (`surface`) and Phase C (`mm`) reuse, unchanged: `protocol/` (pre-registration
and ledger), `render/` (charts and report composition), `pricing/` (Black-Scholes and
greeks), the C++ build and the Philox keying scheme.

What Phase A must expose for them: pricing and greeks as free functions independent
of the hedging code; the ledger keyed by an opaque configuration hash rather than by
anything hedge-specific; and the chart helpers taking plain arrays rather than
`HedgeResult`.

## References

- Boyle, P. and Emanuel, D. (1980). Discretely adjusted option hedges.
- Leland, H. (1985). Option pricing and replication with transactions costs.
- Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Merton, R. (1976). Option pricing when underlying stock returns are discontinuous.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model.
- Ahmad, R. and Wilmott, P. (2005). Which free lunch would you like today, sir?
