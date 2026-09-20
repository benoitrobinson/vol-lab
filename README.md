# vol-lab

A terminal laboratory for the three questions an options market maker answers every
day: what does it cost me to hedge, is my surface a set of real prices, and where do I
quote?

- **hedge** - what is my P&L if I sell an option and delta hedge it, and how does it
  change with hedging frequency, transaction costs, and the process the underlying
  actually follows?
- **surface** - does my fitted smile imply a probability distribution, or does it
  quietly price butterflies negative?
- **mm** - how hard should quotes lean against inventory?

Every finding below is produced by a test that fails if the engine is wrong, and by a
command you can run yourself. The full write-up, including the jump floor and the
schedule comparison, is in [REPORT.md](REPORT.md).

## Findings

Measured on 8,000 paths, a 2,048-step monitoring grid, a one-year at-the-money call.

**F1. Discretisation error falls as the inverse square root of hedging frequency.**

```
fitted log-log slope   -0.5029        theory -0.5
sd(PnL) at 9 rehedges   3.5511
sd(PnL) at 2037         0.2305
```

This is the Boyle and Emanuel (1980) result and it is the engine's primary
correctness test. Freezing gamma over a step, the per-step error is
`0.5 * Gamma * S^2 * s^2 * dt * (z^2 - 1)`, and since `Var(z^2 - 1) = 2` the variance
of the sum goes as `1/N`.

**F2. Hedging at realized volatility locks in the edge; hedging at implied earns the
same mean with six times the dispersion.**

Selling at 35 implied against 25 realized, where the theoretical edge is 3.9444:

| hedged at | mean P&L | sd | P(profit) |
|-----------|---------|-----|-----------|
| realized  | +3.9434 | 0.1950 | 100% |
| implied   | +3.9584 | 1.2789 | 100% |

The two means agree, which holds only when the simulated drift is `r - q`, so the
experiment pins the drift and the test asserts it. The pathwise sign follows
`s_imp - s_real`: short volatility profits when realized comes in below implied.

**F3. The P&L explain closes, and the residual measures what delta hedging cannot
see.**

```
gamma  -5.913      theta  +5.910      delta  -2.49e-20      residual  +0.0001
```

Delta at `1e-20` is a book that is exactly flat. Gamma and theta almost cancel, which
is the trade a short option is: you are paid time decay to carry short convexity. The
residual is third order in `dS` and shrinks by a factor of two for every fourfold grid
refinement, measured at 0.069, 0.035 and 0.017.

**F5. Total cost against rehedge frequency is U-shaped, and the optimum moves with
the cost rate.**

At 60 bps the risk-adjusted objective bottoms at 65 rehedges, against 4.116 at the
sparse end and 12.521 at the dense end. At zero cost the curve is monotone: hedge as
often as you can.

## Run it

```sh
uv sync
uv run vl price --K 100 --T 1 --vol 0.3         # Black-Scholes price and greeks
uv run vl register configs/discretisation.toml  # stamp a config with its hash
uv run vl run configs/discretisation.toml       # run it, chart it, record it
uv run vl bench                                 # NumPy reference against C++
uv run vl surface --noise 1.5                   # fit a smile, check it for arbitrage
uv run vl mm --gam 0.1                          # quote with and without inventory skew
uv run vl view                                  # browse recorded runs
uv run pytest
```

Editing C++ needs `uv sync --reinstall-package vollab`; the editable rebuild hook is
off deliberately, and `pyproject.toml` records why.

## How it stays honest

A config file carries its own hash. Editing a parameter without re-registering blocks
the run, so the hypothesis is fixed before the number exists. Every run writes a row
recording the config hash, the git commit, whether the tree was dirty, the RNG scheme
version and the library versions. `vl compare` refuses a paired bootstrap between two
runs that did not share paths, because that comparison would be silently wrong.

You own the database, so none of this is security. It is friction, plus an honest
record of what was actually tried.

## Design notes

**Randomness is counter-based.** Paths are keyed by `(seed, path_index)` through a
Philox generator. The key is exactly two 64-bit words, so the step index lives in the
counter rather than the key, and `advance()` is never used because it moves whole
four-output blocks and would silently stride the stream.

**Paths are always generated on the finest grid.** Coarser rehedge frequencies
subsample it, so every frequency in a sweep runs on identical Brownian-nested paths.
An unpaired sweep would widen the error on the fitted slope considerably.

**Two volatilities, never one symbol.** The option is marked at `s_imp`; the traded
hedge is sized at `s_hedge`. They differ whenever F2 is the experiment, so the code
names them `delta_mark` and `delta_hedge` and never conflates them.

**The engine never falls back.** Asking for an engine that is not built raises rather
than quietly running the other one, so a future parity test cannot pass by comparing
an engine against itself.

## Status

All three modules built.

| module | what it does |
|--------|--------------|
| `hedge` | NumPy and C++ engines under tiered parity; GBM, Heston and Merton, each gated by an independent ground-truth price; four hedging schedules; a full P&L explain |
| `surface` | Quasi-explicit SVI calibration under Durrleman, Lee and calendar constraints, with the implied density plotted |
| `mm` | Avellaneda-Stoikov quoting against a never-skewed control on identical paths |

Every finding is in [REPORT.md](REPORT.md), each with a test that fails if the engine
is wrong and a command that regenerates it.

## References

- Boyle, P. and Emanuel, D. (1980). Discretely adjusted option hedges.
- Leland, H. (1985). Option pricing and replication with transactions costs.
- Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.
- Ahmad, R. and Wilmott, P. (2005). Which free lunch would you like today, sir?
