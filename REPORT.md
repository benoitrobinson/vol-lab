# What happens when you delta hedge a short option

Five findings from a Monte Carlo laboratory. Each one is produced by a test that
fails if the engine is wrong, and by a command that regenerates it.

Unless stated otherwise: a one-year at-the-money call on a 100 underlying, zero rates
and dividends, 30% volatility, 8,000 paths on a 1,024-step monitoring grid. Every
frequency in a sweep runs on the same Brownian-nested paths, so comparisons across
frequency are paired rather than independent samples.

## 1. Hedging error falls as the inverse square root of frequency

| rehedges | 9 | 2037 |
|----------|---|------|
| sd(PnL)  | 3.489 | 0.321 |

Fitted log-log slope: **-0.5015**, against the theoretical -0.5.

Freeze gamma over a step and write `dS/S = s*sqrt(dt)*z`. The P&L contributed by one
interval is `0.5*Gamma*S^2*s^2*dt*(z^2 - 1)`, which has mean zero and, because
`Var(z^2 - 1) = 2`, a variance proportional to `dt^2`. Summing `T/dt` such steps gives
a total variance proportional to `1/N`. This is Boyle and Emanuel (1980), and it is
the engine's primary correctness test: a sign error, a misaligned shift or a broken
path generator all move that slope off -0.5.

## 2. Hedging at realized volatility locks in the edge; hedging at implied earns the
same mean with six times the dispersion

Selling at 35% implied against 25% realized, where `BS(0.35) - BS(0.25) = 3.9444`:

| hedged at | mean P&L | sd | P(profit) |
|-----------|----------|-----|-----------|
| realized  | +3.9434  | 0.1950 | 100% |
| implied   | +3.9584  | 1.2789 | 100% |

Hedge with the volatility the underlying actually has, and the terminal P&L converges
to a number known at inception; what remains random is the path you take to get there,
and the residual dispersion is nothing but the discretisation error of finding 1.
Hedge at implied instead and the P&L becomes genuinely path dependent, earning the
same amount on average but varying six times as much across paths.

Two conditions the implementation respects. The equality of the two means holds only
when the simulated drift is `r - q`, so the experiment pins the drift and the test
asserts it. And the pathwise sign follows `s_imp - s_real`: a short position profits
when realized volatility comes in below implied, which is why every path is profitable
here rather than merely most of them.

## 3. The P&L explain closes, and the residual measures what hedging cannot see

Per step: `delta*dS`, `0.5*Gamma*dS^2`, `Theta*dt`, `Vega*d(s_imp)`, financing, costs,
and whatever is left.

```
delta  -2.49e-20      gamma  -5.913      theta  +5.910      residual  +0.0001
```

Delta at `1e-20` is a book that is exactly flat: the hedge is doing its job. Gamma and
theta almost cancel, which is precisely what a short option is. You are paid time decay
to carry short convexity, and at zero cost with implied equal to realized the two sides
of that bargain are worth the same.

The residual is third order in `dS`, so it shrinks by a factor of two for every fourfold
refinement of the grid: 0.069, 0.035, 0.017. That scaling is the test with real power.
A mislabelled financing term or an unattributed transaction cost would leave a residual
that does not shrink at all.

## 4. Jumps put a floor under the hedging error that frequency cannot reach

Merton jump diffusion, intensity 1.0 per year, log jumps of mean -10% and dispersion
15%, against a geometric Brownian control on the same grid:

| | sd at 9 rehedges | sd at 2037 | ratio | slope |
|---|---|---|---|---|
| GBM | 3.489 | 0.321 | 0.092 | **-0.5015** |
| Merton | 4.855 | 3.078 | 0.634 | **-0.0890** |

Hedging 226 times more often cuts the diffusive error by a factor of eleven and the
jump-driven error by a factor of 1.6. The Merton curve visibly plateaus, at 3.187,
3.135, 3.097 and 3.078 across the last four frequencies.

The mechanism is simple. Between jumps the diffusive error hedges away exactly as in
finding 1. Each Poisson jump delivers a convexity loss that arrives between rehedges
no matter how close together they are, and the variance it contributes is set by the
intensity, the jump law and the horizon, none of which depend on frequency. Total
variance therefore converges to that floor instead of to zero.

The distributional consequence is what a risk manager cares about:

| | skew | worst path, in sd |
|---|---|---|
| GBM | -0.09 | -5.8 |
| Merton | -2.92 | -10.4 |

The attribution says the same thing from the other direction. A jump is not a small
move, so the second-order expansion cannot absorb it, and the worst single-step
residual relative to gamma is 0.0655 under jumps against 0.0031 under GBM, a factor
of 21. The residual concentrates in the one step that contains the jump.

**This is the finding that matters for a crypto derivatives book.** Gap risk is not a
hedging-frequency problem, and no amount of rehedging discipline converts it into one.

## 5. Under costs the optimum is a band, not a frequency

Five schedules, 10 bps proportional cost, 512-step grid, same paths:

| schedule | mean P&L | sd | rehedges | turnover |
|----------|---------|-----|----------|----------|
| FixedTime(1) | -0.8281 | 0.5592 | 510.4 | 824.7 |
| FixedTime(8) | -0.3355 | 1.3165 | 64.8 | 361.9 |
| DeltaBand(0.02) | -0.6377 | 0.6174 | 151.8 | 634.1 |
| Leland | -0.8174 | 0.5051 | 510.5 | 814.3 |
| Whalley-Wilmott | -0.5137 | 0.6888 | 92.3 | 508.7 |

Paired bootstrap, Whalley-Wilmott against hedging every step: **+0.3144** per option,
95% interval **[+0.3045, +0.3247]**. The interval excludes zero comfortably, and the
band gets there while trading 82% less often.

Read the table by columns rather than by rows. Hedging every step buys the tightest
dispersion of the time-based rules and pays the most for it. Hedging every eighth step
has the best mean and more than double the dispersion, which is not a trade most desks
would take. Leland, which hedges on the same schedule as FixedTime(1) but sizes delta
at a cost-adjusted volatility, achieves the lowest dispersion of all at essentially the
same mean. The band rules dominate on the dimension that matters: Whalley-Wilmott
reaches a better mean than any time rule except the sparsest, with dispersion close to
the dense ones, by spending its trades where gamma is large instead of spreading them
evenly across the calendar.

On the pure frequency sweep the same tension shows up as a U: at 60 bps the
risk-adjusted objective bottoms at 65 rehedges, against 4.116 at the sparse end and
12.521 at the dense end. At zero cost the curve is monotone and the advice is trivial,
hedge as often as you can.

## 6. An unconstrained smile fit implies negative probabilities

SVI calibration to a short-dated smile, nine strikes, one month to expiry, fitted
against a target slice verified arbitrage-free before any noise is added:

| quote noise | unconstrained fits implying a negative density | constrained | refused | fit cost |
|-------------|-----------------------------------------------|-------------|---------|----------|
| 0.5 vol pts | 0 of 30 | 0 of 30 | 0 | -0.008 pts |
| 1.5 vol pts | 3 of 30 | 0 of 30 | 0 | -0.021 pts |
| 3.0 vol pts | 7 of 30 | 0 of 26 | 4 | -0.050 pts |

The target admits a perfect fit, and with no noise the calibrator recovers all five
parameters exactly. Add ordinary quote noise and roughly one fit in four starts to
imply a risk-neutral density that goes negative somewhere in the wings. Such a slice
prices a butterfly at a negative number: it is not a slightly worse fit, it is not a
price at all.

Imposing Durrleman's condition during the fit rather than checking it afterwards
removes every violation, and the fit cost is negative. Being arbitrage-free is free
here, because the constrained solve starts from an exact quasi-explicit solution and
the constraint rules out only regions the data never justified.

At the highest noise level, four of thirty noisy slices admit no arbitrage-free SVI
fit at all. The calibrator refuses them rather than returning a slice that quietly is
not a price.

**A note on how this was measured.** An earlier version of this experiment used a
target slice that was itself arbitrageable, and found 38 of 40 unconstrained fits
implying negative density. That number measured nothing: the fits were faithfully
reproducing an inadmissible target. The target above is checked for admissibility
first, so the violations are caused by noise rather than inherited from the data.

## 7. Inventory skew halves the dispersion of a market maker's P&L

A dealer quotes two sides around a simulated mid for one session, fills arriving as a
Poisson process whose intensity decays with distance from the mid. Two strategies on
identical paths: quotes centred on the Avellaneda-Stoikov reservation price, against a
control quoting the same total width but never skewed for inventory.

| strategy | P&L | sd | peak abs inventory | end inventory sd | fills |
|----------|-----|-----|--------------------|------------------|-------|
| inventory skew | +64.96 | 6.74 | 4.0 | 2.96 | 97.0 |
| symmetric control | +67.93 | 13.44 | 10.1 | 8.34 | 91.9 |

Paired difference: **-2.97**, 95% interval **[-3.32, -2.62]**. The skew costs real money
and buys more than it costs. Holding the width fixed between the two arms is what makes
this attributable: the only difference is where the quotes are centred, not how wide
they are.

Sweeping risk aversion shows an interior optimum, which the control does not have:

| risk aversion | skew ratio | control ratio | skew peak inventory | control peak inventory |
|---------------|-----------|---------------|---------------------|------------------------|
| 0.01 | 7.56 | 4.92 | 7.2 | 10.5 |
| 0.05 | 9.43 | 4.97 | 4.8 | 10.3 |
| 0.10 | **9.65** | 5.05 | 4.0 | 10.1 |
| 0.30 | 9.09 | 5.07 | 3.0 | 9.1 |
| 1.00 | 6.43 | 4.59 | 2.1 | 6.8 |

Lean too little and inventory accumulates; lean too hard and the quotes are wide enough
that the flow stops arriving, taking mean P&L from 68 down to 31. The control's ratio
sits near 5.0 at every level, because widening a symmetric quote changes how much you
earn without changing the shape of the risk.

The sign of the skew is the whole mechanism. A long dealer must quote a *closer* ask
and a *further* bid. Inverting it turns a stabiliser into an amplifier: the first
version of this simulation had the sign backwards, and inventory pinned itself against
the position limit on essentially every path while the naive control stayed bounded.
That is what the control is for.

## Engines

The reference engine is NumPy, vectorised over paths. A C++ core built with nanobind
runs the same experiment for geometric Brownian motion on a fixed-time schedule.

Parity between them is asserted in tiers, because bitwise equality is unattainable by
construction rather than by carelessness:

| tier | scope | result |
|------|-------|--------|
| exact | raw Philox uint64 and uniform doubles | bitwise identical |
| tolerance | inverse-CDF normals | 84% bitwise identical, worst gap 8.9e-16 |
| tolerance | terminal P&L | agrees to 4e-13 relative |
| bounded | rehedge decisions | 8% to 22% of paths differ, never by more than two trades |

The normals cannot match exactly because the Cephes tail branch goes through `log()`,
which is not correctly rounded and differs between libms; roughly a quarter of draws
land there. Those last-bit gaps then flip discrete decisions. Where delta saturates at
exactly 0 or 1, one engine books a trade and the other does not, and the engines differ
by a whole trade rather than by a rounding error. That is why the decision tier asserts
a bound rather than equality, and why the P&L tier still agrees to twelve digits: a
trade at a saturated delta moves almost no money.

Speedups, timed against the reference with attribution disabled so both engines do the
same work:

| config | reference | C++ | speedup |
|--------|-----------|-----|---------|
| 512 steps, every step | 0.444s | 0.252s | 1.8x |
| 512 steps, every 8th | 0.425s | 0.142s | 3.0x |
| 2048 steps, every 4th | 1.757s | 0.642s | 2.7x |

Under 2x when hedging every step: a scalar C++ loop does not beat a vectorised `erf`
over twenty thousand paths. The C++ wins where the schedule is sparse, because it skips
the delta evaluation entirely on steps that do not trade while the vectorised engine
computes it and masks. Quoting a single larger number by timing the reference with
attribution on would compare the C++ against a baseline doing strictly more work.

## Reproducing

```sh
uv sync
uv run pytest                                  # every finding above
uv run vl register configs/discretisation.toml
uv run vl run configs/discretisation.toml
uv run vl bench
uv run vl view
```

## References

- Boyle, P. and Emanuel, D. (1980). Discretely adjusted option hedges.
- Merton, R. (1976). Option pricing when underlying stock returns are discontinuous.
- Leland, H. (1985). Option pricing and replication with transactions costs.
- Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model.
- Ahmad, R. and Wilmott, P. (2005). Which free lunch would you like today, sir?
