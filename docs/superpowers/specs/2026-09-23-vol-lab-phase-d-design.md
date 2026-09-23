# vol-lab: design, phase D

Phase: D (rough volatility in `hedge`, frictions in `mm`)
Date: 2026-09-23
Status: built

## 1. Purpose

Phase A to C measured hedging, smile calibration and quoting under models that were
chosen for being tractable. Two of those choices were doing work the report did not
account for:

- Every path model had a variance that was either constant or a diffusion, so the delta
  hedge was the whole hedge and the error went to zero with frequency. Real surfaces say
  otherwise, and the model that says it loudest is rough volatility.
- The market-making experiment marked terminal inventory at the mid and drew fills that
  were independent of the next price move. Both are subsidies, and both favour the
  strategy the experiment was supposed to be testing against.

Phase D adds the missing model and the missing frictions, and reports what changes.

## 2. Scope

In: a rough Bergomi path model, a priced unwind and informed flow in `mm`, four findings
(9 to 12), their tests, their report blocks, two figures and two lessons.

Out: a rough-Bergomi price or delta (the hedge stays Black-Scholes, which is the point of
finding 9), calibration of any parameter to market data, least-squares Monte Carlo, and
any change to findings 1 to 8, which stay as measured.

## 3. What is added

| unit | responsibility |
|---|---|
| `paths/rbergomi.py` | Volterra process by the hybrid scheme, variance process, spot paths |
| `paths/base.py` | `RoughBergomi(H, eta, rho)` in `MODELS`, so configs and the ledger reach it |
| `rng/scheme.py` | three draws per step for `rbergomi`, fixed like every other model |
| `mm/quoting.py` | `MarketParams.phi`, `MarketParams.markout_steps`, `DealerParams.liq_cost`, `DealerParams.liq_impact` |
| `mm/simulate.py` | informed arrivals, markouts against the mid at the fill, unwind charged at the close, `pnl_gross` and `liq_paid` on the result |

Defaults keep both frictions off, so every phase A to C number is reproduced bit for bit
by the same code. Finding 11 is the comparison between the two settings, not a silent
restatement of finding 7.

## 4. Interfaces

```python
RoughBergomi(H: float, eta: float, rho: float)           # paths.base
rbergomi_paths(S0, r, q, T, n_mon, seed, path_start,     # paths.rbergomi
               n_paths, xi0, H, eta, rho, mu=None) -> (n_paths, n_mon + 1)
volterra_paths(H, n_mon, dt, z1, z2) -> (Y, dW)
variance_paths(T, n_mon, seed, path_start, n_paths, xi0, H, eta) -> (Y, V)

MMResult.pnl_gross, MMResult.liq_paid, MMResult.markout, MMResult.markout_per_fill()
```

`xi0` is the flat forward variance, taken from `VolSpec.s_real ** 2` so a sweep over
realized volatility reaches it the same way Heston's `v0` is reached.

## 5. How the model is gated

There is no closed-form price for rough Bergomi, so the scheme is gated on identities:

| check | what it catches |
|---|---|
| `Var(Y_t) = t^(2H) / (2H)` at four times | a wrong kernel weight or a wrong covariance in the first interval |
| `H = 1/2` reproduces the driving Brownian motion path by path | an off-by-one in the convolution |
| regression of log variance on log time returns `2H` | roughness that is not the roughness asked for |
| `E[V_t] = xi0` | a missing convexity correction in the exponential |
| `eta = 0` prices a call at Black-Scholes, within three standard errors | anything wrong downstream of the variance |
| `E[S_T] = S_0 e^{(r-q)T}` | a broken correlation or drift |

## 6. Findings

- **F9.** Vol-of-vol floors the delta-hedging error. `eta = 0` recovers the square-root
  law; raising `eta` flattens the curve monotonically. `H` does not move the floor, which
  is the control that separates the two knobs.
- **F10.** The at-the-money skew follows a power law in maturity close to `H - 1/2`, while
  a Heston control with the same correlation and initial variance stays nearly flat.
- **F11.** Charging for the unwind flips the mean-P&L comparison between the steady-state
  quoter and the never-skewed control. The dispersion result is unchanged, in all four
  settings.
- **F12.** Informed flow costs all three quoting rules the same markout per fill and the
  same P&L. Inventory skew is not protection against adverse selection.

## 7. Risks

| risk | mitigation |
|---|---|
| The hybrid scheme is subtly wrong and every finding inherits it | six identity gates, two of which are exact path by path |
| The new frictions silently change phase A to C numbers | defaults are off; `test_mm.py` still passes unchanged; `pnl` equals `pnl_gross` when the unwind is free |
| Runtime of the report grows without bound | the rough sweeps run on three seeds and 4,000 paths, stated in `report.py` next to the constant |
| The informed-flow model is mistaken for a model of information | stated as a next-step tilt in the report's limitations and in the lesson text |
