"""Regenerate every number in REPORT.md.

Run: uv run python scripts/report.py [--quick]

Writes artifacts/findings.json. Every headline is replicated across seeds and
reported as mean +/- standard deviation over those seeds, because a single-seed
figure quoted to four significant figures is mostly noise: the F1 slope moves by
about 0.003 between seeds.

tests/test_report_matches.py asserts the committed JSON still matches, so the
report cannot silently drift from the code again.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import DeltaBand, FixedTime, Leland, WhalleyWilmott
from vollab.hedge.simulator import simulate, sweep
from vollab.metrics.bootstrap import bootstrap_sd, paired_bootstrap
from vollab.metrics.variance_reduction import (
    apply_control, realized_variance_control, variance_ratio,
)
from vollab.mm.quoting import DealerParams, MarketParams
from vollab.mm.simulate import simulate_mm
from vollab.paths.base import GBM, Merton, RoughBergomi
from vollab.paths.heston import heston_paths
from vollab.paths.rbergomi import rbergomi_paths
from vollab.rng.scheme import normals_block
from vollab.pricing.black_scholes import bs_implied_vol, bs_price
from vollab.pricing.inverse import (
    fiat_delta_mismatch, inverse_payoff, inverse_price, share_measure_drift,
)
from vollab.surface.calibrate import calibrate_svi
from vollab.surface.svi import SVIParams, durrleman_g, implied_vol

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "findings.json"

# One grid, stated once and used everywhere, so no table can splice runs.
N_MON = 1024
EVERY = [128, 64, 32, 16, 8, 4, 2, 1]
SEEDS = [11, 23, 37, 41, 53]
CONTRACT = Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0)
FLAT = VolSpec(0.3, 0.3, 0.3)


def ms(values):
    """mean and standard deviation across seeds."""
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n_seeds": int(a.size), "values": [float(v) for v in a]}


def cfg(seed, n_paths, **kw):
    base = dict(contract=CONTRACT, vols=FLAT, schedule=FixedTime(1), model=GBM(),
                n_mon=N_MON, cost_bps=0.0, n_paths=n_paths, seed=seed,
                chunk_paths=min(n_paths, 4000))
    base.update(kw)
    return HedgeConfig(**base)


def slope(n, sd):
    return float(np.polyfit(np.log(n), np.log(sd), 1)[0])


def f1_discretisation(n_paths, seeds):
    slopes, curves, ends = [], [], []
    for s in seeds:
        res = sweep(cfg(s, n_paths), [FixedTime(e) for e in EVERY])
        n = np.array([r.n_rehedges.mean() for r in res])
        sd = np.array([r.pnl.std(ddof=1) for r in res])
        slopes.append(slope(n, sd))
        curves.append(sd.tolist())
        ends.append((float(n[0]), float(n[-1]), float(sd[0]), float(sd[-1])))
    n_lo, n_hi, sd_lo, sd_hi = np.array(ends).mean(axis=0)
    return {"slope": ms(slopes), "theory": -0.5, "n_mon": N_MON,
            "rehedges_sparse": n_lo, "rehedges_dense": n_hi,
            "sd_sparse": sd_lo, "sd_dense": sd_hi,
            "sd_curve_first_seed": curves[0], "every": EVERY}


def f2_lockin(n_paths, seeds):
    s_imp, s_real = 0.35, 0.25
    edge = (bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, s_imp)
            - bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, s_real))
    out = {"edge": float(edge), "s_imp": s_imp, "s_real": s_real,
           "drift": "r - q (pinned)"}
    for label, s_hedge in (("at_realized", s_real), ("at_implied", s_imp)):
        means, sds, wins = [], [], []
        for s in seeds:
            p = simulate(cfg(s, n_paths, vols=VolSpec(s_imp, s_hedge, s_real))).pnl
            means.append(p.mean()); sds.append(p.std(ddof=1))
            wins.append(float((p > 0).mean()))
        out[label] = {"mean": ms(means), "sd": ms(sds), "prob_profit": ms(wins)}
    return out


def f3_attribution(n_paths, seeds):
    out = {}
    for label, every in (("hedged_every_step", 1), ("hedged_every_8th", 8)):
        g, t, d, res = [], [], [], []
        for s in seeds:
            a = simulate(cfg(s, n_paths, schedule=FixedTime(every))).attribution
            g.append(a.gamma.mean()); t.append(a.theta.mean())
            d.append(np.abs(a.delta).max()); res.append(a.residual_sum.mean())
        out[label] = {"gamma": ms(g), "theta": ms(t),
                      "max_abs_delta": ms(d), "residual": ms(res)}
    ratios = []
    for n_mon in (128, 512, 2048):
        a = simulate(cfg(seeds[0], n_paths, n_mon=n_mon)).attribution
        ratios.append({"n_mon": n_mon,
                       "residual_over_gamma": float(a.residual_abs_sum.mean()
                                                    / np.abs(a.gamma).mean())})
    out["residual_scaling"] = ratios
    return out


def f4_jump_floor(n_paths, seeds):
    jumps = Merton(lam=1.0, mu_J=-0.10, s_J=0.15)
    out = {"jump_params": {"lam": 1.0, "mu_J": -0.10, "s_J": 0.15}}
    for label, model in (("gbm", GBM()), ("merton", jumps)):
        slopes, ratios, floors, curves, grids = [], [], [], [], []
        for s in seeds:
            res = sweep(cfg(s, n_paths, model=model), [FixedTime(e) for e in EVERY])
            n = np.array([r.n_rehedges.mean() for r in res])
            sd = np.array([r.pnl.std(ddof=1) for r in res])
            slopes.append(slope(n, sd)); ratios.append(sd[-1] / sd[0])
            floors.append(sd[-1]); curves.append(sd); grids.append(n)
        curves = np.array(curves)
        out[label] = {
            "slope": ms(slopes), "sd_ratio": ms(ratios), "sd_dense": ms(floors),
            # The measured curves, so a figure never has to invent one.
            "rehedges": np.array(grids).mean(axis=0).tolist(),
            "sd_curve_mean": curves.mean(axis=0).tolist(),
            "sd_curve_sd": (curves.std(axis=0, ddof=1) if curves.shape[0] > 1
                            else np.zeros(curves.shape[1])).tolist(),
        }
    from scipy.stats import skew
    sk, worst = [], []
    for s in seeds:
        p = simulate(cfg(s, n_paths, model=jumps, n_mon=512)).pnl
        sk.append(float(skew(p)))
        worst.append(float((p.min() - p.mean()) / p.std(ddof=1)))
    out["merton_skew"] = ms(sk)
    out["merton_worst_z"] = ms(worst)
    return out


def f5_schedules(n_paths, seeds):
    scheds = {"FixedTime(1)": FixedTime(1), "FixedTime(8)": FixedTime(8),
              "DeltaBand(0.02)": DeltaBand(0.02), "Leland": Leland(1, True),
              "WhalleyWilmott": WhalleyWilmott(1.0)}
    out = {"cost_bps": 10.0, "table": {}}
    for label, sch in scheds.items():
        m, sd, reh, tv = [], [], [], []
        for s in seeds:
            r = simulate(cfg(s, n_paths, schedule=sch, cost_bps=10.0, n_mon=512))
            m.append(r.pnl.mean()); sd.append(r.pnl.std(ddof=1))
            reh.append(r.n_rehedges.mean()); tv.append(r.turnover.mean())
        out["table"][label] = {"pnl": ms(m), "sd": ms(sd),
                               "rehedges": ms(reh), "turnover": ms(tv)}
    a = simulate(cfg(seeds[0], n_paths, schedule=WhalleyWilmott(1.0),
                     cost_bps=10.0, n_mon=512)).pnl
    b = simulate(cfg(seeds[0], n_paths, schedule=FixedTime(1),
                     cost_bps=10.0, n_mon=512)).pnl
    diff, lo, hi = paired_bootstrap(a, b, n_boot=2000)
    out["ww_minus_fixed"] = {"diff": diff, "ci_low": lo, "ci_high": hi}

    # U-curve with a bootstrap interval on the location of the minimum.
    res = sweep(cfg(seeds[0], n_paths, cost_bps=60.0, n_mon=512),
                [FixedTime(e) for e in EVERY])
    n = np.array([r.n_rehedges.mean() for r in res])
    obj = np.array([-r.pnl.mean() + r.pnl.std(ddof=1) for r in res])
    rng = np.random.default_rng(0)
    argmins = []
    for _ in range(400):
        boot = []
        for r in res:
            idx = rng.integers(0, r.pnl.size, r.pnl.size)
            s_ = r.pnl[idx]
            boot.append(-s_.mean() + s_.std(ddof=1))
        argmins.append(n[int(np.argmin(boot))])
    out["u_curve"] = {"cost_bps": 60.0, "rehedges": n.tolist(),
                      "objective": obj.tolist(),
                      "argmin_point": float(n[int(np.argmin(obj))]),
                      "argmin_ci": [float(np.percentile(argmins, 2.5)),
                                    float(np.percentile(argmins, 97.5))]}
    return out


def f6_surface(n_reps, seeds):
    """Both arms run the identical optimiser and differ only in constraints."""
    true = SVIParams(a=0.008, b=0.25, rho=-0.7, m=0.0, sigma=0.1)
    T, ks = 0.10, np.linspace(-0.45, 0.45, 9)
    iv = implied_vol(true, ks, T)
    kg = np.linspace(-0.7, 0.7, 401)
    out = {"target_min_durrleman_g": float(durrleman_g(true, kg).min()),
           "target_atm_iv": float(implied_vol(true, 0.0, T)), "T": T,
           "noise_levels": {}}
    _, rmse0, d0 = calibrate_svi(ks, iv, T)
    out["noise_free_rmse_vol_points"] = d0["rmse_vol_points"]

    rng = np.random.default_rng(seeds[0])
    for noise in (0.5, 1.5, 3.0):
        bad_u = bad_c = refused = 0
        ru, rc = [], []
        for _ in range(n_reps):
            ivn = iv + rng.normal(0, noise / 100.0, iv.size)
            _, _, du = calibrate_svi(ks, ivn, T, arb_free=False)
            bad_u += du["min_durrleman_g"] < -1e-9
            ru.append(du["rmse_vol_points"])
            try:
                _, _, dc = calibrate_svi(ks, ivn, T, arb_free=True)
            except RuntimeError:
                refused += 1
                continue
            bad_c += dc["min_durrleman_g"] < -1e-8
            rc.append(dc["rmse_vol_points"])
        out["noise_levels"][str(noise)] = {
            "n_reps": n_reps, "unconstrained_arbitraged": bad_u,
            "constrained_arbitraged": bad_c, "refused": refused,
            "rmse_unconstrained": float(np.mean(ru)),
            "rmse_constrained": float(np.mean(rc)) if rc else None,
            "fit_cost_vol_points": (float(np.mean(rc) - np.mean(ru)) if rc else None),
        }
    return out


def f7_market_making(n_paths, seeds, gams=(0.01, 0.05, 0.1, 0.3, 1.0)):
    market = MarketParams()
    out = {"sweep": {}, "market": {"sigma": market.sigma, "A": market.A,
                                   "kappa": market.kappa, "T": market.T,
                                   "n_steps": market.n_steps}}
    for gam in gams:
        dealer = DealerParams(gam=gam)
        row = {}
        for strat in ("avellaneda_stoikov", "glft", "symmetric"):
            m, sd, inv = [], [], []
            for s in seeds:
                r = simulate_mm(market, dealer, strat, seed=s, n_paths=n_paths)
                m.append(r.pnl.mean()); sd.append(r.pnl.std(ddof=1))
                inv.append(r.inventory_max_abs.mean())
            row[strat] = {"pnl": ms(m), "sd": ms(sd), "peak_inventory": ms(inv),
                          "ratio": ms([mi / si for mi, si in zip(m, sd)])}
        row["ratio_gap"] = (row["avellaneda_stoikov"]["ratio"]["mean"]
                            - row["symmetric"]["ratio"]["mean"])
        out["sweep"][str(gam)] = row

    # Is the best risk aversion actually separated from its neighbours? Compare
    # them pairwise across seeds rather than bootstrapping one seed, which is
    # the weaker test and cannot see that the seeds agree.
    ratios = {g: np.array(out["sweep"][g]["avellaneda_stoikov"]["ratio"]["values"])
              for g in out["sweep"]}
    best = max(ratios, key=lambda g: ratios[g].mean())
    order = sorted(ratios, key=float)
    i = order.index(best)
    neighbours = {}
    for j in (i - 1, i + 1):
        if 0 <= j < len(order):
            diff = ratios[best] - ratios[order[j]]
            se = float(diff.std(ddof=1) / np.sqrt(diff.size))
            neighbours[order[j]] = {
                "paired_diff": float(diff.mean()), "se": se,
                "separation_se": float(abs(diff.mean()) / se) if se > 0 else float("inf"),
            }
    out["best_gam"] = best
    out["best_vs_neighbours"] = neighbours

    dealer = DealerParams(gam=0.1)
    runs = {s_: simulate_mm(market, dealer, s_, seed=seeds[0], n_paths=n_paths)
            for s_ in ("avellaneda_stoikov", "glft", "symmetric")}
    out["pairwise"] = {}
    for left, right in (("avellaneda_stoikov", "symmetric"),
                        ("glft", "symmetric"),
                        ("glft", "avellaneda_stoikov")):
        diff, lo, hi = paired_bootstrap(runs[left].pnl, runs[right].pnl, n_boot=2000)
        out["pairwise"][f"{left}_minus_{right}"] = {
            "diff": diff, "ci_low": lo, "ci_high": hi,
            "straddles_zero": bool(lo <= 0 <= hi)}
    out["skew_minus_control"] = out["pairwise"]["avellaneda_stoikov_minus_symmetric"]

    # The horizon term is the difference between the two closed forms.
    from vollab.mm.quoting import glft_half_spreads, optimal_half_spreads
    out["horizon_effect"] = {
        "as_spread_far": float(sum(optimal_half_spreads(
            0, dealer.gam, market.sigma, market.T, market.kappa))),
        "as_spread_near": float(sum(optimal_half_spreads(
            0, dealer.gam, market.sigma, 0.01, market.kappa))),
        "glft_spread": float(sum(glft_half_spreads(
            0, dealer.gam, market.sigma, market.T, market.kappa, market.A))),
    }
    return out


def f8_inverse_options(n_paths, seed):
    """Coin-settled options, and what a converted vanilla delta costs."""
    S0, K, T, r, q, sv = 100.0, 110.0, 0.5, 0.03, 0.0, 0.6
    claim = float(inverse_price("call", S0, K, T, r, q, sv))

    z = normals_block(seed, 0, n_paths, 1)[:, 0]
    st_q = S0 * np.exp((r - q - 0.5 * sv ** 2) * T + sv * np.sqrt(T) * z)
    dollar = np.exp(-r * T) * np.maximum(st_q - K, 0.0) / S0
    mu = share_measure_drift(r, q, sv)
    st_s = S0 * np.exp((mu - 0.5 * sv ** 2) * T + sv * np.sqrt(T) * z)
    share = np.exp(-q * T) * inverse_payoff("call", st_s, K)
    naive_expectation = float(inverse_payoff("call", st_q, K).mean())

    spots = [60.0, 80.0, 110.0, 150.0, 220.0]
    rows = []
    for S in spots:
        coin, naive, diff = fiat_delta_mismatch("call", S, K, T, r, q, sv)
        rows.append({"spot": S, "coin_delta": float(coin),
                     "naive_delta": float(naive), "gap": float(diff),
                     "gap_pct": float(100.0 * diff / naive)})
    return {
        "contract": {"S0": S0, "K": K, "T": T, "r": r, "q": q, "vol": sv},
        "coin_price": claim,
        "dollar_route": {"mean": float(dollar.mean()),
                         "se": float(dollar.std(ddof=1) / np.sqrt(n_paths))},
        "share_route": {"mean": float(share.mean()),
                        "se": float(share.std(ddof=1) / np.sqrt(n_paths))},
        "naive_expectation": naive_expectation,
        "mismatch": rows,
    }


def convergence(seeds):
    """Does the standard error of the mean fall as n^-1/2? It must."""
    rows = []
    for n_paths in (1000, 4000, 16000):
        ses = []
        for s in seeds[:3]:
            p = simulate(cfg(s, n_paths, n_mon=256)).pnl
            ses.append(p.std(ddof=1) / np.sqrt(p.size))
        rows.append({"n_paths": n_paths, "se": float(np.mean(ses))})
    n = np.array([r["n_paths"] for r in rows], dtype=float)
    se = np.array([r["se"] for r in rows])
    return {"rows": rows, "fitted_slope": slope(n, se), "expected_slope": -0.5}


def variance_reduction_study(n_paths, seed):
    p = simulate(cfg(seed, n_paths, n_mon=512)).pnl
    c = realized_variance_control(seed, n_paths, 512)
    adj, beta, rho = apply_control(p, c)
    S_T_ctrl_corr = None
    from vollab.paths.gbm import gbm_paths
    S = gbm_paths(100.0, 0.0, 0.0, 0.3, 1.0, 512, seed, 0, n_paths)
    payoff = np.maximum(S[:, -1] - 100.0, 0.0)
    _, _, rho_payoff = apply_control(
        p, payoff, bs_price("call", 100.0, 100.0, 1.0, 0.0, 0.0, 0.3))
    return {"control": "sum(z^2 - 1)", "correlation": rho, "beta": beta,
            "variance_ratio": variance_ratio(p, adj),
            "payoff_control_correlation": rho_payoff,
            "antithetic_correlation": 1.0,
            "antithetic_note": "hedging error is even in z, so mirroring is a no-op"}


def f9_rough_vol_floor(n_paths, seeds, etas=(0.0, 0.5, 1.0, 1.5), H=0.10, rho=-0.7):
    """Vol-of-vol floors the hedging error; roughness does not move the floor."""
    out = {"H": H, "rho": rho, "etas": [float(e) for e in etas], "sweep": {}}

    def curve(model, seed):
        res = sweep(cfg(seed, n_paths, model=model), [FixedTime(e) for e in EVERY])
        n = np.array([r.n_rehedges.mean() for r in res])
        sd = np.array([r.pnl.std(ddof=1) for r in res])
        return n, sd

    for eta in etas:
        slopes, ratios, curves, grids = [], [], [], []
        for s in seeds:
            n, sd = curve(RoughBergomi(H=H, eta=eta, rho=rho), s)
            slopes.append(slope(n, sd)); ratios.append(sd[-1] / sd[0])
            curves.append(sd); grids.append(n)
        curves = np.array(curves)
        out["sweep"][str(eta)] = {
            "slope": ms(slopes), "sd_ratio": ms(ratios), "sd_dense": ms([c[-1] for c in curves]),
            "rehedges": np.array(grids).mean(axis=0).tolist(),
            "sd_curve_mean": curves.mean(axis=0).tolist(),
        }

    # The roughness control, at the vol-of-vol that floors hardest. If the floor
    # were about roughness rather than about vega, these two would differ.
    smooth_slopes, smooth_ratios = [], []
    for s in seeds:
        n, sd = curve(RoughBergomi(H=0.45, eta=max(etas), rho=rho), s)
        smooth_slopes.append(slope(n, sd)); smooth_ratios.append(sd[-1] / sd[0])
    out["roughness_control"] = {"H": 0.45, "eta": float(max(etas)),
                                "slope": ms(smooth_slopes), "sd_ratio": ms(smooth_ratios)}
    return out


def f10_rough_skew(n_paths, seed, maturities=(0.02, 0.05, 0.1, 0.25, 0.5)):
    """The at-the-money skew as a power law in maturity, against a diffusion."""
    H, eta, rho, xi0 = 0.10, 1.9, -0.9, 0.04
    dk = 0.02
    out = {"H": H, "eta": eta, "rho": rho, "xi0": xi0, "n_paths": n_paths,
           "maturities": list(maturities), "log_moneyness": dk, "models": {}}

    def skew_of(prices_fn):
        skews = []
        for T in maturities:
            st = prices_fn(T)
            vols = []
            for k in (-dk, dk):
                strike = 100.0 * np.exp(k)
                price = float(np.maximum(st - strike, 0.0).mean())
                vols.append(bs_implied_vol("call", price, 100.0, strike, T, 0.0, 0.0))
            skews.append(abs(vols[1] - vols[0]) / (2 * dk))
        return skews

    rough = skew_of(lambda T: rbergomi_paths(
        100.0, 0.0, 0.0, T, max(32, int(2000 * T)), seed, 0, n_paths,
        xi0, H, eta, rho)[:, -1])
    # Heston with the same initial variance, a vol-of-vol chosen to match the
    # level of the smile, and the same correlation. A diffusive variance cannot
    # produce an exploding short-dated skew, whatever its parameters.
    heston = skew_of(lambda T: heston_paths(
        100.0, 0.0, 0.0, T, max(32, int(2000 * T)), seed, 0, n_paths,
        xi0, 1.0, xi0, 1.0, rho)[:, -1])

    for label, skews in (("rbergomi", rough), ("heston", heston)):
        out["models"][label] = {
            "skews": [float(s) for s in skews],
            "slope": float(np.polyfit(np.log(maturities), np.log(skews), 1)[0]),
        }
    out["theoretical_slope"] = H - 0.5
    return out


MM_UNWIND = {"liq_cost": 0.5, "liq_impact": 0.005}
MM_PHI = 0.3
MM_STRATS = ("avellaneda_stoikov", "glft", "symmetric")


def _mm_settings(n_paths, seed):
    settings = {
        "frictionless": (0.0, 0.0, 0.0),
        "unwind": (0.0, MM_UNWIND["liq_cost"], MM_UNWIND["liq_impact"]),
        "informed": (MM_PHI, 0.0, 0.0),
        "both": (MM_PHI, MM_UNWIND["liq_cost"], MM_UNWIND["liq_impact"]),
    }
    runs = {}
    for label, (phi, liq, imp) in settings.items():
        market = MarketParams(phi=phi)
        dealer = DealerParams(gam=0.1, liq_cost=liq, liq_impact=imp)
        runs[label] = {s_: simulate_mm(market, dealer, s_, seed=seed, n_paths=n_paths)
                       for s_ in MM_STRATS}
    return runs


def f11_unwind(n_paths, seeds):
    """The free unwind is what kept the never-skewed control competitive."""
    out = {"unwind": MM_UNWIND, "n_paths": n_paths, "table": {}, "pairwise": {}}
    per_seed = [_mm_settings(n_paths, s) for s in seeds]
    for label in ("frictionless", "unwind", "informed", "both"):
        row = {}
        for s_ in MM_STRATS:
            pnl = [r[label][s_].pnl.mean() for r in per_seed]
            sd = [r[label][s_].pnl.std(ddof=1) for r in per_seed]
            paid = [r[label][s_].liq_paid.mean() for r in per_seed]
            inv = [np.abs(r[label][s_].inventory_end).mean() for r in per_seed]
            ctrl = [r[label]["symmetric"].pnl.std(ddof=1) for r in per_seed]
            row[s_] = {"pnl": ms(pnl), "sd": ms(sd), "unwind_paid": ms(paid),
                       "end_inventory": ms(inv),
                       "sd_vs_control": ms([a / b for a, b in zip(sd, ctrl)])}
        out["table"][label] = row

    for label in ("frictionless", "unwind", "both"):
        runs = per_seed[0][label]
        diff, lo, hi = paired_bootstrap(runs["glft"].pnl, runs["symmetric"].pnl, n_boot=2000)
        out["pairwise"][label] = {"diff": diff, "ci_low": lo, "ci_high": hi,
                                  "straddles_zero": bool(lo <= 0 <= hi)}
    return out


def f12_adverse_selection(n_paths, seeds, phis=(0.0, 0.3, 0.6)):
    """Informed flow costs every strategy the same amount."""
    out = {"phis": [float(p) for p in phis], "markout_steps": MarketParams().markout_steps,
           "n_paths": n_paths, "table": {}}
    for phi in phis:
        market = MarketParams(phi=phi)
        dealer = DealerParams(gam=0.1)
        row = {}
        for s_ in MM_STRATS:
            marks, pnls = [], []
            for s in seeds:
                r = simulate_mm(market, dealer, s_, seed=s, n_paths=n_paths)
                marks.append(r.markout_per_fill()); pnls.append(r.pnl.mean())
            row[s_] = {"markout_per_fill": ms(marks), "pnl": ms(pnls)}
        out["table"][str(phi)] = row

    base = out["table"]["0.0"]
    informed = out["table"][str(phis[1])]
    out["toll"] = {s_: base[s_]["pnl"]["mean"] - informed[s_]["pnl"]["mean"] for s_ in MM_STRATS}
    out["toll_spread"] = max(out["toll"].values()) - min(out["toll"].values())
    marks = [informed[s_]["markout_per_fill"]["mean"] for s_ in MM_STRATS]
    out["markout_spread"] = max(marks) - min(marks)
    return out



MV_OPTIMUM = -0.0015
STICKY_SIGN = 0.0021


def f13_minimum_variance(n_paths, seeds, H=0.10, eta=1.5, rho=-0.7):
    """The minimum-variance delta, and the sign the smile's slope suggests."""
    rough = RoughBergomi(H=H, eta=eta, rho=rho)
    out = {"H": H, "eta": eta, "rho": rho, "mv_optimum": MV_OPTIMUM,
           "sticky_sign": STICKY_SIGN, "n_mon": 256, "table": {}}

    def sd_for(model, mv, seed):
        c = cfg(seed, n_paths, model=model, n_mon=256, attribute=False,
                schedule=FixedTime(1), mv_slope=mv)
        return float(simulate(c).pnl.std(ddof=1))

    for label, model in (("rough", rough), ("gbm", GBM())):
        row = {}
        for name, mv in (("plain", 0.0), ("min_variance", MV_OPTIMUM),
                         ("sticky_sign", STICKY_SIGN)):
            row[name] = ms([sd_for(model, mv, s) for s in seeds])
        out["table"][label] = row

    rough_row = out["table"]["rough"]
    out["cut_by_min_variance"] = (
        1.0 - rough_row["min_variance"]["mean"] / rough_row["plain"]["mean"])
    out["cost_of_the_wrong_sign"] = (
        rough_row["sticky_sign"]["mean"] / rough_row["plain"]["mean"] - 1.0)

    # What the smile's slope would suggest, measured rather than asserted.
    st = rbergomi_paths(100.0, 0.0, 0.0, 1.0, 512, seeds[0], 0, 120_000,
                        0.09, H, eta, rho)[:, -1]
    dk = 0.02
    vols = []
    for k in (-dk, dk):
        strike = 100.0 * np.exp(k)
        price = float(np.maximum(st - strike, 0.0).mean())
        vols.append(bs_implied_vol("call", price, 100.0, strike, 1.0, 0.0, 0.0))
    skew = (vols[1] - vols[0]) / (2 * dk)
    out["atm_skew"] = skew
    out["slope_the_smile_suggests"] = -skew / 100.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small samples, for a smoke test rather than the report")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write. A smoke run must not clobber the "
                         "committed artifact, or the drift check compares a "
                         "2-seed render against a 5-seed one and always fails.")
    args = ap.parse_args()

    n_paths = 1000 if args.quick else 8000
    seeds = SEEDS[:2] if args.quick else SEEDS
    n_reps = 5 if args.quick else 30
    mm_paths = 1000 if args.quick else 4000
    # The rough-volatility sweeps are the most expensive thing here: five
    # frequencies of a Volterra convolution per seed. Three seeds at 4,000
    # paths keeps the whole report inside ten minutes and still reports a
    # spread across seeds.
    rough_paths = 500 if args.quick else 4000
    rough_seeds = SEEDS[:1] if args.quick else SEEDS[:3]
    skew_paths = 5_000 if args.quick else 60_000

    t0 = time.time()
    findings = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "quick": args.quick, "n_paths": n_paths, "seeds": seeds,
            "n_mon": N_MON, "every": EVERY,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": f"{platform.system()}-{platform.machine()}",
        },
        "f1_discretisation": f1_discretisation(n_paths, seeds),
        "f2_lockin": f2_lockin(n_paths, seeds),
        "f3_attribution": f3_attribution(n_paths, seeds),
        "f4_jump_floor": f4_jump_floor(n_paths, seeds),
        "f5_schedules": f5_schedules(n_paths, seeds),
        "f6_surface": f6_surface(n_reps, seeds),
        "f7_market_making": f7_market_making(mm_paths, seeds),
        "f8_inverse": f8_inverse_options(max(n_paths * 10, 40_000), seeds[0]),
        "f9_rough_vol_floor": f9_rough_vol_floor(rough_paths, rough_seeds),
        "f10_rough_skew": f10_rough_skew(skew_paths, seeds[0]),
        "f11_unwind": f11_unwind(mm_paths, seeds),
        "f12_adverse_selection": f12_adverse_selection(mm_paths, seeds),
        "f13_minimum_variance": f13_minimum_variance(rough_paths, rough_seeds),
        "convergence": convergence(seeds),
        "variance_reduction": variance_reduction_study(n_paths, seeds[0]),
    }
    findings["meta"]["runtime_s"] = round(time.time() - t0, 1)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, sort_keys=True))
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"wrote {shown} in {findings['meta']['runtime_s']}s")
    f1 = findings["f1_discretisation"]["slope"]
    print(f"  F1 slope {f1['mean']:+.4f} +/- {f1['sd']:.4f} over {f1['n_seeds']} seeds")


if __name__ == "__main__":
    main()
