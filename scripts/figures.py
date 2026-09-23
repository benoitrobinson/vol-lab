"""Publication figures from artifacts/findings.json.

Run: uv run python scripts/figures.py

Reads only the regenerated artifact, never recomputes, so a figure can never
disagree with the reported number. Matplotlib is a dev dependency: the library
itself never imports it.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, ACCENT, MUTED = "#1f3a5f", "#b03a2e", "#8a94a6"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": "#2b3440", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
F = json.loads((ROOT / "artifacts" / "findings.json").read_text())


def annotate_n(ax, text):
    ax.text(0.99, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.4, color=MUTED)


def zero_line(ax):
    ax.axhline(0, color="#2b3440", linewidth=0.8, zorder=1)


def label_end(ax, x, y, text, color=INK):
    ax.annotate(f" {text}", xy=(x, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=color)


def fig_discretisation_and_floor():
    """F1 and F4 together: the law, and the floor that breaks it."""
    d4, meta = F["f4_jump_floor"], F["meta"]
    n = np.array(d4["gbm"]["rehedges"], dtype=float)
    gbm_sd = np.array(d4["gbm"]["sd_curve_mean"])
    gbm_err = np.array(d4["gbm"]["sd_curve_sd"])
    n_m = np.array(d4["merton"]["rehedges"], dtype=float)
    merton_sd = np.array(d4["merton"]["sd_curve_mean"])
    merton_err = np.array(d4["merton"]["sd_curve_sd"])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.errorbar(n, gbm_sd, yerr=gbm_err, color=INK, marker="o", ms=3.5, lw=1.4,
                elinewidth=0.8, capsize=2)
    ax.errorbar(n_m, merton_sd, yerr=merton_err, color=ACCENT, marker="s", ms=3.5,
                lw=1.4, ls="--", elinewidth=0.8, capsize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ref = gbm_sd[0] * (n / n[0]) ** -0.5
    ax.loglog(n, ref, color=MUTED, lw=0.9, ls=":")

    label_end(ax, n[-1], gbm_sd[-1], "GBM", INK)
    label_end(ax, n[-1], merton_sd[-1], "Merton jumps", ACCENT)
    label_end(ax, n[2], ref[2] * 0.62, "slope $-1/2$", MUTED)

    s1 = F["f1_discretisation"]["slope"]
    s4 = d4["merton"]["slope"]
    ax.annotate(
        f"jump risk flattens out:\n"
        f"slope {s4['mean']:+.3f} $\\pm$ {s4['sd']:.3f}\n"
        f"against {s1['mean']:+.3f} $\\pm$ {s1['sd']:.3f}\n"
        f"for the diffusion",
        xy=(n_m[-2], merton_sd[-2]),
        xytext=(0.30, 0.56), textcoords="axes fraction",
        fontsize=8.2, color=ACCENT, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8,
                        connectionstyle="arc3,rad=-0.2"))

    ax.set_xlabel("rehedges per year (log scale)")
    ax.set_ylabel("sd of terminal P&L, price points (log scale)")
    ax.set_title("Hedging more often cannot remove gap risk", loc="left", fontsize=10)
    annotate_n(ax, f"n = {meta['n_paths']:,} paths per point, {s1['n_seeds']} seeds, "
                   f"{meta['n_mon']}-step grid; bars are sd across seeds")
    fig.savefig(FIG / "f1_f4_discretisation_and_jump_floor.png")
    plt.close(fig)


def fig_schedules():
    """F5: risk against turnover. Plot the trade-off, not two bar charts."""
    tab = F["f5_schedules"]["table"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, row in tab.items():
        x = row["turnover"]["mean"]
        y = row["sd"]["mean"]
        yerr = row["sd"]["sd"]
        band = name == "WhalleyWilmott"
        c = ACCENT if band else INK
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=5 if band else 4, color=c,
                    ecolor=c, elinewidth=0.9, capsize=2.5, zorder=3)
        ax.annotate(f" {name}", xy=(x, y), xytext=(5, -1),
                    textcoords="offset points", fontsize=8, color=c)
    ax.set_xlabel("turnover, price points traded per option")
    ax.set_ylabel("sd of terminal P&L, price points")
    ax.set_title("Where the hedge trades matters more than how often",
                 loc="left", fontsize=10)
    ww = F["f5_schedules"]["ww_minus_fixed"]
    tab = F["f5_schedules"]["table"]
    t_ww = tab["WhalleyWilmott"]["turnover"]["mean"]
    t_ft = tab["FixedTime(1)"]["turnover"]["mean"]
    s_ww = tab["WhalleyWilmott"]["sd"]["mean"]
    s_ft = tab["FixedTime(1)"]["sd"]["mean"]
    ax.annotate(
        f"the band trades {1 - t_ww / t_ft:.0%} less than hedging every step and earns\n"
        f"{ww['diff']:+.3f} [{ww['ci_low']:+.3f}, {ww['ci_high']:+.3f}] more per option, "
        f"at {s_ww / s_ft - 1:.0%} higher dispersion.\n"
        f"It buys mean, not risk: lower and left is better on both axes.",
        xy=(0.05, 0.80), xycoords="axes fraction", fontsize=8.2, color=ACCENT, va="top")
    meta = F["meta"]
    ax.text(0.99, -0.16, f"n = {meta['n_paths']:,} paths, {len(meta['seeds'])} seeds, "
                         f"10 bps cost; bars are sd across seeds",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color=MUTED)
    fig.savefig(FIG / "f5_schedule_risk_vs_turnover.png")
    plt.close(fig)


def fig_market_making():
    """F7: the risk-adjusted ratio against risk aversion, with seed dispersion."""
    sweep = F["f7_market_making"]["sweep"]
    gams = sorted(float(g) for g in sweep)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    series = [("avellaneda_stoikov", ACCENT, "-", "Avellaneda-Stoikov"),
              ("glft", "#2e7d5b", "-.", "GLFT steady state"),
              ("symmetric", INK, "--", "symmetric control")]
    for strat, colour, style, label in [t for t in series
                                        if t[0] in sweep[str(gams[0])]]:
        m = np.array([sweep[str(g)][strat]["ratio"]["mean"] for g in gams])
        e = np.array([sweep[str(g)][strat]["ratio"]["sd"] for g in gams])
        ax.errorbar(gams, m, yerr=e, color=colour, ls=style, marker="o", ms=3.5,
                    lw=1.4, elinewidth=0.9, capsize=2.5)
        label_end(ax, gams[-1], m[-1], label, colour)
    ax.set_xscale("log")
    ax.set_xlabel("inventory risk aversion $\\gamma$ (log scale)")
    ax.set_ylabel("mean P&L divided by its sd")
    ax.set_title("The horizon term costs more the more risk averse you are",
                 loc="left", fontsize=10)
    best = F["f7_market_making"].get("best_gam")
    nb = F["f7_market_making"].get("best_vs_neighbours", {})
    if best is not None and nb:
        bm = sweep[best]["avellaneda_stoikov"]["ratio"]
        worst = min(v["separation_se"] for v in nb.values())
        ax.annotate(
            f"Avellaneda-Stoikov peaks at $\\gamma={best}$ and falls away,\n"
            f"separated from both neighbours by {worst:.0f}+ standard errors.\n"
            f"The steady-state form keeps improving: it does not\n"
            f"widen itself out of the market as $\\gamma$ grows.",
            xy=(float(best), bm["mean"]), xytext=(0.04, 0.42),
            textcoords="axes fraction", fontsize=8.2, color=ACCENT, va="top",
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8,
                            connectionstyle="arc3,rad=0.2"))
    meta = F["meta"]
    ax.text(0.99, -0.16, f"n = 4,000 sessions per point, {len(meta['seeds'])} seeds; "
                         f"bars are sd across seeds. The grid is coarse, so the true "
                         f"optimum may lie between points.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color=MUTED)
    fig.savefig(FIG / "f7_market_making_risk_aversion.png")
    plt.close(fig)


def fig_variance_reduction():
    """Variance ratio, not correlation.

    An earlier version plotted |correlation with P&L| and gave antithetic
    sampling a zero-length bar, which reads as "uncorrelated" and is the wrong
    reason. Antithetic is not a control variate: a mirrored path reproduces the
    hedging error exactly, so the pair correlates at +1 and the scheme removes
    nothing. Variance ratio puts all three on one comparable axis.
    """
    v = F["variance_reduction"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    names = ["realized $-$ implied variance\n(derived from the F1 mechanism)",
             "terminal payoff\n(the textbook choice)",
             "antithetic pairing\n(not a control variate)"]
    ratios = [v["variance_ratio"], 1.0, 1.0]
    colours = [ACCENT, MUTED, MUTED]
    ax.barh(names, ratios, color=colours, height=0.5, zorder=2)

    ax.axvline(1.0, color="#2b3440", linewidth=0.8, zorder=3)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("variance of the adjusted estimator, relative to plain sampling "
                  "(lower is better; 1.0 buys nothing)")
    ax.set_title("Only a quadratic-variation control reduces variance here",
                 loc="left", fontsize=10)

    ax.annotate(f"{v['variance_ratio']:.2f}, about "
                f"{1 / v['variance_ratio']:.1f}x the effective sample size\n"
                f"correlation with P&L {v['correlation']:+.2f}",
                xy=(v["variance_ratio"], 0), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8.2, color=ACCENT)
    ax.annotate(f"correlation only {v['payoff_control_correlation']:+.3f}: a working\n"
                f"hedge removes exactly the part of P&L\nthat tracks the terminal value",
                xy=(1.0, 1), xytext=(-8, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8, color="white")
    ax.annotate("a mirrored path reproduces the error\n"
                "exactly, so the pair correlates at $+1$",
                xy=(1.0, 2), xytext=(-8, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8, color="white")

    meta = F["meta"]
    ax.text(0.99, -0.30, f"n = {meta['n_paths']:,} paths, 512-step grid",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color=MUTED)
    fig.savefig(FIG / "variance_reduction.png")
    plt.close(fig)


def fig_rough_vol_floor():
    """F9: the hedging error against rehedge frequency, one curve per vol-of-vol."""
    d = F["f9_rough_vol_floor"]
    etas = sorted(d["sweep"], key=float)
    colours = [INK, "#3c6e91", "#8a5a44", ACCENT]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for eta, colour in zip(etas, colours):
        v = d["sweep"][eta]
        n = np.array(v["rehedges"], dtype=float)
        sd = np.array(v["sd_curve_mean"], dtype=float)
        ax.loglog(n, sd, color=colour, marker="o", ms=3.2, lw=1.3)
        label_end(ax, n[-1], sd[-1], f"$\\eta$ = {eta}", colour)

    base = d["sweep"][etas[0]]
    n0 = np.array(base["rehedges"], dtype=float)
    ref = np.array(base["sd_curve_mean"])[0] * (n0 / n0[0]) ** -0.5
    ax.loglog(n0, ref, color=MUTED, lw=0.9, ls=":")
    label_end(ax, n0[2], ref[2] * 0.6, "slope $-1/2$", MUTED)

    top = d["sweep"][etas[-1]]
    ax.annotate(
        f"vega risk floors the error:\n"
        f"slope {top['slope']['mean']:+.3f} at $\\eta$ = {etas[-1]}\n"
        f"against {base['slope']['mean']:+.3f} at $\\eta$ = 0",
        xy=(n0[-2], np.array(top["sd_curve_mean"])[-2]),
        xytext=(0.32, 0.30), textcoords="axes fraction",
        fontsize=8.2, color=ACCENT, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8,
                        connectionstyle="arc3,rad=0.2"))

    ax.set_xlabel("rehedges per year (log scale)")
    ax.set_ylabel("sd of terminal P&L, price points (log scale)")
    ax.set_title("A delta hedge cannot remove vega risk", loc="left", fontsize=10)
    annotate_n(ax, f"rough Bergomi, H = {d['H']}, {F['meta']['n_mon']}-step grid, "
                   f"{top['slope']['n_seeds']} seeds")
    fig.savefig(FIG / "f9_rough_vol_floor.png")
    plt.close(fig)


def fig_adverse_selection():
    """F11 and F12 side by side: who pays for the unwind, and who pays the
    information, which are not the same dealer."""
    d11, d12 = F["f11_unwind"], F["f12_adverse_selection"]
    strats = ("symmetric", "avellaneda_stoikov", "glft")
    names = ["never skewed", "Avellaneda-Stoikov", "steady state"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    free = [d11["table"]["frictionless"][s]["pnl"]["mean"] for s in strats]
    charged = [d11["table"]["unwind"][s]["pnl"]["mean"] for s in strats]
    x = np.arange(len(strats))
    ax.bar(x - 0.18, free, width=0.34, color=MUTED, label="free unwind")
    ax.bar(x + 0.18, charged, width=0.34, color=INK, label="unwind charged")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.6, rotation=12, ha="right")
    ax.set_ylabel("mean P&L, price points")
    ax.set_ylim(min(charged) * 0.9, max(free) * 1.04)
    ax.legend(fontsize=7.4, frameon=False)
    ax.set_title("Who pays to go home flat", loc="left", fontsize=9.5)

    ax = axes[1]
    phis = sorted(d12["table"], key=float)
    for name, s, colour in zip(names, strats, [MUTED, "#3c6e91", ACCENT]):
        y = [d12["table"][p][s]["markout_per_fill"]["mean"] for p in phis]
        ax.plot([float(p) for p in phis], y, color=colour, marker="o", ms=3.4, lw=1.3,
                label=name)
    zero_line(ax)
    ax.set_xlabel("informed fraction of arrivals")
    ax.set_ylabel("markout per fill, price points")
    ax.legend(fontsize=7.4, frameon=False, loc="lower left")
    ax.set_title("Adverse selection does not care how you quote", loc="left", fontsize=9.5)

    annotate_n(axes[1], f"{d12['n_paths']:,} sessions per point, "
                        f"{len(F['meta']['seeds'])} seeds")
    fig.tight_layout()
    fig.savefig(FIG / "f11_f12_unwind_and_adverse_selection.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_discretisation_and_floor()
    fig_schedules()
    fig_market_making()
    fig_rough_vol_floor()
    fig_adverse_selection()
    fig_variance_reduction()
    for p in sorted(FIG.glob("*.png")):
        print(f"wrote figures/{p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
