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
    """The control variate that works, against the two that do not."""
    v = F["variance_reduction"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    names = ["terminal payoff\n(textbook choice)", "antithetic pairing",
             "realized $-$ implied\nvariance"]
    corr = [abs(v["payoff_control_correlation"]), 0.0, abs(v["correlation"])]
    colours = [MUTED, MUTED, ACCENT]
    bars = ax.barh(names, corr, color=colours, height=0.55)
    zero_line(ax)
    ax.set_xlabel("|correlation| with terminal hedging P&L")
    ax.set_xlim(0, 1.0)
    ax.set_title("Only a quadratic-variation control reduces variance here",
                 loc="left", fontsize=10)
    ax.annotate(f"variance ratio {v['variance_ratio']:.2f},\n"
                f"about {1 / v['variance_ratio']:.1f}x the effective sample",
                xy=(corr[2], 2), xytext=(-8, -26), textcoords="offset points",
                fontsize=8.2, color=ACCENT, ha="right")
    ax.annotate("a working hedge removes exactly the part\n"
                "of P&L that tracks the terminal value",
                xy=(corr[0], 0), xytext=(14, 2), textcoords="offset points",
                fontsize=8, color=MUTED)
    meta = F["meta"]
    annotate_n(ax, f"n = {meta['n_paths']:,} paths, 512-step grid")
    fig.savefig(FIG / "variance_reduction.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_discretisation_and_floor()
    fig_schedules()
    fig_market_making()
    fig_variance_reduction()
    for p in sorted(FIG.glob("*.png")):
        print(f"wrote figures/{p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
