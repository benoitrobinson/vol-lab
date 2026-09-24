"""Graded backtesting exercises: commit to a prediction, then run the experiment.

The correct answer is never stored. Each exercise runs the real engine when it is
checked and reads the answer off the measurement, so an exercise cannot drift
from the code, and on lob-lab it grades against whatever data has been recorded.
The explanation says why the answer is what the mechanism predicts; the measured
lines say what this run actually found, and the two are allowed to disagree.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from vollab.tui import siblings

# Paths per experiment. Tests lower it; the panel uses enough for the answer to
# be stable across seeds.
PATHS = {"hedge": 4000, "mm": 2000}


@dataclass(frozen=True)
class Exercise:
    key: str
    tab: str             # which page it exercises
    title: str
    question: str
    options: tuple
    run: Callable        # () -> (index of the correct option, measured lines)
    why: str


# ------------------------------------------------------------------ grading
# Pure functions from a measurement to the correct option, kept apart from the
# runs so they are testable without simulating anything.

def nearest_ratio(ratio, choices):
    """The choice closest to `ratio` on a log scale."""
    return int(np.argmin([abs(np.log(ratio / c)) for c in choices]))


def direction(before, after, tol=0.03):
    """0 up, 1 down, 2 within `tol` relative of each other."""
    if abs(after - before) <= tol * max(abs(before), 1e-12):
        return 2
    return 0 if after > before else 1


def best_or_tie(values, rel_tol):
    """Index of the largest value, or len(values) when they are within
    `rel_tol` of each other relative to their mean magnitude."""
    values = np.asarray(values, float)
    if np.ptp(values) <= rel_tol * np.abs(values).mean():
        return len(values)
    return int(np.argmax(values))


# ---------------------------------------------------------------- vol-lab

def _hedge(every, bps=0.0):
    from vollab.hedge.config import Contract, HedgeConfig, VolSpec
    from vollab.hedge.schedule import FixedTime
    from vollab.hedge.simulator import simulate
    n = PATHS["hedge"]
    return simulate(HedgeConfig(
        contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(every), n_mon=512,
        cost_bps=bps, n_paths=n, seed=11, chunk_paths=n)).pnl


def _mm(strategy, gam=0.1, phi=0.0, seed=5):
    from vollab.mm.quoting import DealerParams, MarketParams
    from vollab.mm.simulate import simulate_mm
    return simulate_mm(MarketParams(phi=phi), DealerParams(gam=gam), strategy,
                       seed=seed, n_paths=PATHS["mm"])


def run_frequency():
    every_step, every_4th = _hedge(1), _hedge(4)
    ratio = every_step.std() / every_4th.std()
    return nearest_ratio(ratio, (0.25, 0.5, 1.0, 2.0)), [
        f"sd of P&L, hedged every 4th step   {every_4th.std():.4f}",
        f"sd of P&L, hedged every step       {every_step.std():.4f}",
        f"ratio                              {ratio:.3f}",
    ]


def run_costs():
    from vollab.metrics.bootstrap import paired_bootstrap
    every_step, every_8th = _hedge(1, 10.0), _hedge(8, 10.0)
    d, lo, hi = paired_bootstrap(every_step, every_8th, n_boot=800)
    answer = 2 if lo <= 0.0 <= hi else (0 if d > 0 else 1)
    return answer, [
        f"every step   mean {every_step.mean():+.4f}   sd {every_step.std():.4f}",
        f"every 8th    mean {every_8th.mean():+.4f}   sd {every_8th.std():.4f}",
        f"difference   {d:+.4f}  95% interval [{lo:+.4f}, {hi:+.4f}], same paths",
    ]


def run_aversion():
    low, high = _mm("avellaneda_stoikov", gam=0.1), _mm("avellaneda_stoikov", gam=1.0)
    sd_down = high.pnl.std() < low.pnl.std()
    mean_down = high.pnl.mean() < low.pnl.mean()
    answer = {(True, False): 0, (True, True): 1, (False, True): 2,
              (False, False): 3}[(sd_down, mean_down)]
    return answer, [
        f"gamma 0.1   mean {low.pnl.mean():7.2f}   sd {low.pnl.std():6.2f}"
        f"   largest |inventory| {low.inventory_max_abs.mean():5.2f}",
        f"gamma 1.0   mean {high.pnl.mean():7.2f}   sd {high.pnl.std():6.2f}"
        f"   largest |inventory| {high.inventory_max_abs.mean():5.2f}",
    ]


STRATS = ("avellaneda_stoikov", "symmetric", "glft")


def run_informed():
    runs = {s: _mm(s, phi=0.3) for s in STRATS}
    marks = [runs[s].markout_per_fill() for s in STRATS]
    # Markouts are negative under informed flow; the best is the least negative.
    return best_or_tie(marks, rel_tol=0.10), [
        f"{s:20s} markout per fill {m:+.5f}   mean P&L {runs[s].pnl.mean():7.2f}"
        for s, m in zip(STRATS, marks)
    ] + [f"spread across strategies {np.ptp(marks):.5f}"]


def run_pairing():
    from vollab.metrics.bootstrap import paired_bootstrap
    skew, control = _mm("avellaneda_stoikov"), _mm("symmetric")
    control_elsewhere = _mm("symmetric", seed=99)
    d, lo, hi = paired_bootstrap(skew.pnl, control.pnl, n_boot=800)
    d2, lo2, hi2 = paired_bootstrap(skew.pnl, control_elsewhere.pnl, n_boot=800)
    ratio = (hi2 - lo2) / (hi - lo)
    answer = 0 if ratio < 0.95 else 1 if ratio < 1.05 else 2 if ratio < 1.5 else 3
    return answer, [
        f"same paths        {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]   width {hi - lo:.3f}",
        f"different paths   {d2:+.3f}  [{lo2:+.3f}, {hi2:+.3f}]   width {hi2 - lo2:.3f}",
        f"the unpaired interval is {ratio:.2f}x as wide",
    ]


# ---------------------------------------------------------------- lob-lab

def _study(*args):
    with tempfile.TemporaryDirectory() as out:
        return siblings.run_study(siblings.lob_home(), Path(out), *args)


def _touch(runs, model, field):
    return next(float(r[field]) for r in runs
                if r["quoter"] == "touch" and r["model"] == model)


def run_latency():
    fast, _ = _study("--latency-ms", "10")
    slow, _ = _study("--latency-ms", "500")
    a, b = fast["fill_inflation_mean"], slow["fill_inflation_mean"]
    return direction(a, b), [
        f"10 ms    fill-at-touch hands out {a:.2f}x the fills of the queue",
        f"500 ms   fill-at-touch hands out {b:.2f}x the fills of the queue",
        f"on {fast['days']} recorded day{'s' * (fast['days'] != 1)}",
    ]


def run_markout():
    _, runs = _study()
    naive = _touch(runs, "naive", "markout_1s_bps")
    queue = _touch(runs, "pessimistic", "markout_1s_bps")
    answer = 2 if abs(naive - queue) < 0.05 else (0 if queue < naive else 1)
    return answer, [
        f"fills fill-at-touch grants    markout 1s {naive:+.3f} bps",
        f"fills a queue model grants    markout 1s {queue:+.3f} bps",
    ]


def run_optimistic():
    head, _ = _study()
    ratio = head["cancel_position_sensitivity"]["ahead100"]["fill_inflation"]
    answer = 0 if ratio > 1.05 else 1 if ratio < 0.95 else 2
    return answer, [
        f"every cancel assumed to come from ahead of us: fill-at-touch still hands"
        f" out {ratio:.2f}x the fills",
    ]


def run_ofi_horizon():
    ofi = siblings.run_ofi(siblings.lob_home())
    rows = [r for r in ofi["horizons"] if r["sign_pct"] is not None]
    best = max(rows, key=lambda r: r["sign_pct"])
    order = [500, 1000, 5000, 30000]
    return order.index(best["horizon_ms"]), [
        f"{r['horizon_ms'] / 1000:>5g}s   right {r['sign_pct']:5.1f}% of {r['moved']} moves"
        for r in rows
    ]


EXERCISES = [
    Exercise(
        "frequency", "hedge", "Hedge four times as often",
        "A short call is delta hedged every 4th step of a 512-step grid. Hedge "
        "every step instead. By what factor does the standard deviation of the "
        "terminal P&L change?",
        ("a quarter", "a half", "unchanged", "it doubles"), run_frequency,
        "Each rehedge interval leaves a gamma error proportional to (z^2 - 1) dt, "
        "whose variance scales with dt^2. Summing N of them gives variance "
        "proportional to 1/N, so the sd goes as 1/sqrt(N): four times the "
        "rehedges halves it. This is finding 1, and the reason no desk hedges "
        "continuously: halving the risk costs four times the trading."),
    Exercise(
        "costs", "hedge", "Hedging into 10 bps of costs",
        "Same short call, now paying 10 bps on every hedge trade. Which schedule "
        "has the better mean P&L: hedging every step, or every 8th step?",
        ("every step", "every 8th step", "no difference within noise"), run_costs,
        "Costs are paid on turnover, and turnover grows with frequency while the "
        "risk only falls as 1/sqrt(N). Hedging every step buys a smaller sd with a "
        "mean that bleeds away in costs. The optimum under costs is a trade-off, "
        "not the finest grid, which is why finding 5 ends at a band rather than a "
        "frequency."),
    Exercise(
        "aversion", "making", "Ten times the inventory aversion",
        "An Avellaneda-Stoikov market maker raises its risk aversion gamma from 0.1 "
        "to 1.0. What happens to its P&L?",
        ("sd down, mean up", "sd down, mean down", "sd up, mean down",
         "sd up, mean up"), run_aversion,
        "A higher gamma leans the quotes harder against inventory, so the position "
        "stays smaller and the P&L varies less. It also widens the spread and "
        "gives away fills to get flat, so the mean falls with it. Risk aversion is "
        "not free money; it is a price paid in edge for a narrower distribution."),
    Exercise(
        "informed", "making", "Which rule survives informed flow?",
        "Order flow becomes informed: 30% of fills come from traders who know the "
        "next move (phi = 0.3). Which quoting rule has the best markout per fill?",
        ("Avellaneda-Stoikov", "symmetric control", "GLFT",
         "all within 10% of each other"), run_informed,
        "A quoting rule is a function of your inventory. Whether the next fill is "
        "informed is a property of the counterparty, which none of the three rules "
        "can see, so they are adversely selected alike. This is finding 12, and "
        "the premise of lob-lab: only a signal about the fill itself, such as order "
        "flow imbalance, can do anything about it."),
    Exercise(
        "pairing", "making", "Paired against unpaired",
        "Compare inventory skew against the symmetric control twice: on the same "
        "simulated paths, and on paths from a different seed. How much wider is "
        "the bootstrap interval of the difference when the paths are not shared?",
        ("narrower", "about the same", "up to 1.5x wider", "more than 1.5x wider"),
        run_pairing,
        "On shared paths both strategies meet the same price moves, so the noise "
        "they have in common cancels in the difference. On different paths it does "
        "not, and the interval pays for it. This is why vl compare refuses to pair "
        "runs that did not share paths: a backtest comparison is only as sharp as "
        "the noise its arms have in common."),
    Exercise(
        "latency", "book", "Half a second of latency",
        "On the recorded Deribit data, the quoter's orders arrive 500 ms late "
        "instead of 10 ms. Does fill-at-touch overstate the fills by more or by "
        "less?",
        ("more", "less", "about the same, within 3%"), run_latency,
        "Fill-at-touch fills you whenever a trade prints at your price, so latency "
        "barely touches it. A queue model has to wait for the queue ahead to clear, "
        "and a late order joins the back of a longer queue. The queue loses more "
        "fills than the naive model does, so the overstatement grows. The slower "
        "you are, the more a naive backtest flatters you."),
    Exercise(
        "markout", "book", "Whose fills are worse?",
        "Compare the fills each model grants the same join-the-touch quoter. Which "
        "fills have the worse markout one second later?",
        ("the queue model's", "fill-at-touch's", "the same within 0.05 bps"),
        run_markout,
        "In a queue your order is reached when everything ahead of it has traded or "
        "cancelled, which happens most often just before the price moves through "
        "your level. The fills a queue grants are therefore selected towards the "
        "bad ones. Fill-at-touch also hands out the harmless fills that never "
        "reached you, which dilutes the markout. This is hypothesis H3 of the "
        "preregistration."),
    Exercise(
        "optimistic", "book", "The most optimistic queue",
        "Assume every cancellation came from ahead of your order, the most "
        "generous queue model there is. Does fill-at-touch still hand out more "
        "fills than the queue?",
        ("yes, more", "no, fewer", "about the same"), run_optimistic,
        "Even when every cancel moves you forward, you still wait behind the size "
        "that was resting before you arrived. Fill-at-touch ignores that size "
        "altogether. The assumption changes how much fill-at-touch overstates, not "
        "whether it does, which is why the study reports the whole sweep instead "
        "of one number."),
    Exercise(
        "ofi", "book", "How far ahead does imbalance see?",
        "Order flow imbalance over the last second is measured against the next "
        "move of the mid. Over which horizon does its sign point the right way most "
        "often?",
        ("0.5 s", "1 s", "5 s", "30 s"), run_ofi_horizon,
        "Imbalance is pressure on the touch that has not yet moved the price. It "
        "should say most about the next second or so and fade as other flow "
        "arrives, so the longer horizons drift back towards a coin flip. Read the "
        "counts: on a few minutes of data a horizon rests on a few dozen moves, and "
        "a few points of hit rate is inside the noise."),
]

BY_KEY = {e.key: e for e in EXERCISES}


def grade(exercise, choice):
    """Run the experiment and mark `choice`. Returns the text to show."""
    try:
        answer, measured = exercise.run()
    except siblings.Unavailable as exc:
        return f"cannot run this one yet:\n{exc}"
    right = choice == answer
    return "\n".join([
        ("RIGHT" if right else "NOT QUITE") + f"   you said: {exercise.options[choice]}"
        + ("" if right else f"   measured: {exercise.options[answer]}"),
        "",
        "MEASURED",
        *[f"  {line}" for line in measured],
        "",
        "WHY",
        exercise.why,
    ])
