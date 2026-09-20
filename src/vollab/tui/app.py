"""vol-lab: the whole laboratory in one panel.

Tabs cover the findings with their charts, an interactive pricer, surface
calibration, a hedging experiment, market making, the engine benchmark, and the
run ledger. Compute-heavy tabs run on a worker thread so the interface never
blocks, and every chart is drawn by the same terminal renderer the CLI uses.
"""

import json
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, Static,
    TabbedContent, TabPane,
)

from vollab.protocol.ledger import Ledger
from vollab.tui.lessons import LESSONS

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACT = ROOT / "artifacts" / "findings.json"

CHART_W, CHART_H = 84, 20


def ansi(text):
    """plotext emits ANSI; Textual renders it through Rich."""
    return Text.from_ansi(text)


def format_run(row):
    """Plain text for one run. Separated from the widget so it is testable."""
    m = json.loads(row["metrics"])
    lines = [
        f"run        {row['run_id'][:12]}",
        f"when       {row['ts']}",
        f"config     {row['config_hash'][:12]}",
        f"paths fp   {row['paths_fingerprint'][:12]}",
        f"engine     {row['engine']}  rng scheme v{row['rng_scheme_version']}",
        f"status     {row['status']}  ({row['n_paths_completed']} paths)",
        f"git        {row['git_commit'][:8] or 'none'}"
        f"{'  DIRTY TREE' if row['git_dirty'] else ''}",
        f"versions   numpy {row['numpy_version']}  scipy {row['scipy_version']}"
        f"  python {row['python_version']}",
        f"runtime    {row['runtime_s']:.2f}s",
        "",
        "hypothesis",
        f"  {row['hypothesis']}",
        "",
        "metrics",
    ]
    for k in sorted(m):
        lines.append(f"  {k:20s} {m[k]:.6f}" if isinstance(m[k], float)
                     else f"  {k:20s} {m[k]}")
    return "\n".join(lines)


def _fmt(d, places=4):
    """mean +/- sd, for a block written by scripts/report.py."""
    if isinstance(d, dict) and "mean" in d:
        return f"{d['mean']:+.{places}f} +/- {d['sd']:.{places}f}"
    return str(d)


def lesson_body(lesson, findings):
    """The lesson text, followed by whatever the artifact measured for it."""
    out = [
        lesson.question,
        "",
        "WHY",
        lesson.mechanism,
        "",
        "SO WHAT",
        lesson.takeaway,
        "",
    ]
    f = (findings or {}).get(lesson.key)
    if f is None:
        out += ["MEASURED", "  run scripts/report.py to populate this"]
        return "\n".join(out)

    out.append("MEASURED")
    if lesson.key == "f1_discretisation":
        out += [f"  fitted slope        {_fmt(f['slope'])}   (theory {f['theory']})",
                f"  rehedges            {f['rehedges_sparse']:.0f} to {f['rehedges_dense']:.0f}"
                f" on a {f['n_mon']}-step grid",
                f"  sd of terminal P&L  {f['sd_sparse']:.3f} to {f['sd_dense']:.3f}"]
    elif lesson.key == "f2_lockin":
        out += [f"  edge, BS(imp) - BS(real)  {f['edge']:.4f}",
                f"  hedged at realized        mean {_fmt(f['at_realized']['mean'])}"
                f"   sd {f['at_realized']['sd']['mean']:.4f}",
                f"  hedged at implied         mean {_fmt(f['at_implied']['mean'])}"
                f"   sd {f['at_implied']['sd']['mean']:.4f}",
                f"  profitable paths          {f['at_implied']['prob_profit']['mean']:.1%}"]
    elif lesson.key == "f3_attribution":
        e1, e8 = f["hedged_every_step"], f["hedged_every_8th"]
        out += [f"  every step   gamma {e1['gamma']['mean']:+.3f}"
                f"  theta {e1['theta']['mean']:+.3f}"
                f"  max|delta| {e1['max_abs_delta']['mean']:.2e}",
                f"  every 8th    gamma {e8['gamma']['mean']:+.3f}"
                f"  theta {e8['theta']['mean']:+.3f}"
                f"  max|delta| {e8['max_abs_delta']['mean']:.3f}",
                "  residual over gamma, as the grid refines:"]
        out += [f"    n_mon {r['n_mon']:5d}   {r['residual_over_gamma']:.4f}"
                for r in f["residual_scaling"]]
    elif lesson.key == "f4_jump_floor":
        out += [f"  GBM control   slope {_fmt(f['gbm']['slope'], 3)}"
                f"   sd ratio {f['gbm']['sd_ratio']['mean']:.3f}",
                f"  Merton jumps  slope {_fmt(f['merton']['slope'], 3)}"
                f"   sd ratio {f['merton']['sd_ratio']['mean']:.3f}",
                f"  terminal skew {f['merton_skew']['mean']:+.2f}"
                f"   worst path {f['merton_worst_z']['mean']:.1f} sd from the mean"]
    elif lesson.key == "f5_schedules":
        out.append(f"  at {f['cost_bps']:.0f} bps:")
        for name, r in f["table"].items():
            out.append(f"    {name:18s} pnl {r['pnl']['mean']:+8.4f}"
                       f"  sd {r['sd']['mean']:7.4f}"
                       f"  rehedges {r['rehedges']['mean']:6.1f}"
                       f"  turnover {r['turnover']['mean']:7.1f}")
        u = f["u_curve"]
        out += ["", f"  U-curve minimum at {u['argmin_point']:.0f} rehedges,"
                    f" bootstrap interval [{u['argmin_ci'][0]:.0f},"
                    f" {u['argmin_ci'][1]:.0f}]"]
    elif lesson.key == "f6_surface":
        out += [f"  target min Durrleman g   {f['target_min_durrleman_g']:+.4f}"
                f"  (admissible before noise)",
                f"  noise-free fit           {f['noise_free_rmse_vol_points']:.5f} vol points",
                ""]
        for lvl, v in sorted(f["noise_levels"].items(), key=lambda kv: float(kv[0])):
            cost = ("n/a" if v["fit_cost_vol_points"] is None
                    else f"{v['fit_cost_vol_points']:+.3f}")
            out.append(f"    {lvl:>4s} vol pts noise   unconstrained "
                       f"{v['unconstrained_arbitraged']:2d}/{v['n_reps']} arbitraged"
                       f"   constrained {v['constrained_arbitraged']}"
                       f"   refused {v['refused']}   cost {cost}")
    elif lesson.key == "f7_market_making":
        for gam in sorted(f["sweep"], key=float):
            row = f["sweep"][gam]
            cells = "   ".join(
                f"{s_.split('_')[0]:>6s} {row[s_]['ratio']['mean']:5.2f}"
                for s_ in row if isinstance(row[s_], dict) and "ratio" in row[s_])
            out.append(f"    gamma {gam:>5s}   {cells}")
        for k, v in f.get("pairwise", {}).items():
            out.append(f"    {k.replace('_minus_', ' minus ').replace('_', ' '):48s}"
                       f" {v['diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    elif lesson.key == "f8_inverse":
        out += [f"  coin price {f['coin_price']:.6f}, confirmed by two routes:",
                f"    dollar payoff under Q   {f['dollar_route']['mean']:.6f}"
                f" +/- {f['dollar_route']['se']:.6f}",
                f"    coin payoff, share measure  {f['share_route']['mean']:.6f}"
                f" +/- {f['share_route']['se']:.6f}",
                f"  naive E[coin payoff] {f['naive_expectation']:.6f}  <- not a price",
                "", "  converting a vanilla delta instead:"]
        out += [f"    spot {r['spot']:6.0f}   coin {r['coin_delta']:.6f}"
                f"   converted {r['naive_delta']:.6f}   off by {r['gap_pct']:.1f}%"
                for r in f["mismatch"]]
    return "\n".join(out)


class VolLabApp(App):
    CSS = """
    Screen { layout: vertical; }
    #runs-table { width: 45%; border: solid $accent; }
    #run-detail { width: 1fr; border: solid $accent; padding: 1; }
    #lesson-list { width: 34; border: solid $accent; }
    #lesson-body { width: 1fr; padding: 1 2; }
    .controls { height: auto; padding: 1 2; }
    .controls Input { width: 14; margin-right: 2; }
    .controls Label { padding: 1 1 0 0; }
    .out { padding: 1 2; }
    Button { margin-right: 2; }
    """
    BINDINGS = [
        ("q", "quit", "quit"),
        ("r", "refresh", "run this tab"),
    ]

    def __init__(self, db_path=None):
        super().__init__()
        self.db_path = db_path
        self.rows = Ledger(db_path).all() if db_path else []
        self.findings = (json.loads(ARTIFACT.read_text())
                         if ARTIFACT.exists() else {})

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="lessons"):
            with TabPane("Lessons", id="lessons"):
                with Horizontal():
                    yield ListView(
                        *[ListItem(Label(le.title), id=f"les-{i}")
                          for i, le in enumerate(LESSONS)],
                        id="lesson-list")
                    with VerticalScroll():
                        yield Static(id="lesson-body")
            with TabPane("Price", id="price"):
                with Horizontal(classes="controls"):
                    yield Label("S"); yield Input("100", id="p-s")
                    yield Label("K"); yield Input("100", id="p-k")
                    yield Label("T"); yield Input("1.0", id="p-t")
                    yield Label("vol"); yield Input("0.3", id="p-v")
                    yield Button("price", id="p-go", variant="primary")
                with VerticalScroll():
                    yield Static(id="price-out", classes="out")
            with TabPane("Surface", id="surface"):
                with Horizontal(classes="controls"):
                    yield Label("T"); yield Input("1.0", id="s-t")
                    yield Label("noise"); yield Input("1.5", id="s-n")
                    yield Button("fit", id="s-go", variant="primary")
                    yield Button("unconstrained", id="s-free")
                with VerticalScroll():
                    yield Static(id="surface-out", classes="out")
            with TabPane("Hedge", id="hedge"):
                with Horizontal(classes="controls"):
                    yield Label("n_mon"); yield Input("512", id="h-n")
                    yield Label("every"); yield Input("1", id="h-e")
                    yield Label("bps"); yield Input("0", id="h-c")
                    yield Label("paths"); yield Input("4000", id="h-p")
                    yield Button("run", id="h-go", variant="primary")
                with VerticalScroll():
                    yield Static(id="hedge-out", classes="out")
            with TabPane("Making", id="making"):
                with Horizontal(classes="controls"):
                    yield Label("gamma"); yield Input("0.1", id="m-g")
                    yield Label("paths"); yield Input("3000", id="m-p")
                    yield Button("run", id="m-go", variant="primary")
                with VerticalScroll():
                    yield Static(id="mm-out", classes="out")
            with TabPane("Engines", id="engines"):
                with Horizontal(classes="controls"):
                    yield Label("paths"); yield Input("8000", id="b-p")
                    yield Button("benchmark", id="b-go", variant="primary")
                with VerticalScroll():
                    yield Static(id="bench-out", classes="out")
            with TabPane("Runs", id="runs"):
                with Horizontal():
                    yield DataTable(cursor_type="row", id="runs-table")
                    with VerticalScroll():
                        yield Static(id="run-detail")
        yield Footer()

    def on_mount(self):
        self.title = "vol-lab"
        self.sub_title = f"{len(self.rows)} runs recorded"
        table = self.query_one("#runs-table", DataTable)
        table.add_columns("when", "run", "config", "status")
        for r in self.rows:
            table.add_row(r["ts"], r["run_id"][:8], r["config_hash"][:8], r["status"])
        self._show_lesson(0)
        if self.rows:
            self._show_run(0)

    # --------------------------------------------------------------- lessons

    def _show_lesson(self, index):
        if 0 <= index < len(LESSONS):
            body = lesson_body(LESSONS[index], self.findings)
            self.query_one("#lesson-body", Static).update(body)

    @on(ListView.Highlighted, "#lesson-list")
    def _lesson_highlighted(self, event):
        self._show_lesson(event.list_view.index or 0)

    # ------------------------------------------------------------------ runs

    def _show_run(self, index):
        if 0 <= index < len(self.rows):
            self.query_one("#run-detail", Static).update(format_run(self.rows[index]))

    @on(DataTable.RowHighlighted, "#runs-table")
    def _row_highlighted(self, event):
        self._show_run(event.cursor_row)

    # -------------------------------------------------------------- handlers

    def _num(self, wid, cast=float, default=0.0):
        try:
            return cast(self.query_one(wid, Input).value)
        except (ValueError, TypeError):
            return default

    @on(Button.Pressed)
    def _pressed(self, event):
        {"p-go": self.run_price, "s-go": lambda: self.run_surface(True),
         "s-free": lambda: self.run_surface(False), "h-go": self.run_hedge,
         "m-go": self.run_mm, "b-go": self.run_bench}.get(
            event.button.id, lambda: None)()

    def action_refresh(self):
        {"price": self.run_price, "surface": lambda: self.run_surface(True),
         "hedge": self.run_hedge, "making": self.run_mm,
         "engines": self.run_bench}.get(
            self.query_one(TabbedContent).active, lambda: None)()

    def _busy(self, wid, what):
        self.query_one(wid, Static).update(f"{what}...")

    # ------------------------------------------------------------- the work

    @work(thread=True)
    def run_price(self):
        from vollab.pricing.black_scholes import (
            bs_delta, bs_gamma, bs_price, bs_theta, bs_vega,
        )
        from vollab.pricing.inverse import fiat_delta_mismatch, inverse_price
        from vollab.render.charts import curve
        import numpy as np

        self.call_from_thread(self._busy, "#price-out", "pricing")
        S = self._num("#p-s", float, 100.0); K = self._num("#p-k", float, 100.0)
        T = self._num("#p-t", float, 1.0); v = self._num("#p-v", float, 0.3)
        a = (S, K, T, 0.0, 0.0, v)
        grid = np.linspace(max(S * 0.5, 1.0), S * 1.6, 120)
        coin, naive, gap = fiat_delta_mismatch("call", S, K, T, 0.0, 0.0, v)
        body = "\n".join([
            f"call  {bs_price('call', *a):12.6f}      put  {bs_price('put', *a):12.6f}",
            f"delta {bs_delta('call', *a):+12.6f}      gamma {bs_gamma('call', *a):+12.6f}",
            f"vega  {bs_vega('call', *a):+12.6f}      theta {bs_theta('call', *a):+12.6f}",
            "",
            f"coin-settled (inverse) call   {float(inverse_price('call', *a)):.8f} coin",
            f"  coin delta {float(coin):+.6f}   converted vanilla {float(naive):+.6f}"
            f"   off by {float(gap / naive) * 100:.1f}%",
            "",
            curve(grid, bs_delta("call", grid, K, T, 0.0, 0.0, v),
                  "call delta against spot", "spot", "delta",
                  width=CHART_W, height=CHART_H),
        ])
        self.call_from_thread(
            lambda: self.query_one("#price-out", Static).update(ansi(body)))

    @work(thread=True)
    def run_surface(self, constrained=True):
        import numpy as np
        from vollab.pricing.black_scholes import bs_implied_vol
        from vollab.pricing.heston_cf import heston_price
        from vollab.render.charts import density, smile
        from vollab.surface.calibrate import calibrate_svi
        from vollab.surface.svi import implied_vol as svi_iv
        from vollab.surface.svi import risk_neutral_density

        self.call_from_thread(self._busy, "#surface-out",
                              "fitting" + ("" if constrained else " (unconstrained)"))
        T = self._num("#s-t", float, 1.0); noise = self._num("#s-n", float, 0.0)
        S, par = 100.0, dict(v0=0.06, kap_h=2.0, th_h=0.05, xi=0.5, rho=-0.6)
        ks, ivs = [], []
        for k in np.linspace(-0.4, 0.4, 13):
            strike = S * np.exp(k)
            try:
                ivs.append(bs_implied_vol(
                    "call", heston_price("call", S, strike, T, 0.0, 0.0, **par),
                    S, strike, T, 0.0, 0.0))
                ks.append(k)
            except ValueError:
                continue
        ks, iv = np.array(ks), np.array(ivs)
        if noise:
            iv = iv + np.random.default_rng(0).normal(0, noise / 100.0, iv.size)
        try:
            p, _, diag = calibrate_svi(ks, iv, T, arb_free=constrained)
        except RuntimeError as exc:
            self.call_from_thread(
                lambda: self.query_one("#surface-out", Static).update(f"refused: {exc}"))
            return
        fine = np.linspace(ks.min() * 1.4, ks.max() * 1.4, 200)
        bad = diag["min_durrleman_g"] < -1e-8
        body = "\n".join([
            smile(ks, iv, fine, svi_iv(p, fine, T),
                  f"Heston smile, T={T}" + (f", {noise}bp noise" if noise else ""),
                  width=CHART_W, height=CHART_H),
            "",
            density(fine, risk_neutral_density(p, fine, T), "implied density",
                    width=CHART_W, height=14),
            "",
            f"SVI  a={p.a:+.5f} b={p.b:.5f} rho={p.rho:+.4f} m={p.m:+.5f}"
            f" sigma={p.sigma:.5f}",
            f"fit  rmse {diag['rmse_vol_points']:.4f} vol pts"
            f"   max err {diag['max_abs_err_vol_points']:.4f}",
            f"arb  min Durrleman g {diag['min_durrleman_g']:+.3e}"
            f"   wings {diag['wing_left']:.3f} / {diag['wing_right']:.3f}"
            f" (Lee bound 2)",
            "",
            ("*** this slice implies a NEGATIVE density: it prices butterflies "
             "negative ***" if bad else "density is non-negative: these are prices"),
        ])
        self.call_from_thread(
            lambda: self.query_one("#surface-out", Static).update(ansi(body)))

    @work(thread=True)
    def run_hedge(self):
        import numpy as np
        from vollab.hedge.config import Contract, HedgeConfig, VolSpec
        from vollab.hedge.schedule import FixedTime
        from vollab.hedge.simulator import simulate
        from vollab.metrics.bootstrap import bootstrap_sd
        from vollab.render.charts import histogram

        self.call_from_thread(self._busy, "#hedge-out", "simulating")
        n_mon = self._num("#h-n", int, 512); every = self._num("#h-e", int, 1)
        bps = self._num("#h-c", float, 0.0); paths = self._num("#h-p", int, 4000)
        cfg = HedgeConfig(
            contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
            vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(max(every, 1)),
            n_mon=max(n_mon, 2), cost_bps=bps, n_paths=max(paths, 100),
            seed=11, chunk_paths=min(max(paths, 100), 4000))
        r = simulate(cfg)
        sd, se = bootstrap_sd(r.pnl, n_boot=300)
        a = r.attribution
        body = "\n".join([
            histogram(r.pnl, "terminal P&L", width=CHART_W, height=CHART_H),
            "",
            f"mean      {r.pnl.mean():+.6f} +/- "
            f"{r.pnl.std(ddof=1) / np.sqrt(r.pnl.size):.6f}",
            f"sd        {sd:.6f} +/- {se:.6f}",
            f"rehedges  {r.n_rehedges.mean():.1f}      turnover {r.turnover.mean():.1f}",
            "",
            "P&L explain, per path",
            f"  delta {a.delta.mean():+9.4f}   gamma {a.gamma.mean():+9.4f}"
            f"   theta {a.theta.mean():+9.4f}",
            f"  carry {a.carry.mean():+9.4f}   cost  {a.cost.mean():+9.4f}"
            f"   resid {a.residual_sum.mean():+9.4f}",
        ])
        self.call_from_thread(
            lambda: self.query_one("#hedge-out", Static).update(ansi(body)))

    @work(thread=True)
    def run_mm(self):
        from vollab.metrics.bootstrap import paired_bootstrap
        from vollab.mm.quoting import DealerParams, MarketParams
        from vollab.mm.simulate import STRATEGIES, simulate_mm
        from vollab.render.charts import histogram

        self.call_from_thread(self._busy, "#mm-out", "quoting")
        gam = self._num("#m-g", float, 0.1); paths = self._num("#m-p", int, 3000)
        market, dealer = MarketParams(), DealerParams(gam=gam)
        runs = {s_: simulate_mm(market, dealer, s_, seed=5, n_paths=max(paths, 200))
                for s_ in STRATEGIES}
        lines = [f"{'strategy':22s} {'pnl':>9s} {'sd':>8s} {'ratio':>7s}"
                 f" {'|q| max':>8s} {'fills':>7s}"]
        for name, r in runs.items():
            lines.append(f"{name:22s} {r.pnl.mean():9.3f} {r.pnl.std(ddof=1):8.3f}"
                         f" {r.pnl.mean() / r.pnl.std(ddof=1):7.3f}"
                         f" {r.inventory_max_abs.mean():8.2f} {r.n_fills.mean():7.1f}")
        lines.append("")
        for left in ("avellaneda_stoikov", "glft"):
            if left in runs and "symmetric" in runs:
                d, lo, hi = paired_bootstrap(runs[left].pnl, runs["symmetric"].pnl,
                                             n_boot=800)
                lines.append(f"{left} minus control: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append("")
        for name in ("avellaneda_stoikov", "symmetric"):
            if name in runs:
                lines.append(histogram(runs[name].inventory_end.astype(float),
                                       f"end inventory, {name}", bins=31,
                                       width=CHART_W, height=14))
                lines.append("")
        body = "\n".join(lines)
        self.call_from_thread(
            lambda: self.query_one("#mm-out", Static).update(ansi(body)))

    @work(thread=True)
    def run_bench(self):
        import time
        import numpy as np
        from vollab.hedge.config import Contract, HedgeConfig, VolSpec
        from vollab.hedge.schedule import FixedTime
        from vollab.hedge.simulator import cpp_available, simulate

        self.call_from_thread(self._busy, "#bench-out", "benchmarking")
        if not cpp_available():
            self.call_from_thread(
                lambda: self.query_one("#bench-out", Static).update(
                    "C++ extension is not built.\n"
                    "  uv sync --reinstall-package vollab"))
            return
        paths = max(self._num("#b-p", int, 8000), 500)
        lines = [f"{'config':22s} {'reference':>10s} {'cpp':>9s} {'speedup':>8s}"
                 f" {'rel dPnL':>10s}"]
        for n_mon, every in ((512, 1), (512, 8), (2048, 4)):
            base = dict(contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
                        vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(every),
                        n_mon=n_mon, cost_bps=5.0, n_paths=paths, seed=1234,
                        chunk_paths=paths)
            ref_full = simulate(HedgeConfig(**base))
            t = time.time(); simulate(HedgeConfig(attribute=False, **base))
            t_ref = time.time() - t
            cpp = simulate(HedgeConfig(**base), engine="cpp")
            t = time.time(); simulate(HedgeConfig(**base), engine="cpp")
            t_cpp = time.time() - t
            scale = max(float(np.abs(ref_full.pnl).max()), 1.0)
            d = float(np.abs(cpp.pnl - ref_full.pnl).max()) / scale
            lines.append(f"n_mon={n_mon:<5d} every={every:<2d}   {t_ref:9.3f}s"
                         f" {t_cpp:8.3f}s {t_ref / t_cpp:7.1f}x {d:10.1e}")
        lines += ["", "Equal work: the reference runs with attribution off, since the",
                  "C++ engine computes P&L and rehedge counts only."]
        self.call_from_thread(
            lambda: self.query_one("#bench-out", Static).update("\n".join(lines)))


# Kept for the existing entry point and tests.
ViewerApp = VolLabApp
