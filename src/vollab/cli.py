"""Command line interface. Commands are numbered in run order in --help."""

import argparse
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import scipy

from vollab.hedge.config import HedgeConfig
from vollab.hedge.registry import build_config, paths_fingerprint_input
from vollab.hedge.simulator import simulate
from vollab.metrics.bootstrap import bootstrap_sd, paired_bootstrap
from vollab.pricing.black_scholes import (
    bs_delta, bs_gamma, bs_price, bs_theta, bs_vega,
)
from vollab.protocol.hashing import config_hash
from vollab.protocol.ledger import SCHEMA_VERSION, Ledger
from vollab.protocol.prereg import HashMismatch, load_registered
from vollab.render.charts import density, histogram, smile
from vollab.hedge.config import Contract, VolSpec
from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import cpp_available
from vollab.rng.scheme import RNG_SCHEME_VERSION

VERSION = "0.1.0"
LEDGER_PATH = Path(".vollab") / "ledger.db"


def _git(*args, default=""):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return default


def _cmd_price(a):
    args = (a.S, a.K, a.T, a.r, a.q, a.vol)
    print(f"price {bs_price(a.kind, *args):.6f}")
    print(f"delta {bs_delta(a.kind, *args):+.6f}   gamma {bs_gamma(a.kind, *args):+.6f}")
    print(f"vega  {bs_vega(a.kind, *args):+.6f}   theta {bs_theta(a.kind, *args):+.6f}")
    return 0


def _cmd_register(a):
    """Stamp a config file with its hash so it can be run."""
    import tomllib

    import tomli_w

    path = Path(a.config)
    d = tomllib.loads(path.read_text())
    if not d.get("hypothesis"):
        print("refused: config must declare a hypothesis first", file=sys.stderr)
        return 2
    d.pop("config_hash", None)
    body = dict(d)
    body["config_hash"] = config_hash(d)
    path.write_bytes(tomli_w.dumps(body).encode())
    print(f"registered {path} as {body['config_hash'][:12]}")
    return 0


def _cmd_run(a):
    try:
        d = load_registered(a.config)
    except (HashMismatch, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    cfg = build_config(d)
    run_id = uuid.uuid4().hex
    t0 = time.time()
    status, n_done, metrics = "failed", 0, {}
    try:
        r = simulate(cfg)
        status, n_done = "ok", int(r.pnl.size)
        sd, se = bootstrap_sd(r.pnl, n_boot=400)
        mean_se = r.pnl.std(ddof=1) / np.sqrt(r.pnl.size)
        metrics = {"pnl_mean": float(r.pnl.mean()), "pnl_mean_se": float(mean_se),
                   "pnl_sd": sd, "pnl_sd_se": se,
                   "n_rehedges_mean": float(r.n_rehedges.mean()),
                   "turnover_mean": float(r.turnover.mean())}
        print(histogram(r.pnl, "terminal P&L"))
        print(f"mean      {metrics['pnl_mean']:+.6f} +/- {mean_se:.6f}")
        print(f"sd        {sd:.6f} +/- {se:.6f}")
        print(f"rehedges  {metrics['n_rehedges_mean']:.1f}")
        print(f"hypothesis: {d['hypothesis']}")
    finally:
        diff = _git("diff")
        Ledger(LEDGER_PATH).insert(dict(
            run_id=run_id, schema_version=SCHEMA_VERSION,
            ts=time.strftime("%Y-%m-%dT%H:%M:%S"), config_hash=config_hash(d),
            config_toml=Path(a.config).read_text(), hypothesis=d["hypothesis"],
            paths_fingerprint=config_hash(paths_fingerprint_input(d)),
            git_commit=_git("rev-parse", "HEAD"),
            git_dirty=int(bool(_git("status", "--porcelain"))),
            git_diff_sha=config_hash({"diff": diff}) if diff else None,
            vollab_version=VERSION, rng_scheme_version=RNG_SCHEME_VERSION,
            engine="numpy", engine_build_id=None,
            numpy_version=np.__version__, scipy_version=scipy.__version__,
            python_version=platform.python_version(),
            platform=f"{platform.system()}-{platform.machine()}",
            status=status, n_paths_completed=n_done,
            metrics=json.dumps(metrics), artifacts="[]", artifact_sha256="[]",
            runtime_s=time.time() - t0))
        print(f"run {run_id[:8]} recorded ({status})")
    return 0


def _cmd_compare(a):
    led = Ledger(LEDGER_PATH)
    ra, rb = led.get(a.run_a), led.get(a.run_b)
    if ra is None or rb is None:
        print("refused: unknown run id", file=sys.stderr)
        return 2
    if ra["paths_fingerprint"] != rb["paths_fingerprint"]:
        print("refused: runs did not share paths, a paired comparison would be "
              "silently wrong", file=sys.stderr)
        return 2
    ma, mb = json.loads(ra["metrics"]), json.loads(rb["metrics"])
    print(f"mean  {ma['pnl_mean']:+.6f}  vs  {mb['pnl_mean']:+.6f}")
    print(f"sd    {ma['pnl_sd']:.6f}  vs  {mb['pnl_sd']:.6f}")
    return 0


def _cmd_surface(a):
    """Fit an SVI slice to a synthetic Heston smile and report its diagnostics."""
    import numpy as np

    from vollab.pricing.black_scholes import bs_implied_vol
    from vollab.pricing.heston_cf import heston_price
    from vollab.surface.calibrate import calibrate_svi
    from vollab.surface.svi import implied_vol as svi_iv
    from vollab.surface.svi import risk_neutral_density

    S, r, q = 100.0, 0.0, 0.0
    par = dict(v0=a.v0, kap_h=a.kappa, th_h=a.theta, xi=a.xi, rho=a.rho)
    # Deep strikes at short maturity can price outside the invertible range,
    # where vega is numerically zero. A real chain has such strikes; drop them
    # rather than failing, and say how many went.
    ks_all = np.linspace(-a.width, a.width, a.points)
    ks_list, iv_list = [], []
    for k in ks_all:
        strike = S * np.exp(k)
        try:
            iv_list.append(bs_implied_vol(
                "call", heston_price("call", S, strike, a.T, r, q, **par),
                S, strike, a.T, r, q))
            ks_list.append(k)
        except ValueError:
            continue
    dropped = len(ks_all) - len(ks_list)
    if len(ks_list) < 5:
        print(f"refused: only {len(ks_list)} invertible strikes; widen T or narrow "
              f"--width", file=sys.stderr)
        return 2
    ks = np.array(ks_list)
    iv = np.array(iv_list)
    if dropped:
        print(f"note: dropped {dropped} strike(s) where implied vol is not "
              f"identifiable")
    if a.noise > 0:
        iv = iv + np.random.default_rng(a.seed).normal(0, a.noise / 100.0, iv.size)

    p, _, diag = calibrate_svi(ks, iv, a.T, arb_free=not a.unconstrained)
    fine = np.linspace(ks.min() * 1.4, ks.max() * 1.4, 240)

    print(smile(ks, iv, fine, svi_iv(p, fine, a.T),
                f"Heston smile, T={a.T}" + ("" if a.noise == 0 else f", {a.noise}bp noise")))
    print(density(fine, risk_neutral_density(p, fine, a.T), "implied density"))
    print(f"SVI  a={p.a:+.5f} b={p.b:.5f} rho={p.rho:+.4f} m={p.m:+.5f} sigma={p.sigma:.5f}")
    print(f"fit  rmse {diag['rmse_vol_points']:.4f} vol pts   "
          f"max err {diag['max_abs_err_vol_points']:.4f}")
    print(f"arb  min Durrleman g {diag['min_durrleman_g']:+.3e}   "
          f"wings {diag['wing_left']:.3f} / {diag['wing_right']:.3f} (Lee bound 2)")
    if diag["min_durrleman_g"] < -1e-8:
        print("WARNING: this slice implies a negative density", file=sys.stderr)
        return 1
    return 0


def _cmd_mm(a):
    """Quote two-sided against a simulated mid, with and without inventory skew."""
    import numpy as np

    from vollab.metrics.bootstrap import paired_bootstrap
    from vollab.mm.quoting import DealerParams, MarketParams
    from vollab.mm.simulate import simulate_mm

    market = MarketParams(sigma=a.sigma, A=a.A, kappa=a.kappa,
                          T=a.T, n_steps=a.steps)
    runs = {}
    for strat in ("avellaneda_stoikov", "symmetric"):
        runs[strat] = simulate_mm(market, DealerParams(gam=a.gam, max_inventory=a.cap),
                                  strat, seed=a.seed, n_paths=a.paths)

    hdr = f"{'strategy':22s} {'pnl':>9s} {'sd':>8s} {'ratio':>7s} {'|q| max':>8s} {'fills':>7s}"
    print(hdr)
    for name, r in runs.items():
        print(f"{name:22s} {r.pnl.mean():9.3f} {r.pnl.std(ddof=1):8.3f} "
              f"{r.pnl.mean() / r.pnl.std(ddof=1):7.3f} "
              f"{r.inventory_max_abs.mean():8.2f} {r.n_fills.mean():7.1f}")

    a_pnl = runs["avellaneda_stoikov"].pnl
    s_pnl = runs["symmetric"].pnl
    diff, lo, hi = paired_bootstrap(a_pnl, s_pnl, n_boot=2000)
    print()
    print(f"skew minus control, paired: {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(histogram(runs["avellaneda_stoikov"].inventory_end.astype(float),
                    "end inventory, with skew", bins=41))
    print(histogram(runs["symmetric"].inventory_end.astype(float),
                    "end inventory, control", bins=41))
    return 0


def _cmd_view(a):
    from vollab.tui.app import ViewerApp

    if not LEDGER_PATH.exists():
        print("no ledger here; run an experiment first", file=sys.stderr)
        return 2
    ViewerApp(LEDGER_PATH).run()
    return 0


def _cmd_bench(a):
    """Two speedups, both reported.

    The reference engine uses inverse-CDF normals so its stream can be matched
    bit for bit in C++, and that costs about 3x against numpy's native
    standard_normal. Quoting only the first number would compare C++ against a
    deliberately handicapped baseline.
    """
    import time

    if not cpp_available():
        print("C++ extension not built; run: uv sync --reinstall-package vollab",
              file=sys.stderr)
        return 2

    print(f"{'config':24s} {'reference':>10s} {'cpp':>10s} {'speedup':>9s} "
          f"{'rel dPnL':>10s} {'d rehedge':>10s}")
    for n_mon, every, n_paths in [(512, 1, a.paths), (512, 8, a.paths),
                                  (2048, 4, a.paths)]:
        base = dict(
            contract=Contract("call", 100.0, 100.0, 1.0, 0.0, 0.0),
            vols=VolSpec(0.3, 0.3, 0.3), schedule=FixedTime(every),
            n_mon=n_mon, cost_bps=5.0, n_paths=n_paths, seed=1234,
            chunk_paths=n_paths,
        )
        ref_full = simulate(HedgeConfig(**base))

        def timed(fn, repeats=a.repeats):
            fn()                                  # warm up, then take the best
            return min(_one(fn) for _ in range(repeats))

        def _one(fn):
            t = time.time()
            fn()
            return time.time() - t

        t_ref = timed(lambda: simulate(HedgeConfig(attribute=False, **base)))
        cpp = simulate(HedgeConfig(**base), engine="cpp")
        t_cpp = timed(lambda: simulate(HedgeConfig(**base), engine="cpp"))

        scale = max(float(np.abs(ref_full.pnl).max()), 1.0)
        d_pnl = float(np.abs(cpp.pnl - ref_full.pnl).max()) / scale
        d_n = float(np.abs(cpp.n_rehedges - ref_full.n_rehedges).max())
        print(f"n_mon={n_mon:<5d} every={every:<2d}      {t_ref:9.3f}s {t_cpp:9.3f}s "
              f"{t_ref / t_cpp:8.1f}x {d_pnl:10.1e} {d_n:10.0f}")

    print()
    print(f"Best of {a.repeats} timed repeats after a warmup. Timings compare equal")
    print("work: the reference runs with attribution off, since the C++ engine")
    print("computes P&L and rehedge counts only.")
    return 0


def _cmd_ledger(a):
    rows = Ledger(LEDGER_PATH).all()
    if not rows:
        print("no runs recorded")
        return 0
    print(f"{'when':20} {'run':9} {'config':9} {'status':7} {'dirty':5}  hypothesis")
    for row in rows:
        print(f"{row['ts']:20} {row['run_id'][:8]:9} {row['config_hash'][:8]:9} "
              f"{row['status']:7} {row['git_dirty']:<5}  {row['hypothesis'][:44]}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="vl", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("price", help="1. Black-Scholes price and greeks")
    pr.add_argument("--kind", default="call", choices=["call", "put"])
    pr.add_argument("--S", type=float, default=100.0)
    pr.add_argument("--K", type=float, required=True)
    pr.add_argument("--T", type=float, required=True)
    pr.add_argument("--r", type=float, default=0.0)
    pr.add_argument("--q", type=float, default=0.0)
    pr.add_argument("--vol", type=float, required=True)
    pr.set_defaults(fn=_cmd_price)

    rg = sub.add_parser("register", help="2. stamp a config with its hash")
    rg.add_argument("config")
    rg.set_defaults(fn=_cmd_register)

    rn = sub.add_parser("run", help="3. run a pre-registered experiment")
    rn.add_argument("config")
    rn.set_defaults(fn=_cmd_run)

    cp = sub.add_parser("compare", help="4. compare two runs that shared paths")
    cp.add_argument("run_a")
    cp.add_argument("run_b")
    cp.set_defaults(fn=_cmd_compare)

    bn = sub.add_parser("bench", help="5. NumPy reference against the C++ engine")
    bn.add_argument("--paths", type=int, default=20000)
    bn.add_argument("--repeats", type=int, default=3,
                    help="timing repeats; the best is reported, after a warmup")
    bn.set_defaults(fn=_cmd_bench)

    sf = sub.add_parser("surface", help="6. fit an SVI slice and check it for arbitrage")
    sf.add_argument("--T", type=float, default=1.0)
    sf.add_argument("--v0", type=float, default=0.06)
    sf.add_argument("--kappa", type=float, default=2.0)
    sf.add_argument("--theta", type=float, default=0.05)
    sf.add_argument("--xi", type=float, default=0.5)
    sf.add_argument("--rho", type=float, default=-0.6)
    sf.add_argument("--width", type=float, default=0.4)
    sf.add_argument("--points", type=int, default=15)
    sf.add_argument("--noise", type=float, default=0.0, help="vol points of noise")
    sf.add_argument("--seed", type=int, default=0)
    sf.add_argument("--unconstrained", action="store_true",
                    help="fit without the no-arbitrage constraints")
    sf.set_defaults(fn=_cmd_surface)

    mmp = sub.add_parser("mm", help="7. market making, with and without inventory skew")
    mmp.add_argument("--gam", type=float, default=0.1, help="inventory risk aversion")
    mmp.add_argument("--sigma", type=float, default=2.0)
    mmp.add_argument("--A", type=float, default=140.0)
    mmp.add_argument("--kappa", type=float, default=1.5)
    mmp.add_argument("--T", type=float, default=1.0)
    mmp.add_argument("--steps", type=int, default=200)
    mmp.add_argument("--paths", type=int, default=4000)
    mmp.add_argument("--cap", type=int, default=50)
    mmp.add_argument("--seed", type=int, default=5)
    mmp.set_defaults(fn=_cmd_mm)

    vw = sub.add_parser("view", help="8. browse recorded runs in a TUI")
    vw.set_defaults(fn=_cmd_view)

    lg = sub.add_parser("ledger", help="9. list recorded runs")
    lg.set_defaults(fn=_cmd_ledger)

    a = p.parse_args(argv)
    return a.fn(a)
