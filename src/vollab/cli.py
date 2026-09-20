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

from vollab.hedge.registry import build_config, paths_fingerprint_input
from vollab.hedge.simulator import simulate
from vollab.metrics.bootstrap import bootstrap_sd, paired_bootstrap
from vollab.pricing.black_scholes import (
    bs_delta, bs_gamma, bs_price, bs_theta, bs_vega,
)
from vollab.protocol.hashing import config_hash
from vollab.protocol.ledger import SCHEMA_VERSION, Ledger
from vollab.protocol.prereg import HashMismatch, load_registered
from vollab.render.charts import histogram
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

    lg = sub.add_parser("ledger", help="5. list recorded runs")
    lg.set_defaults(fn=_cmd_ledger)

    a = p.parse_args(argv)
    return a.fn(a)
