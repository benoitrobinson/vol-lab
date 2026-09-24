"""The two sibling repositories, drawn inside the panel.

lob-lab (Rust) and contract-lab (OCaml) are separate programs with their own
toolchains. The panel runs their built binaries and reads the JSON they print,
so neither is a dependency of vol-lab: a missing one leaves a tab that says how
to build it, and nothing else in the lab notices.

Loading and formatting are split, so the formatting is testable against a
fixture without either toolchain installed.
"""

import csv
import json
import os
import subprocess
from pathlib import Path

from vollab.render.charts import bars

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TIMEOUT_S = 600

# The preregistered minimum in lob-lab's docs/preregistration.md. Below it the
# tab says it is showing the pipeline, not a result.
MIN_DAYS = 7


class Unavailable(Exception):
    """The sibling cannot be shown; the message says what to do about it."""


def lob_home():
    return Path(os.environ.get("LOBLAB_HOME", ROOT.parent / "lob-lab"))


def contract_home():
    return Path(os.environ.get("CONTRACTLAB_HOME", ROOT.parent / "contract-lab"))


def _run(cmd, cwd):
    try:
        out = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                             text=True, timeout=TIMEOUT_S, check=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or exc.stdout or "").strip().splitlines()[-6:])
        raise Unavailable(f"{Path(cmd[0]).name} failed:\n{tail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise Unavailable(f"{Path(cmd[0]).name} ran past {TIMEOUT_S}s") from exc
    return out.stdout


def _run_json(cmd, cwd):
    text = _run(cmd, cwd)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise Unavailable(f"{Path(cmd[0]).name} printed something other than "
                          "JSON; is the binary older than its --json flag?") from exc


# ------------------------------------------------------------------ lob-lab

def load_book(home=None):
    home = Path(home or lob_home())
    release = home / "target" / "release"
    study, ofi = release / "study", release / "measure_ofi"
    if not home.is_dir():
        raise Unavailable(f"lob-lab not found at {home}\n"
                          "  clone it beside vol-lab, or set LOBLAB_HOME")
    if not (study.exists() and ofi.exists()):
        raise Unavailable(f"lob-lab is not built.\n  cd {home} && cargo build --release")
    days = home / "data" / "days"
    if not days.is_dir() or not any(days.iterdir()):
        raise Unavailable("lob-lab has no recorded data yet; market data is never "
                          "committed.\n"
                          f"  cd {home} && make record     # leave it running, "
                          "then stop with ctrl-c")
    _run([study, "--data", "data/days", "--out", "artifacts"], home)
    with (home / "artifacts" / "runs.csv").open() as f:
        runs = list(csv.DictReader(f))
    return {
        "home": os.path.relpath(home, ROOT),
        "headline": json.loads((home / "artifacts" / "headline.json").read_text()),
        "runs": runs,
        "ofi": _run_json([ofi, "--dir", "data", "--json"], home),
    }


def _horizon(ms):
    return f"{ms / 1000:g}s"


def book_body(data, width, height):
    h, ofi = data["headline"], data["ofi"]
    days = h["days"]
    sweep = sorted(h.get("cancel_position_sensitivity", {}).values(),
                   key=lambda v: v["cancels_from_ahead_pct"])
    out = [
        f"lob-lab   {data['home']}",
        f"{days} recorded day{'s' * (days != 1)}, {ofi['book_updates']:,} book "
        f"updates of Deribit BTC-PERPETUAL",
    ]
    if days < MIN_DAYS:
        out.append(f"the preregistered study needs {MIN_DAYS} days: this is the "
                   "pipeline working, not a result")
    out.append("")

    if sweep:
        ratios = [v["fill_inflation"] for v in sweep]
        out += [
            bars([f"{v['cancels_from_ahead_pct']}%" for v in sweep], ratios,
                 "fills handed out by fill-at-touch, per queue fill",
                 xlabel="cancels assumed to come from ahead of us",
                 width=width, height=height),
            f"{min(ratios):.2f}x to {max(ratios):.2f}x across the whole range: the "
            "unfalsifiable assumption moves the second decimal, not the conclusion",
            "",
        ]

    rows = [r for r in ofi["horizons"] if r["sign_pct"] is not None]
    if rows:
        out += [
            bars([_horizon(r["horizon_ms"]) for r in rows],
                 [r["sign_pct"] - 50.0 for r in rows],
                 "order flow imbalance: points above a coin flip",
                 ref=0.0, xlabel="horizon", width=width,
                 height=max(height - 4, 9)),
            "   ".join(f"{_horizon(r['horizon_ms'])} {r['sign_pct']:.1f}%"
                              f" of {r['moved']}" for r in rows),
            "",
        ]

    if days >= 2:
        lo, hi = h["fill_inflation_ci95"]
        out.append(f"fill inflation {h['fill_inflation_mean']:.2f}x "
                   f"[{lo:.2f}, {hi:.2f}] over {days} days, paired bootstrap")
    else:
        out.append(f"fill inflation {h['fill_inflation_mean']:.2f}x on one day: "
                   "no interval until there are days to resample")
    out += ["", f"{'quoter':10s} {'fill model':13s} {'fills':>6s} {'pnl sat':>9s}"
                f" {'markout 1s bps':>15s}"]
    for r in data["runs"]:
        if r["model"] in ("naive", "pessimistic", "proportional"):
            out.append(f"{r['quoter']:10s} {r['model']:13s} {int(r['fills']):6d}"
                       f" {float(r['pnl_btc']) * 1e8:9.0f}"
                       f" {float(r['markout_1s_bps']):15.3f}")
    return "\n".join(out)


# ------------------------------------------------------------- contract-lab

def load_contracts(home=None):
    home = Path(home or contract_home())
    exe = home / "_build" / "default" / "bin" / "main.exe"
    if not home.is_dir():
        raise Unavailable(f"contract-lab not found at {home}\n"
                          "  clone it beside vol-lab, or set CONTRACTLAB_HOME")
    if not exe.exists():
        raise Unavailable(f"contract-lab is not built.\n  cd {home} && "
                          "opam switch create . 5.4.1 && dune build")
    slice_file = home / "data" / "svi_slice.txt"
    # The header's second paragraph says when and from what the slice was fitted.
    paragraphs = " ".join(line[1:].strip() if line[1:].strip() else "\n"
                          for line in slice_file.read_text().splitlines()
                          if line.startswith("#")).split("\n")
    provenance = [paragraphs[1].strip()] if len(paragraphs) > 1 else []
    return {
        "home": os.path.relpath(home, ROOT),
        "provenance": provenance,
        "study": _run_json([exe, "study", "--json"], home),
        "checks": _run_json([exe, "checks", "--json"], home),
        "termsheet": _run_json([exe, "termsheet", "--json"], home),
    }


def contracts_body(data, width, height):
    s = data["study"]
    rows = s["rows"]
    out = [
        f"contract-lab   {data['home']}",
        f"BTC smile, {s['expiry_years'] * 365:.0f} days, forward {s['forward']:,.0f},"
        f" at-the-money {s['atm_vol']:.1%}",
        *data["provenance"],
        "",
        bars([f"{r['moneyness']:.2f}" for r in rows],
             [r["difference_bps"] for r in rows],
             "a digital as a call spread on the smile, minus N(d2), in bps",
             ref=0.0, xlabel="strike / forward", width=width, height=height),
        "the sign follows the slope of the smile: a desk quoting N(d2) is off"
        " the same way every time",
        "",
        f"{'K/F':>5s} {'strike':>10s} {'N(d2)':>9s} {'call spread':>12s} {'bps':>8s}",
    ]
    out += [f"{r['moneyness']:5.2f} {r['strike']:10,.0f} {r['n_d2']:9.5f}"
            f" {r['call_spread']:12.5f} {r['difference_bps']:+8.1f}" for r in rows]
    out += ["", "CHECKS  every identity the tests hold, as computed just now"]
    for c in data["checks"]:
        v = c["value"]
        val = f"{v:.0f}" if float(v).is_integer() and v > 100 else f"{v:.4f}"
        if c["se"] is not None:
            val += f" +/- {c['se']:.4f}"
        out.append(f"  {c['label']:36s} {val:>9s}")
    out += ["", "TERM SHEETS  the same algebra, printed"]
    for t in data["termsheet"]:
        out += [f"  {t['name']}", f"    {t['contract']}"]
    return "\n".join(out)
