"""Fit one SVI slice to a live Deribit BTC smile and write it out.

    uv run python scripts/export_slice.py --out ../contract-lab/data/svi_slice.txt

This is an export tool, not part of the lab: it is the only thing here that
touches the network, it is never imported by the library or the tests, and the
findings do not depend on it. What it writes is five fitted parameters plus the
provenance needed to refit them, not market data.

The smile is built from out-of-the-money quotes only, puts below the forward and
calls above it, which is the convention that avoids double counting a strike and
keeps the illiquid in-the-money wing out of the fit. Deribit publishes a
`mark_iv` per instrument and the forward of that expiry as `underlying_price`.
"""

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

import numpy as np

from vollab.surface.calibrate import calibrate_svi
from vollab.surface.svi import durrleman_g, total_variance

SUMMARY = ("https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
           "?currency={currency}&kind=option")
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def fetch(currency):
    with urllib.request.urlopen(SUMMARY.format(currency=currency), timeout=30) as r:
        return json.loads(r.read())["result"]


def parse_name(name):
    """BTC-24SEP26-90000-C -> (expiry datetime, strike, right)."""
    _, expiry, strike, right = name.split("-")
    day = int(expiry[:-5])
    month = MONTHS[expiry[-5:-2]]
    year = 2000 + int(expiry[-2:])
    # Deribit options expire at 08:00 UTC.
    return (dt.datetime(year, month, day, 8, tzinfo=dt.timezone.utc),
            float(strike), right)


def slices(rows, now):
    out = {}
    for row in rows:
        iv = row.get("mark_iv")
        forward = row.get("underlying_price")
        if not iv or not forward:
            continue
        expiry, strike, right = parse_name(row["instrument_name"])
        years = (expiry - now).total_seconds() / (365.0 * 24 * 3600)
        if years <= 0:
            continue
        out.setdefault(expiry, {"T": years, "F": forward, "quotes": []})
        out[expiry]["quotes"].append((strike, right, iv / 100.0))
    return out


def otm_smile(sl):
    """One implied vol per strike: puts below the forward, calls above it."""
    forward = sl["F"]
    keep = {}
    for strike, right, iv in sl["quotes"]:
        wanted = "P" if strike < forward else "C"
        if right == wanted:
            keep[strike] = iv
    strikes = np.array(sorted(keep))
    return strikes, np.array([keep[s] for s in strikes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-days", type=float, default=30.0,
                    help="pick the listed expiry closest to this many days out")
    ap.add_argument("--min-strikes", type=int, default=10)
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    all_slices = slices(fetch(args.currency), now)
    usable = {e: s for e, s in all_slices.items()
              if len(otm_smile(s)[0]) >= args.min_strikes}
    if not usable:
        raise SystemExit("no expiry had enough strikes")
    expiry = min(usable, key=lambda e: abs(usable[e]["T"] * 365.0 - args.target_days))
    sl = usable[expiry]
    strikes, iv = otm_smile(sl)
    k = np.log(strikes / sl["F"])

    params, rmse, _diag = calibrate_svi(k, iv, sl["T"])
    dense = np.linspace(k.min() - 0.5, k.max() + 0.5, 401)
    g = durrleman_g(params, dense)
    fitted = np.sqrt(np.maximum(total_variance(params, k), 1e-12) / sl["T"])

    print(f"expiry {expiry:%Y-%m-%d}  T {sl['T']:.4f}y  forward {sl['F']:.2f}  "
          f"{len(strikes)} strikes")
    print(f"a {params.a:.6f}  b {params.b:.6f}  rho {params.rho:+.6f}  "
          f"m {params.m:+.6f}  sigma {params.sigma:.6f}")
    print(f"rmse {rmse:.5f} vol points   min Durrleman g {g.min():+.6f}   "
          f"atm iv {np.interp(0.0, k, fitted):.4f}")
    if g.min() < 0:
        print("WARNING: the fitted slice implies a negative density; refusing to write")
        raise SystemExit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"# SVI slice fitted to a live Deribit {args.currency} smile.\n"
        f"#\n"
        f"# Fitted {now:%Y-%m-%d %H:%M} UTC by vol-lab's scripts/export_slice.py,\n"
        f"# from public mark IVs on {len(strikes)} out-of-the-money quotes,\n"
        f"# expiry {expiry:%Y-%m-%d} 08:00 UTC.\n"
        f"#\n"
        f"# rmse {rmse:.5f} vol points, minimum Durrleman g {g.min():+.6f}, so the\n"
        f"# slice is free of butterfly arbitrage on the fitted range.\n"
        f"#\n"
        f"# Refit with:\n"
        f"#   uv run python scripts/export_slice.py --out {args.out}\n"
        f"a {params.a:.6f}\n"
        f"b {params.b:.6f}\n"
        f"rho {params.rho:.6f}\n"
        f"m {params.m:.6f}\n"
        f"s {params.sigma:.6f}\n"
        f"t {sl['T']:.6f}\n"
        f"forward {sl['F']:.4f}\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
