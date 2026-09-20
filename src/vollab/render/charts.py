"""Terminal charts.

Every function takes plain arrays, never a result object, so this module stays
a leaf that Phase B can reuse. plotext is a single-maintainer dependency, so
each chart degrades to a built-in ANSI renderer rather than failing.
"""

import numpy as np

_BLOCKS = " .:-=+*#%@"


def _plt():
    try:
        import plotext
        return plotext
    except Exception:
        return None


def loglog(x, y, title, ref_slope=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = _plt()
    if p is None or x.size < 2:
        return sparkline(y, title)
    p.clf()
    p.title(title)
    lx, ly = np.log(x), np.log(y)
    p.plot(lx.tolist(), ly.tolist(), marker="braille", label="measured")
    if ref_slope is not None:
        ref = ly[0] + ref_slope * (lx - lx[0])
        p.plot(lx.tolist(), ref.tolist(), marker="dot", label=f"slope {ref_slope}")
    p.xlabel("log rehedges")
    p.ylabel("log sd(PnL)")
    return p.build()


def histogram(x, title, bins=60):
    x = np.asarray(x, float)
    p = _plt()
    if p is None:
        return sparkline(x, title)
    p.clf()
    p.title(title)
    p.hist(x.tolist(), bins=bins)
    return p.build()


def curve(x, y, title, xlabel="x", ylabel="y"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = _plt()
    if p is None or x.size < 2:
        return sparkline(y, title)
    p.clf()
    p.title(title)
    p.plot(x.tolist(), y.tolist(), marker="braille")
    p.xlabel(xlabel)
    p.ylabel(ylabel)
    return p.build()


def sparkline(y, title):
    """The named fallback for the plotext risk."""
    y = np.asarray(y, float)
    lo, hi = float(y.min()), float(y.max())
    span = (hi - lo) or 1.0
    line = "".join(_BLOCKS[int((v - lo) / span * (len(_BLOCKS) - 1))] for v in y)
    return f"{title}\n{line}\n[{lo:.4g} .. {hi:.4g}]"


def smile(k_mkt, iv_mkt, k_fit, iv_fit, title):
    """Market points against a fitted slice, in volatility points."""
    p = _plt()
    if p is None:
        return sparkline(np.asarray(iv_fit), title)
    p.clf()
    p.title(title)
    p.scatter(np.asarray(k_mkt, float).tolist(),
              (np.asarray(iv_mkt, float) * 100).tolist(), label="market")
    p.plot(np.asarray(k_fit, float).tolist(),
           (np.asarray(iv_fit, float) * 100).tolist(), marker="braille", label="SVI")
    p.xlabel("log-moneyness")
    p.ylabel("implied vol %")
    return p.build()


def density(k, dens, title):
    """Risk-neutral density implied by a slice. Dipping below zero is arbitrage."""
    p = _plt()
    if p is None:
        return sparkline(np.asarray(dens), title)
    p.clf()
    p.title(title)
    p.plot(np.asarray(k, float).tolist(), np.asarray(dens, float).tolist(),
           marker="braille", label="density")
    p.plot(np.asarray(k, float).tolist(), [0.0] * len(k), marker="dot", label="zero")
    p.xlabel("log-moneyness")
    return p.build()
