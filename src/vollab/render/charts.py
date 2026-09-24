"""Terminal charts.

Every function takes plain arrays, never a result object, so this module stays
a leaf that Phase B can reuse. plotext is a single-maintainer dependency, so
each chart degrades to a built-in ANSI renderer rather than failing.
"""

import numpy as np

_BLOCKS = " .:-=+*#%@"

# Terminal charts are read on a dark background beside other output, so they are
# themed to match and sized to stay readable rather than filling the window. A
# full-width white canvas is what the first version produced, and it swamped
# everything around it.
WIDTH, HEIGHT = 76, 18
THEME = "pro"


def _plt(width=None, height=None):
    try:
        import plotext
    except Exception:
        return None
    plotext.clf()
    plotext.theme(THEME)
    plotext.plotsize(width or WIDTH, height or HEIGHT)
    return plotext


def loglog(x, y, title, ref_slope=None, width=None, height=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = _plt(width, height)
    if p is None or x.size < 2:
        return sparkline(y, title)
    p.title(title)
    lx, ly = np.log(x), np.log(y)
    p.plot(lx.tolist(), ly.tolist(), marker="braille", label="measured")
    if ref_slope is not None:
        ref = ly[0] + ref_slope * (lx - lx[0])
        p.plot(lx.tolist(), ref.tolist(), marker="dot", label=f"slope {ref_slope}")
    p.xlabel("log rehedges")
    p.ylabel("log sd(PnL)")
    return p.build()


def histogram(x, title, bins=40, width=None, height=None):
    x = np.asarray(x, float)
    p = _plt(width, height)
    if p is None:
        return sparkline(x, title)
    p.title(title)
    p.hist(x.tolist(), bins=bins)
    return p.build()


def curve(x, y, title, xlabel="x", ylabel="y", width=None, height=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = _plt(width, height)
    if p is None or x.size < 2:
        return sparkline(y, title)
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


def smile(k_mkt, iv_mkt, k_fit, iv_fit, title, width=None, height=None):
    """Market points against a fitted slice, in volatility points."""
    p = _plt(width, height)
    if p is None:
        return sparkline(np.asarray(iv_fit), title)
    p.title(title)
    p.scatter(np.asarray(k_mkt, float).tolist(),
              (np.asarray(iv_mkt, float) * 100).tolist(), label="market")
    p.plot(np.asarray(k_fit, float).tolist(),
           (np.asarray(iv_fit, float) * 100).tolist(), marker="braille", label="SVI")
    p.xlabel("log-moneyness")
    p.ylabel("implied vol %")
    return p.build()


def density(k, dens, title, width=None, height=None):
    """Risk-neutral density implied by a slice. Dipping below zero is arbitrage."""
    p = _plt(width, height)
    if p is None:
        return sparkline(np.asarray(dens), title)
    p.title(title)
    p.plot(np.asarray(k, float).tolist(), np.asarray(dens, float).tolist(),
           marker="braille", label="density")
    p.plot(np.asarray(k, float).tolist(), [0.0] * len(k), marker="dot", label="zero")
    p.xlabel("log-moneyness")
    return p.build()


def bars(labels, values, title, ref=None, xlabel=None, ylabel=None,
         width=None, height=None):
    """One bar per label, coloured by sign. `ref` draws a horizontal line."""
    values = np.asarray(values, float)
    p = _plt(width, height)
    if p is None or values.size == 0:
        return sparkline(values, title) if values.size else title
    p.title(title)
    labels = [str(s) for s in labels]
    # Two series rather than a colour per bar: plotext 5.3 misapplies a list of
    # colours, painting one bar's colour into its neighbour.
    p.bar(labels, np.where(values > 0, values, 0.0).tolist(), color="cyan", width=0.4)
    if (values < 0).any():
        p.bar(labels, np.where(values < 0, values, 0.0).tolist(),
              color="orange", width=0.4)
    if ref is not None:
        p.hline(ref, "white")
    if xlabel:
        p.xlabel(xlabel)
    if ylabel:
        p.ylabel(ylabel)
    return p.build()
