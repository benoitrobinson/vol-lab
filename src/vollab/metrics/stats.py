"""Summary statistics over a frequency sweep."""

import numpy as np

from vollab.hedge.schedule import FixedTime
from vollab.hedge.simulator import sweep


def sd_vs_frequency(cfg, every_list):
    """Dispersion against realized rehedge count.

    Every frequency runs on the same Brownian-nested paths: the paths are
    generated once on the fine monitoring grid and each schedule subsamples it.
    An unpaired sweep would widen the error on the fitted slope badly.
    """
    results = sweep(cfg, [FixedTime(every=e) for e in every_list])
    n = np.array([r.n_rehedges.mean() for r in results])
    sd = np.array([r.pnl.std(ddof=1) for r in results])
    return n, sd


def loglog_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)[0])


def cost_curve(cfg, every_list):
    """Risk-adjusted total cost against rehedge frequency (finding F5).

    The objective is mean loss plus one standard deviation of P&L: hedging more
    often cuts dispersion but pays more spread, so the sum is U-shaped once
    costs are non-zero. All frequencies share the same paths.
    """
    results = sweep(cfg, [FixedTime(every=e) for e in every_list])
    n = np.array([r.n_rehedges.mean() for r in results])
    obj = np.array([-r.pnl.mean() + r.pnl.std(ddof=1) for r in results])
    return n, obj
