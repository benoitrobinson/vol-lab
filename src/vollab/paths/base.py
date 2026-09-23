"""Path model selection.

The model carries only its own parameters. Spot, rate, dividend and maturity
come from the contract, and the diffusion level comes from VolSpec.s_real, so
a sweep over realized volatility never has to reach inside a model object.
"""

from dataclasses import dataclass

from vollab.paths.gbm import gbm_paths
from vollab.paths.heston import heston_paths
from vollab.paths.merton import merton_paths
from vollab.paths.rbergomi import rbergomi_paths


@dataclass(frozen=True)
class GBM:
    name: str = "gbm"


@dataclass(frozen=True)
class Heston:
    kap_h: float
    th_h: float
    xi: float
    rho: float
    name: str = "heston"


@dataclass(frozen=True)
class RoughBergomi:
    """H below 1/2 is the rough regime; H = 1/2 is a lognormal Bergomi model."""

    H: float
    eta: float
    rho: float
    name: str = "rbergomi"


@dataclass(frozen=True)
class Merton:
    lam: float
    mu_J: float
    s_J: float
    name: str = "merton"


MODELS = {"gbm": GBM, "heston": Heston, "merton": Merton,
          "rbergomi": RoughBergomi}


def generate(model, c, v, n_mon, seed, path_start, n_paths, mu=None):
    if isinstance(model, GBM):
        return gbm_paths(c.S0, c.r, c.q, v.s_real, c.T, n_mon,
                         seed, path_start, n_paths, mu)
    if isinstance(model, Merton):
        return merton_paths(c.S0, c.r, c.q, v.s_real, c.T, n_mon,
                            seed, path_start, n_paths,
                            model.lam, model.mu_J, model.s_J, mu)
    if isinstance(model, Heston):
        # s_real is the initial volatility, so v0 is its square.
        return heston_paths(c.S0, c.r, c.q, c.T, n_mon, seed, path_start, n_paths,
                            v.s_real ** 2, model.kap_h, model.th_h,
                            model.xi, model.rho, mu)
    if isinstance(model, RoughBergomi):
        # s_real is the initial volatility, so the flat forward variance curve
        # starts at its square.
        return rbergomi_paths(c.S0, c.r, c.q, c.T, n_mon, seed, path_start, n_paths,
                              v.s_real ** 2, model.H, model.eta, model.rho, mu)
    raise ValueError(f"unknown model {model!r}")
