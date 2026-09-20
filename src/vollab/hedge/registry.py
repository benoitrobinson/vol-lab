"""TOML to configuration objects.

This lives in hedge/ rather than protocol/ so the protocol layer stays
domain-free: it hashes canonical bytes and stores JSON, and Phase B can reuse
it without inheriting anything about hedging.
"""

from vollab.hedge.config import Contract, HedgeConfig, VolSpec
from vollab.hedge.schedule import DeltaBand, FixedTime, Leland, WhalleyWilmott
from vollab.paths.base import MODELS

SCHEDULES = {
    "fixed_time": FixedTime,
    "delta_band": DeltaBand,
    "leland": Leland,
    "whalley_wilmott": WhalleyWilmott,
}

PATH_KEYS = ("contract", "vols", "n_mon", "n_paths", "seed", "mu", "model")


def build_config(d):
    spec = dict(d["schedule"])
    name = spec.pop("name")
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; known: {sorted(SCHEDULES)}")
    mspec = dict(d.get("model", {"name": "gbm"}))
    mname = mspec.pop("name", "gbm")
    if mname not in MODELS:
        raise ValueError(f"unknown model {mname!r}; known: {sorted(MODELS)}")
    return HedgeConfig(
        contract=Contract(**d["contract"]), vols=VolSpec(**d["vols"]),
        schedule=SCHEDULES[name](**spec), model=MODELS[mname](**mspec),
        n_mon=d["n_mon"], cost_bps=d["cost_bps"], n_paths=d["n_paths"],
        seed=d["seed"], mu=d.get("mu"),
        chunk_paths=d.get("chunk_paths", 50_000),
    )


def paths_fingerprint_input(d):
    """Only the inputs that determine the paths, so vl compare can refuse a
    paired bootstrap between runs that did not share them."""
    return {k: d[k] for k in PATH_KEYS if k in d}
