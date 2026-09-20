"""Configuration and result types for the hedging experiment."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Contract:
    kind: Literal["call", "put"]
    S0: float
    K: float
    T: float
    r: float
    q: float


@dataclass(frozen=True)
class VolSpec:
    """The three volatilities are separate inputs; their differences are F2."""

    s_imp: float
    s_hedge: float
    s_real: float


@dataclass(frozen=True)
class HedgeConfig:
    contract: Contract
    vols: VolSpec
    schedule: object
    n_mon: int
    cost_bps: float
    n_paths: int
    seed: int
    model: object = None          # None means GBM; see paths.base
    attribute: bool = True        # False skips the P&L explain, for fair benchmarking
    mu: float | None = None
    chunk_paths: int = 50_000
    trace_paths: tuple[int, ...] = ()


@dataclass(frozen=True)
class HedgeResult:
    pnl: np.ndarray
    n_rehedges: np.ndarray
    turnover: np.ndarray
    rehedge_mask_hash: np.ndarray
    engine_used: str
    rng_scheme_version: int
    attribution: object | None = None
    trace: dict | None = None
