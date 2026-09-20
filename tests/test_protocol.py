import json

import pytest
import tomli_w

from vollab.hedge.registry import build_config, paths_fingerprint_input
from vollab.protocol.hashing import canonical_bytes, config_hash
from vollab.protocol.ledger import Ledger
from vollab.protocol.prereg import HashMismatch, load_registered

CFG = {
    "hypothesis": "sd falls as N^-0.5",
    "contract": {"kind": "call", "S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.0, "q": 0.0},
    "vols": {"s_imp": 0.3, "s_hedge": 0.3, "s_real": 0.3},
    "schedule": {"name": "fixed_time", "every": 1},
    "n_mon": 64, "cost_bps": 0.0, "n_paths": 500, "seed": 1,
}


def _write(tmp_path, cfg, hash_override=None):
    p = tmp_path / "c.toml"
    body = dict(cfg)
    body["config_hash"] = hash_override or config_hash(cfg)
    p.write_bytes(tomli_w.dumps(body).encode())
    return p


def test_hash_is_invariant_to_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_hash_changes_when_a_value_changes():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_stored_hash_field_is_excluded():
    assert config_hash({"a": 1, "config_hash": "deadbeef"}) == config_hash({"a": 1})


def test_canonical_bytes_rejects_nan():
    with pytest.raises(ValueError):
        canonical_bytes({"a": float("nan")})


def test_load_registered_accepts_a_matching_hash(tmp_path):
    assert load_registered(_write(tmp_path, CFG))["hypothesis"] == CFG["hypothesis"]


def test_load_registered_rejects_an_edited_config(tmp_path):
    """Editing a parameter without re-registering must block the run."""
    stale = config_hash(CFG)
    edited = {**CFG, "n_paths": 999}
    with pytest.raises(HashMismatch):
        load_registered(_write(tmp_path, edited, hash_override=stale))


def test_load_registered_requires_a_hash(tmp_path):
    p = tmp_path / "n.toml"
    p.write_bytes(tomli_w.dumps(CFG).encode())
    with pytest.raises(HashMismatch):
        load_registered(p)


def test_load_registered_requires_a_hypothesis(tmp_path):
    bare = {k: v for k, v in CFG.items() if k != "hypothesis"}
    with pytest.raises(ValueError, match="hypothesis"):
        load_registered(_write(tmp_path, bare))


def test_ledger_roundtrip(tmp_path):
    led = Ledger(tmp_path / "l.db")
    row = dict(run_id="r1", schema_version=1, ts="2026-09-19T00:00:00",
               config_hash="abc", config_toml="x=1", hypothesis="h",
               paths_fingerprint="fp", git_commit="c", git_dirty=0, git_diff_sha=None,
               vollab_version="0.1.0", rng_scheme_version=1, engine="numpy",
               engine_build_id=None, numpy_version="2.5.3", scipy_version="1.18.1",
               python_version="3.14.3", platform="Darwin-arm64", status="ok",
               n_paths_completed=1000, metrics=json.dumps({"pnl_mean": 1.0}),
               artifacts="[]", artifact_sha256="[]", runtime_s=1.5)
    led.insert(row)
    assert len(led.all()) == 1
    assert led.get("r1")["status"] == "ok"
    assert led.get("nope") is None


def test_registry_builds_a_config():
    cfg = build_config(CFG)
    assert cfg.n_mon == 64 and cfg.schedule.every == 1


def test_registry_rejects_an_unknown_schedule():
    with pytest.raises(ValueError, match="unknown schedule"):
        build_config({**CFG, "schedule": {"name": "magic"}})


def test_paths_fingerprint_ignores_schedule_and_costs():
    """Two runs differing only in schedule share paths, so must be comparable."""
    a = paths_fingerprint_input(CFG)
    b = paths_fingerprint_input({**CFG, "schedule": {"name": "fixed_time", "every": 8},
                                 "cost_bps": 25.0})
    assert config_hash(a) == config_hash(b)


def test_paths_fingerprint_changes_with_seed():
    a = paths_fingerprint_input(CFG)
    b = paths_fingerprint_input({**CFG, "seed": 2})
    assert config_hash(a) != config_hash(b)
