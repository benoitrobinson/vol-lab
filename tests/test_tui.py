"""The viewer is read-only, so its logic is a pure formatting function."""

import json

import pytest

from vollab.protocol.ledger import Ledger
from vollab.tui.app import ViewerApp, format_run

ROW = dict(
    run_id="abcdef0123456789", schema_version=1, ts="2026-09-20T10:00:00",
    config_hash="feedface0000", config_toml="x=1", hypothesis="sd falls as N^-0.5",
    paths_fingerprint="0badc0de1111", git_commit="deadbeefcafe", git_dirty=1,
    git_diff_sha="aa", vollab_version="0.1.0", rng_scheme_version=1,
    engine="numpy", engine_build_id=None, numpy_version="2.5.3",
    scipy_version="1.18.1", python_version="3.14.3", platform="Darwin-arm64",
    status="ok", n_paths_completed=20000,
    metrics=json.dumps({"pnl_mean": -0.000265, "pnl_sd": 0.46392}),
    artifacts="[]", artifact_sha256="[]", runtime_s=12.5,
)


def test_format_includes_the_hypothesis_and_metrics():
    out = format_run(ROW)
    assert "sd falls as N^-0.5" in out
    assert "pnl_mean" in out and "-0.000265" in out
    assert "pnl_sd" in out


def test_format_flags_a_dirty_tree():
    assert "DIRTY TREE" in format_run(ROW)
    assert "DIRTY TREE" not in format_run({**ROW, "git_dirty": 0})


def test_format_records_provenance():
    out = format_run(ROW)
    assert "rng scheme v1" in out
    assert "numpy 2.5.3" in out
    assert "deadbeef" in out


def test_format_handles_missing_git_commit():
    assert "none" in format_run({**ROW, "git_commit": "", "git_dirty": 0})


def test_viewer_loads_an_empty_ledger(tmp_path):
    app = ViewerApp(tmp_path / "l.db")
    assert app.rows == []


def test_viewer_loads_recorded_runs(tmp_path):
    led = Ledger(tmp_path / "l.db")
    led.insert(ROW)
    assert len(ViewerApp(tmp_path / "l.db").rows) == 1


@pytest.mark.asyncio
async def test_viewer_renders_without_error(tmp_path):
    led = Ledger(tmp_path / "l.db")
    led.insert(ROW)
    app = ViewerApp(tmp_path / "l.db")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.sub_title == "1 runs"
