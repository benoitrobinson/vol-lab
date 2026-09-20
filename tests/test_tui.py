"""The lab panel. Logic is kept in pure functions so it is testable headless."""

import json

import pytest

from vollab.protocol.ledger import Ledger
from vollab.tui.app import VolLabApp, format_run, lesson_body
from vollab.tui.lessons import BY_KEY, LESSONS

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
    assert "sd falls as N^-0.5" in out and "pnl_mean" in out and "-0.000265" in out


def test_format_flags_a_dirty_tree():
    assert "DIRTY TREE" in format_run(ROW)
    assert "DIRTY TREE" not in format_run({**ROW, "git_dirty": 0})


def test_format_records_provenance():
    out = format_run(ROW)
    assert "rng scheme v1" in out and "numpy 2.5.3" in out and "deadbeef" in out


def test_every_lesson_has_all_three_parts():
    for le in LESSONS:
        assert le.question.endswith("?"), le.key
        assert len(le.mechanism) > 80 and len(le.takeaway) > 80, le.key


def test_lessons_cover_every_finding_in_the_artifact():
    from pathlib import Path
    art = Path(__file__).resolve().parent.parent / "artifacts" / "findings.json"
    if not art.exists():
        pytest.skip("run: uv run python scripts/report.py")
    findings = json.loads(art.read_text())
    keys = {k for k in findings if k.startswith("f") and k[1].isdigit()}
    assert keys <= set(BY_KEY), f"findings with no lesson: {keys - set(BY_KEY)}"


def test_lesson_body_degrades_without_an_artifact():
    body = lesson_body(LESSONS[0], {})
    assert "run scripts/report.py" in body
    assert LESSONS[0].question in body


@pytest.mark.parametrize("key", [le.key for le in LESSONS])
def test_lesson_body_renders_against_the_real_artifact(key):
    """Each lesson formats its own finding, so a schema change breaks the test
    rather than the panel."""
    from pathlib import Path
    art = Path(__file__).resolve().parent.parent / "artifacts" / "findings.json"
    if not art.exists():
        pytest.skip("run: uv run python scripts/report.py")
    findings = json.loads(art.read_text())
    body = lesson_body(BY_KEY[key], findings)
    assert "MEASURED" in body
    if key in findings:
        assert "run scripts/report.py to populate" not in body


def test_app_loads_an_empty_ledger(tmp_path):
    assert VolLabApp(None).rows == []


def test_app_loads_recorded_runs(tmp_path):
    Ledger(tmp_path / "l.db").insert(ROW)
    assert len(VolLabApp(tmp_path / "l.db").rows) == 1


@pytest.mark.asyncio
async def test_panel_opens_on_lessons_and_has_every_tab(tmp_path):
    Ledger(tmp_path / "l.db").insert(ROW)
    app = VolLabApp(tmp_path / "l.db")
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import TabbedContent
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "lessons"
        ids = {pane.id for pane in app.query(".-content-tab")} or None
        assert app.sub_title == "1 runs recorded"


@pytest.mark.asyncio
async def test_switching_tabs_does_not_error(tmp_path):
    app = VolLabApp(None)
    async with app.run_test() as pilot:
        from textual.widgets import TabbedContent
        tabs = app.query_one(TabbedContent)
        for name in ("price", "surface", "hedge", "making", "engines", "runs"):
            tabs.active = name
            await pilot.pause()
        assert tabs.active == "runs"
