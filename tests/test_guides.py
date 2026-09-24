"""Guides (key g) and the graded exercises (tab 0)."""

from pathlib import Path

import pytest

from vollab.tui import exercises as ex
from vollab.tui.app import HELP_TEXT_COLUMNS, HelpScreen, VolLabApp
from vollab.tui.guides import BY_TAB, guide_text, tutorial_markdown

TITLES = {e.key: e.title for e in ex.EXERCISES}


def test_every_tab_has_a_guide():
    assert set(BY_TAB) == set(VolLabApp.TABS)


@pytest.mark.parametrize("tab", VolLabApp.TABS)
def test_guides_fit_the_box_and_name_real_exercises(tab):
    text = guide_text(tab, HELP_TEXT_COLUMNS, TITLES)
    assert max(len(line) for line in text.splitlines()) <= HELP_TEXT_COLUMNS
    for key in BY_TAB[tab].exercises:
        assert ex.BY_KEY[key].tab == tab


def test_every_exercise_is_listed_on_its_tab():
    for e in ex.EXERCISES:
        assert e.key in BY_TAB[e.tab].exercises, e.key


def test_tutorial_matches_the_guides():
    committed = Path(__file__).resolve().parent.parent / "docs" / "TUTORIAL.md"
    assert committed.read_text() == tutorial_markdown(TITLES), (
        "docs/TUTORIAL.md is stale: uv run python scripts/render_tutorial.py")


# --- grading, without running anything --------------------------------------

def test_nearest_ratio_reads_a_log_scale():
    assert ex.nearest_ratio(0.54, (0.25, 0.5, 1.0, 2.0)) == 1
    assert ex.nearest_ratio(1.9, (0.25, 0.5, 1.0, 2.0)) == 3


def test_direction_has_a_dead_band():
    assert ex.direction(2.0, 2.4) == 0
    assert ex.direction(2.0, 1.6) == 1
    assert ex.direction(2.0, 2.02) == 2


def test_best_or_tie_calls_close_values_a_tie():
    assert ex.best_or_tie([-0.0343, -0.0333, -0.0338], rel_tol=0.10) == 3
    assert ex.best_or_tie([-0.05, -0.01, -0.04], rel_tol=0.10) == 1


def test_grade_marks_and_explains():
    e = ex.Exercise("k", "hedge", "t", "q?", ("x", "y"),
                    lambda: (1, ["measured line"]), "because")
    assert ex.grade(e, 1).startswith("RIGHT")
    wrong = ex.grade(e, 0)
    assert wrong.startswith("NOT QUITE") and "measured: y" in wrong
    assert "measured line" in wrong and "because" in wrong


def test_an_unavailable_sibling_is_reported_not_graded(monkeypatch, tmp_path):
    monkeypatch.setenv("LOBLAB_HOME", str(tmp_path / "absent"))
    out = ex.grade(ex.BY_KEY["latency"], 0)
    assert "cannot run this one yet" in out and "RIGHT" not in out


# --- every exercise runs against the real engines ---------------------------

@pytest.mark.parametrize("key", [e.key for e in ex.EXERCISES if e.tab != "book"])
def test_vol_lab_exercises_run_and_answer_as_explained(key, monkeypatch):
    """At small sample sizes the answer must still be the one the WHY explains;
    if it flips, the exercise is too close to call and needs a sharper setup."""
    monkeypatch.setitem(ex.PATHS, "hedge", 1000)
    monkeypatch.setitem(ex.PATHS, "mm", 500)
    expected = {"frequency": 1, "costs": 1, "aversion": 1, "informed": 3,
                "pairing": 2}
    answer, measured = ex.BY_KEY[key].run()
    assert answer == expected[key], measured


@pytest.mark.parametrize("key", [e.key for e in ex.EXERCISES if e.tab == "book"])
def test_lob_lab_exercises_run_when_built(key):
    e = ex.BY_KEY[key]
    try:
        answer, measured = e.run()
    except ex.siblings.Unavailable as exc:
        pytest.skip(str(exc).splitlines()[0])
    assert 0 <= answer < len(e.options) and measured


# --- the panel --------------------------------------------------------------

@pytest.mark.asyncio
async def test_g_opens_the_guide_for_the_current_tab():
    app = VolLabApp(None)
    async with app.run_test() as pilot:
        await pilot.press("4")
        await pilot.press("g")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        assert "Hedge: what delta hedging costs" in app.screen.text


@pytest.mark.asyncio
async def test_zero_opens_exercises_and_a_letter_grades(monkeypatch):
    from textual.widgets import TabbedContent
    monkeypatch.setitem(ex.PATHS, "hedge", 500)
    app = VolLabApp(None)
    async with app.run_test() as pilot:
        await pilot.press("0")
        await pilot.pause()
        assert app.query_one(TabbedContent).active == "exercises"
        await pilot.press("b")
        await app.workers.wait_for_complete()
        await pilot.pause()
        body = str(app.query_one("#exercise-body").render())
        assert "RIGHT" in body and "WHY" in body


@pytest.mark.asyncio
async def test_letters_do_nothing_outside_the_exercises_tab():
    app = VolLabApp(None)
    async with app.run_test() as pilot:
        await pilot.press("4")
        await pilot.press("a")
        await app.workers.wait_for_complete()
        assert "commit to an answer" in str(app.query_one("#exercise-body").render())
