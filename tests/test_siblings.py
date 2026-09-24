"""Tabs 8 and 9: lob-lab and contract-lab, run as sibling binaries."""

import pytest

from vollab.tui import siblings
from vollab.tui.app import VolLabApp

BOOK = {
    "home": "/x/lob-lab",
    "headline": {
        "days": 1, "fill_inflation_mean": 2.07, "fill_inflation_ci95": [2.07, 2.07],
        "cancel_position_sensitivity": {
            f"ahead{p}": {"cancels_from_ahead_pct": p, "fill_inflation": r,
                          "pnl_gap": -0.2}
            for p, r in ((0, 2.07), (50, 1.86), (100, 1.72))},
    },
    "runs": [
        {"quoter": "touch", "model": "naive", "fills": "184",
         "pnl_btc": "-0.000023", "markout_1s_bps": "-0.589"},
        {"quoter": "touch", "model": "ahead50", "fills": "99",
         "pnl_btc": "-0.00002", "markout_1s_bps": "-0.8"},
    ],
    "ofi": {"book_updates": 717, "horizons": [
        {"horizon_ms": 500, "sign_pct": 57.7, "moved": 26},
        {"horizon_ms": 30000, "sign_pct": None, "moved": 0}]},
}

CONTRACTS = {
    "home": "/x/contract-lab",
    "provenance": ["Fitted 2026-09-23 08:54 UTC."],
    "study": {"forward": 86203.19, "expiry_years": 0.1013, "atm_vol": 0.3639,
              "rows": [{"moneyness": 0.9, "strike": 77582.9, "n_d2": 0.78376,
                        "call_spread": 0.82516, "difference_bps": 414.0},
                       {"moneyness": 1.1, "strike": 94823.5, "n_d2": 0.19498,
                        "call_spread": 0.17242, "difference_bps": -225.6}]},
    "checks": [{"label": "knock-out, Boyle-Lau steps", "value": 2825.0, "se": None},
               {"label": "american put, Longstaff-Schwartz", "value": 4.4584,
                "se": 0.0148}],
    "termsheet": [{"name": "digital call", "contract": "when (...) scale 1 of one"}],
}


def test_book_says_one_day_is_not_a_result():
    body = siblings.book_body(BOOK, 80, 12)
    assert "not a result" in body and "no interval" in body
    assert "1.72x to 2.07x" in body
    assert "57.7% of 26" in body, "a horizon with no moves is dropped, not drawn"
    assert "ahead50" not in body, "the sweep is charted, not repeated in the table"


def test_book_quotes_the_interval_once_there_are_days():
    data = {**BOOK, "headline": {**BOOK["headline"], "days": 9,
                                 "fill_inflation_ci95": [1.8, 2.3]}}
    body = siblings.book_body(data, 80, 12)
    assert "[1.80, 2.30] over 9 days" in body and "not a result" not in body


def test_contracts_body_carries_the_table_checks_and_term_sheets():
    body = siblings.contracts_body(CONTRACTS, 80, 12)
    assert "+414.0" in body and "-225.6" in body
    assert "2825" in body and "4.4584 +/- 0.0148" in body
    assert "scale 1 of one" in body and "Fitted 2026-09-23" in body


@pytest.mark.parametrize("load", [siblings.load_book, siblings.load_contracts])
def test_a_missing_sibling_says_where_it_looked(load, tmp_path):
    with pytest.raises(siblings.Unavailable, match="not found at"):
        load(tmp_path / "absent")


@pytest.mark.parametrize("load", [siblings.load_book, siblings.load_contracts])
def test_an_unbuilt_sibling_says_how_to_build_it(load, tmp_path):
    with pytest.raises(siblings.Unavailable, match="not built"):
        load(tmp_path)


@pytest.mark.asyncio
async def test_the_tab_shows_the_build_hint_rather_than_failing(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTRACTLAB_HOME", str(tmp_path))
    app = VolLabApp(None)
    async with app.run_test() as pilot:
        await pilot.press("9")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "not built" in str(app.query_one("#contracts-out").render())


# The real binaries, when they are built beside vol-lab. Skipped otherwise, so
# vol-lab's own suite never depends on a Rust or OCaml toolchain.

def _real(load):
    try:
        return load()
    except siblings.Unavailable as exc:
        pytest.skip(str(exc).splitlines()[0])


def test_contract_lab_json_renders():
    data = _real(siblings.load_contracts)
    assert len(data["study"]["rows"]) >= 5
    assert "N(d2)" in siblings.contracts_body(data, 80, 12)


def test_lob_lab_json_renders():
    data = _real(siblings.load_book)
    assert data["ofi"]["horizons"]
    assert "fill-at-touch" in siblings.book_body(data, 80, 12)
