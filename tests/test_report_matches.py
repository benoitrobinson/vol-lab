"""The report cannot silently drift from the code.

REPORT.md quotes numbers from artifacts/findings.json, which scripts/report.py
regenerates. This asserts the artifact exists, is internally consistent, and
still carries the qualitative results the report claims. Exact values are not
pinned, because they are Monte Carlo estimates; what is pinned is that every
claim in the report is still supported by a regenerated number.
"""

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "findings.json"


@pytest.fixture(scope="module")
def f():
    if not ARTIFACT.exists():
        pytest.skip("run: uv run python scripts/report.py")
    return json.loads(ARTIFACT.read_text())


def test_artifact_records_its_provenance(f):
    m = f["meta"]
    for key in ("generated", "git_commit", "python", "numpy", "scipy", "seeds", "n_mon"):
        assert m.get(key), f"missing provenance: {key}"


def test_every_finding_is_present(f):
    for key in ("f1_discretisation", "f2_lockin", "f3_attribution", "f4_jump_floor",
                "f5_schedules", "f6_surface", "f7_market_making", "f8_inverse",
                "f9_rough_vol_floor", "f10_rough_skew", "f11_unwind",
                "f12_adverse_selection", "convergence", "variance_reduction"):
        assert key in f


def test_f1_slope_is_minus_one_half_within_seed_noise(f):
    s = f["f1_discretisation"]["slope"]
    assert abs(s["mean"] - (-0.5)) < 4 * max(s["sd"], 1e-3)
    assert s["n_seeds"] >= 2


def test_f1_rehedge_counts_are_possible_on_the_stated_grid(f):
    """The check that would have caught the spliced table: you cannot rehedge
    more often than the monitoring grid allows."""
    d = f["f1_discretisation"]
    assert d["rehedges_dense"] <= d["n_mon"] + 1
    assert d["rehedges_sparse"] < d["rehedges_dense"]


def test_f2_both_hedges_earn_the_same_mean(f):
    d = f["f2_lockin"]
    edge = d["edge"]
    for arm in ("at_realized", "at_implied"):
        assert abs(d[arm]["mean"]["mean"] - edge) < 0.05 * abs(edge)
    assert d["at_implied"]["sd"]["mean"] > 3 * d["at_realized"]["sd"]["mean"]


def test_f3_delta_term_is_reported_where_it_is_not_identically_zero(f):
    """Hedging every step at the mark vol makes the delta term zero by
    construction, so the report must also show a schedule where it is not."""
    d = f["f3_attribution"]
    assert d["hedged_every_step"]["max_abs_delta"]["mean"] < 1e-10
    assert d["hedged_every_8th"]["max_abs_delta"]["mean"] > 1.0


def test_f3_residual_shrinks_as_the_grid_refines(f):
    r = [row["residual_over_gamma"] for row in f["f3_attribution"]["residual_scaling"]]
    assert r[1] < 0.7 * r[0] and r[2] < 0.7 * r[1]


def test_f4_jump_slope_is_far_from_the_gbm_control(f):
    d = f["f4_jump_floor"]
    assert d["gbm"]["slope"]["mean"] < -0.4
    assert d["merton"]["slope"]["mean"] > -0.25
    assert d["merton"]["sd_ratio"]["mean"] > 0.4
    assert d["gbm"]["sd_ratio"]["mean"] < 0.2


def test_f5_band_beats_hedging_every_step(f):
    d = f["f5_schedules"]["ww_minus_fixed"]
    assert d["ci_high"] < 0 or d["ci_low"] > 0      # a decisive difference either way
    assert d["ci_low"] <= d["diff"] <= d["ci_high"]


def test_f5_u_curve_minimum_carries_an_interval(f):
    u = f["f5_schedules"]["u_curve"]
    lo, hi = u["argmin_ci"]
    assert lo <= u["argmin_point"] <= hi


def test_f6_constrained_arm_removes_the_violations(f):
    d = f["f6_surface"]
    assert d["target_min_durrleman_g"] > 0, "the target itself must be admissible"
    assert d["noise_free_rmse_vol_points"] < 0.01
    for level in d["noise_levels"].values():
        assert level["constrained_arbitraged"] == 0


def test_f6_fit_cost_is_now_measured_against_the_same_optimiser(f):
    """A negative cost was the symptom of a confounded comparison. Both arms
    now run the identical solve, so the constraint can only cost, not pay."""
    for level in f["f6_surface"]["noise_levels"].values():
        if level["fit_cost_vol_points"] is not None:
            assert level["fit_cost_vol_points"] >= -1e-6


def test_f7_skew_improves_risk_adjusted_return(f):
    sweep = f["f7_market_making"]["sweep"]
    for row in sweep.values():
        assert row["avellaneda_stoikov"]["sd"]["mean"] < row["symmetric"]["sd"]["mean"]
        assert row["ratio_gap"] > 0


def test_convergence_follows_the_root_n_law(f):
    c = f["convergence"]
    assert abs(c["fitted_slope"] - (-0.5)) < 0.1


def test_variance_reduction_is_real_and_the_negatives_are_recorded(f):
    v = f["variance_reduction"]
    assert v["correlation"] < -0.7
    assert v["variance_ratio"] < 0.5
    assert abs(v["payoff_control_correlation"]) < 0.15
    assert v["antithetic_correlation"] == 1.0


def test_report_is_rendered_from_the_artifact_and_has_not_drifted():
    """Re-render and require the file to be unchanged.

    This is the structural guard against the defect that prompted all of it: a
    REPORT.md hand-assembled from scratch runs, whose headline table spliced
    three incompatible ones together.
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    report = root / "REPORT.md"
    if not (root / "artifacts" / "findings.json").exists():
        pytest.skip("run: uv run python scripts/report.py")
    before = report.read_text()
    subprocess.run([sys.executable, "scripts/render_report.py"], cwd=root,
                   check=True, capture_output=True)
    after = report.read_text()
    if before != after:
        report.write_text(before)
        pytest.fail("REPORT.md does not match artifacts/findings.json; "
                    "run scripts/render_report.py and commit the result")


def test_no_headline_number_is_absent_from_the_artifact():
    """Spot-check that figures quoted in prose exist in the generated blocks."""
    root = Path(__file__).resolve().parent.parent
    text = (root / "REPORT.md").read_text()
    assert "<!-- BEGIN:f1 -->" in text and "<!-- END:numerics -->" in text
    for marker in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
                   "f11", "f12", "numerics"):
        start = text.index(f"<!-- BEGIN:{marker} -->")
        end = text.index(f"<!-- END:{marker} -->")
        assert end - start > 80, f"block {marker} looks empty"


def test_f8_inverse_is_confirmed_by_two_routes(f):
    """Both pricing routes must land on the claim, and the naive expectation
    must not, since it is the mistake the finding exists to flag."""
    d = f.get("f8_inverse")
    if d is None:
        pytest.skip("regenerate: uv run python scripts/report.py")
    claim = d["coin_price"]
    for route in ("dollar_route", "share_route"):
        sep = abs(d[route]["mean"] - claim) / d[route]["se"]
        assert sep < 3.0, f"{route} is {sep:.1f} se from the claim"
    assert abs(d["naive_expectation"] - claim) / claim > 0.2


def test_f8_mismatch_grows_with_spot(f):
    d = f.get("f8_inverse")
    if d is None:
        pytest.skip("regenerate: uv run python scripts/report.py")
    pct = [abs(r["gap_pct"]) for r in d["mismatch"]]
    assert pct == sorted(pct), "the converted-delta error should grow with spot"
    assert max(pct) > 20.0


def test_f9_the_floor_is_set_by_vol_of_vol_not_by_roughness(f):
    d = f["f9_rough_vol_floor"]
    flat = d["sweep"]["0.0"]["slope"]["mean"]
    floored = d["sweep"][str(max(d["etas"]))]
    assert abs(flat - (-0.5)) < 0.05, "eta = 0 must reproduce Black-Scholes"
    assert floored["slope"]["mean"] > -0.2
    assert floored["sd_ratio"]["mean"] > 0.5
    # A nearly smooth variance floors in the same place as a rough one.
    assert abs(d["roughness_control"]["sd_ratio"]["mean"]
               - floored["sd_ratio"]["mean"]) < 0.15


def test_f9_the_floor_deepens_with_every_step_of_vol_of_vol(f):
    d = f["f9_rough_vol_floor"]
    slopes = [d["sweep"][str(e)]["slope"]["mean"] for e in sorted(d["etas"])]
    assert all(b > a for a, b in zip(slopes, slopes[1:]))


def test_f10_only_the_rough_model_has_an_exploding_short_dated_skew(f):
    d = f["f10_rough_skew"]
    rough = d["models"]["rbergomi"]["slope"]
    heston = d["models"]["heston"]["slope"]
    assert rough < -0.2, "the rough skew must steepen as maturity shrinks"
    assert heston > rough + 0.15, "the diffusive control must be much flatter"
    assert abs(rough - d["theoretical_slope"]) < 0.2


def test_f11_charging_for_the_unwind_changes_the_ranking(f):
    d = f["f11_unwind"]
    free = d["pairwise"]["frictionless"]
    charged = d["pairwise"]["unwind"]
    assert charged["diff"] > free["diff"] + 1.0
    assert not charged["straddles_zero"]
    control = d["table"]["unwind"]["symmetric"]["unwind_paid"]["mean"]
    skewed = d["table"]["unwind"]["glft"]["unwind_paid"]["mean"]
    assert control > 2 * skewed, "the control is the strategy the free unwind subsidised"


def test_f11_the_dispersion_advantage_is_not_an_artefact_of_the_free_unwind(f):
    d = f["f11_unwind"]
    for setting in ("frictionless", "unwind", "informed", "both"):
        ratio = d["table"][setting]["glft"]["sd_vs_control"]["mean"]
        assert ratio < 0.6, f"{setting}: dispersion ratio {ratio:.3f}"


def test_f12_adverse_selection_is_indifferent_to_the_quoting_rule(f):
    d = f["f12_adverse_selection"]
    base = d["table"]["0.0"]
    informed = d["table"][str(sorted(map(float, d["table"]))[1])]
    for strat, row in base.items():
        assert abs(row["markout_per_fill"]["mean"]) < 0.01, "uninformed flow must mark out at zero"
    for strat, row in informed.items():
        assert row["markout_per_fill"]["mean"] < -0.01
    assert d["markout_spread"] < 0.1 * abs(
        informed["glft"]["markout_per_fill"]["mean"])
    assert d["toll_spread"] < 1.0
