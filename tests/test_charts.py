import numpy as np

from vollab.render.charts import curve, histogram, loglog, sparkline


def test_loglog_returns_a_non_empty_string():
    out = loglog(np.array([16.0, 64.0, 256.0]), np.array([1.0, 0.5, 0.25]),
                 "sd vs N", ref_slope=-0.5)
    assert isinstance(out, str) and out.strip()


def test_histogram_returns_a_non_empty_string():
    assert histogram(np.random.default_rng(0).standard_normal(500), "pnl").strip()


def test_curve_returns_a_non_empty_string():
    assert curve(np.arange(10.0), np.arange(10.0) ** 2, "u").strip()


def test_charts_do_not_raise_on_a_single_point():
    assert isinstance(loglog(np.array([1.0]), np.array([1.0]), "t"), str)


def test_sparkline_reports_the_range():
    out = sparkline(np.array([0.0, 1.0, 2.0]), "t")
    assert "0" in out and "2" in out


def test_sparkline_handles_a_flat_series():
    assert isinstance(sparkline(np.ones(5), "flat"), str)


def test_bars_draw_both_signs_and_a_reference():
    from vollab.render.charts import bars
    out = bars(["0.9", "1.1"], [414.0, -225.6], "bps", ref=0.0)
    assert "414" in out and "-225.6" in out
