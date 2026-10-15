"""Regression coverage for issue #4; synthetic data only."""
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl
from adata_science_tools._plotting import _corr_dotplots as module


@pytest.fixture
def data():
    return pd.DataFrame({
        "x": [1., 2., 3., 4., 4., 4., 9., 10., 11., 12.],
        "y": [5., 5., 5., 2., 3., 4., 8., 9., 13., 12.],
        "group": ["constant_y"] * 3 + ["constant_x"] * 3
                 + ["single"] + ["varying"] * 3,
    })


@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("entrypoint", [adtl.corr_dotplot, adtl.corr_dotplot_dev])
def test_nonestimable_subgroups(data, method, entrypoint):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = entrypoint(df=data, column_key_x="x", column_key_y="y",
                            subset_key="group", method=method, show_all_obs_fit=True,
                            show_y_intercept=False, axes_lines=False, show=False)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)
                or "ConstantInput" in w.category.__name__
                or "SmallSample" in w.category.__name__]
    assert len(result) == 5
    fig, axes, fit, corr, pvalue = result
    if isinstance(axes, dict):
        axes = axes["main"]
    try:
        expected = getattr(stats, method + "r")(data.x, data.y)
        assert corr == pytest.approx(expected.statistic)
        assert pvalue == pytest.approx(expected.pvalue)
        assert fit == stats.linregress(data.x, data.y)
        text = "\n".join(t.get_text() for t in fig.texts)
        legend = "\n".join(t.get_text() for t in axes.get_legend().texts)
        assert "group=constant_x: fit unavailable: constant x" in text
        assert "group=single: fit unavailable: insufficient observations" in text
        assert "correlation unavailable: constant y" in text
        assert "R^2 unavailable: constant y" in text
        assert "correlation unavailable: constant y" in legend
        assert "nan" not in (text + legend).lower()
        assert sum(len(c.get_offsets()) for c in axes.collections) == len(data)
        lines = {line.get_label().split("\n")[0]: line for line in axes.lines}
        assert set(lines) == {"All data", "group=constant_y", "group=varying"}
        assert np.allclose(lines["group=constant_y"].get_ydata(), 5.)
        varying = data[data.group == "varying"]
        expected_fit = stats.linregress(varying.x, varying.y)
        line = lines["group=varying"]
        assert np.allclose(line.get_ydata(), expected_fit.intercept
                           + expected_fit.slope * np.asarray(line.get_xdata()))
        expected_corr = getattr(stats, method + "r")(varying.x, varying.y)
        assert f"Corr={expected_corr.statistic:.3f},p={expected_corr.pvalue:.2e}" in legend
    finally:
        plt.close(fig)


@pytest.mark.parametrize("entrypoint", [adtl.corr_dotplot, adtl.corr_dotplot_dev])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_unexpected_subgroup_calculation_error_propagates(data, entrypoint, method):
    original = module._compute_corr_and_fit
    def calculate(x, y, method):
        if len(x) == 3:
            raise RuntimeError("unexpected implementation error")
        return original(x, y, method)
    with patch.object(module, "_compute_corr_and_fit", side_effect=calculate):
        with pytest.raises(RuntimeError, match="unexpected implementation error"):
            entrypoint(df=data, column_key_x="x", column_key_y="y",
                       subset_key="group", method=method, show=False)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_two_observations_preserve_finite_estimates(method):
    x, y = pd.Series([1., 2.]), pd.Series([3., 4.])
    fit, corr, pvalue, reason = module._try_compute_corr_and_fit(x, y, method)
    expected = getattr(stats, method + "r")(x, y)
    assert corr == expected.statistic
    np.testing.assert_allclose(fit, stats.linregress(x, y), equal_nan=True)
    assert reason is None
    text = module._format_subset_stats_line("pair", method, fit, corr, pvalue)
    legend = module._format_subset_fit_legend_label("pair", corr, pvalue)
    assert "nan" not in (text + legend).lower()
    if method == "spearman":
        assert "pvalue unavailable" in text
    else:
        assert pvalue == expected.pvalue


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_formatters_explain_nonfinite_statistics(value):
    fit = stats.linregress([1., 2., 3.], [3., 2., 4.])
    text = module._format_subset_stats_line("group", "pearson", fit, value, value)
    legend = module._format_subset_fit_legend_label("group", value, value)
    assert "correlation unavailable: non-finite result" in text
    assert "correlation unavailable: non-finite result" in legend


@pytest.mark.parametrize("x,y,reason", [
    ([], [], "insufficient observations"),
    ([1.], [2.], "insufficient observations"),
    ([1., 1.], [2., 2.], "constant x"),
    ([1., 1.], [2., 3.], "constant x"),
    ([1., 2.], [3., 3.], "constant y"),
    ([1., np.nan], [2., 3.], "non-finite input"),
])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_known_cases_do_not_invoke_undefined_calculations(x, y, reason, method):
    with patch.object(module, "_compute_corr_and_fit", side_effect=AssertionError):
        fit, corr, pvalue, actual_reason = module._try_compute_corr_and_fit(
            pd.Series(x, dtype=float), pd.Series(y, dtype=float), method)
    assert reason in actual_reason
    assert corr is None and pvalue is None
    if reason == "constant y":
        assert fit.slope == 0 and fit.intercept == 3.
        assert np.isnan(fit.rvalue)
    else:
        assert fit is None
