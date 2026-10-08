"""Synthetic display-label regressions; no biological data are used."""
import inspect
import sys
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl
from adata_science_tools._plotting._feature_labels import _resolve_feature_labels
import test_column_plot_renderers as column_fixtures

MISSING = [None, pd.NA, np.nan, "", "   ", "nan", "NaN", " nAn "]
COLUMNS = ["barh_column", "l2fc_dotplot_single", "l2fc_dotplot_column",
           "barh_l2fc_dotplot_column", "barh_dotplot_dotplot_column",
           "barh_dotplot_dotplot_dotplot_column", "barh_4X_dotplot_column",
           "datapoints_effect_panels_column",
           "plot_column_of_bar_h_2groups_GEX_adata",
           "plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata"]
VOLCANOS = ["volcano_plot_generic", "volcano_plot_sns_single_comparison_generic"]

@pytest.fixture(autouse=True)
def close_figures():
    with mock.patch.object(plt, "show"), mock.patch("builtins.print"):
        yield
    plt.close("all")

@pytest.mark.parametrize("missing", MISSING)
def test_resolver_missing_valid_both_missing_and_no_mutation(missing):
    frame = pd.DataFrame({"preferred": [missing, "  VALID  ", missing],
                          "fallback": ["ID", "OTHER", missing]}, index=["a", "b", "c"])
    original = frame.copy(deep=True)
    result = _resolve_feature_labels(frame, "preferred", "fallback")
    assert result.iloc[0] == "ID"
    assert result.iloc[1] == "  VALID  "
    pd.testing.assert_series_equal(result.iloc[2:], frame.preferred.astype(object).iloc[2:])
    assert _resolve_feature_labels(frame, "preferred", None) is None
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="feature_label_fallback"):
        _resolve_feature_labels(frame, "preferred", "absent")


def geometry(fig):
    """Only scientific artist coordinates, not text/layout pixel positions."""
    return [([np.asarray(c.get_offsets()).tolist() for c in ax.collections],
             [(np.asarray(l.get_xdata()).tolist(), np.asarray(l.get_ydata()).tolist())
              for l in ax.lines]) for ax in fig.axes]


def render_column(name, frame, fallback="omitted"):
    fixture = column_fixtures.ColumnPlotRendererTests(); fixture.setUp()
    kwargs = dict(fixture.direct_table_kwargs, var_df=frame)
    signature = inspect.signature(getattr(adtl, name))
    kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    for k, value in {"include_stripplot": False, "barh_legend": False,
                     "legend": False, "dotplot_legend": False,
                     "dotplot2_legend": False, "dotplot3_legend": False,
                     "dotplot4_legend": False, "use_tight_layout": False}.items():
        if k in signature.parameters:
            kwargs[k] = value
    if fallback != "omitted":
        kwargs["feature_label_fallback"] = fallback
    np.random.seed(4)
    # Legacy seaborn bootstrap uses an unseeded default_rng, independently of
    # NumPy global jitter. Fix only that test RNG, not library computations.
    default_rng = np.random.default_rng
    with mock.patch.object(np.random, "default_rng",
                           side_effect=lambda seed=None: default_rng(4 if seed is None else seed)):
        result = getattr(adtl, name)(**kwargs)
    return result[0]


@pytest.mark.parametrize("name", COLUMNS)
@pytest.mark.parametrize("missing", MISSING)
def test_column_labels_geometry_identity_and_inputs(name, missing):
    fixture = column_fixtures.ColumnPlotRendererTests(); fixture.setUp()
    frame = fixture.var_df.copy()
    frame["feature_label"] = [missing, "  VALID  "]
    frame["fallback"] = ["FALLBACK", "ignored"]
    before = frame.copy(deep=True)
    old = render_column(name, frame)
    old_geometry = geometry(old)
    new = render_column(name, frame, "fallback")
    assert geometry(new) == old_geometry
    assert any(t.get_text() == "FALLBACK" for t in new.findobj(Text))
    assert any(t.get_text() == "  VALID  " for t in new.findobj(Text))
    pd.testing.assert_frame_equal(frame, before)
    plt.close(old); plt.close(new)


@pytest.mark.parametrize("name", COLUMNS)
def test_column_defaults_absent_fallback_both_missing_duplicate_labels(name):
    fixture = column_fixtures.ColumnPlotRendererTests(); fixture.setUp()
    frame = fixture.var_df.copy()
    original = render_column(name, frame)
    explicit = render_column(name, frame, None)
    assert geometry(original) == geometry(explicit)
    assert [t.get_text() for t in original.findobj(Text)] == [t.get_text() for t in explicit.findobj(Text)]
    with pytest.raises(ValueError, match="feature_label_fallback"):
        render_column(name, frame, "absent")
    frame["fallback"] = [None, "nan"]
    frame["feature_label"] = [None, " "]
    old = render_column(name, frame)
    new = render_column(name, frame, "fallback")
    assert [t.get_text() for t in old.findobj(Text)] == [t.get_text() for t in new.findobj(Text)]
    frame["feature_label"] = [None, " "]
    frame["fallback"] = ["DUPLICATE", "DUPLICATE"]
    new = render_column(name, frame, "fallback")
    assert sum(t.get_text() == "DUPLICATE" for t in new.findobj(Text)) >= 2


@pytest.mark.parametrize("name", VOLCANOS)
@pytest.mark.parametrize("missing", MISSING)
def test_volcano_display_only(name, missing):
    frame = pd.DataFrame({"gene_names": [missing, "  VALID  "], "fallback": ["FALLBACK", "ignored"],
                          "log2FoldChange": [-1., 1.], "pvalue": [.01, .01], "padj": [.01, .01]})
    before = frame.copy(deep=True)
    kwargs = dict(label_top_features=True, n_top_features=2, xlimit=2, ylimit=4)
    fn = getattr(adtl, name)
    if name == "volcano_plot_generic":
        kwargs["label_layout"] = "ranked_columns"
    old = fn(frame, **kwargs); new = fn(frame, feature_label_fallback="fallback", **kwargs)
    assert geometry(old.figure) == geometry(new.figure)
    assert "FALLBACK" in [t.get_text() for t in new.texts]
    if name == "volcano_plot_generic":
        assert [t.xy for t in old.texts] == [t.xy for t in new.texts]
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="feature_label_fallback"):
        fn(frame, feature_label_fallback="absent", **kwargs)


@pytest.mark.parametrize("missing", MISSING)
def test_forest_feature_ids_numerics_and_labels(missing):
    frame = pd.DataFrame({"estimate": [1., 2.], "low": [.5, 1.5], "high": [1.5, 2.5],
                          "label": [missing, "  VALID  "], "fallback": ["FALLBACK", "ignored"]}, index=["a", "b"])
    before = frame.copy(deep=True)
    kwargs = dict(var_df=frame, feature_list=["b", "a"], estimate_col="estimate",
                  ci_low_col="low", ci_high_col="high", feature_label_col="label", show=False)
    old = adtl.forest(**kwargs)
    new = adtl.forest(**kwargs, feature_label_fallback="fallback")
    assert geometry(old[0]) == geometry(new[0])
    pd.testing.assert_frame_equal(old[2].drop(columns="feature_label"), new[2].drop(columns="feature_label"))
    assert "FALLBACK" in new[2].feature_label.tolist()
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="feature_label_fallback"):
        adtl.forest(**kwargs, feature_label_fallback="absent")


@pytest.mark.parametrize("name", ["timeseries_paired_datapoints", "plot_paired_point_anndata"])
@pytest.mark.parametrize("missing", MISSING)
def test_timeseries_anndata_labels(name, missing):
    import anndata
    obs = pd.DataFrame({"TimePoint": pd.Categorical(["Pre", "Post"] * 2),
                        "Treatment_unique": pd.Categorical(["control"] * 4),
                        "Subject_ID": ["one", "one", "two", "two"]},
                       index=["s1", "s2", "s3", "s4"])
    var = pd.DataFrame({"label": [missing], "fallback": ["FALLBACK"]}, index=["a"])
    data = anndata.AnnData(X=np.zeros((4, 1)), obs=obs, var=var)
    data.layers["norm"] = np.array([[1.], [2.], [3.], [4.]])
    original = data.copy()
    fn = getattr(adtl, name)
    with mock.patch.object(plt, "close"):
        fn(data, "a", feature_name_label_col="label", jitter_amount=0)
        old = plt.gcf()
        fn(data, "a", feature_name_label_col="label", jitter_amount=0,
           feature_label_fallback="fallback")
        new = plt.gcf()
    assert geometry(old) == geometry(new)
    assert any("FALLBACK" in t.get_text() for t in new.findobj(Text))
    pd.testing.assert_frame_equal(data.var, original.var)
    pd.testing.assert_frame_equal(data.obs, original.obs)
    np.testing.assert_array_equal(data.layers["norm"], original.layers["norm"])
    with pytest.raises(ValueError, match="feature_label_fallback"):
        fn(data, "a", feature_name_label_col="label", feature_label_fallback="absent")


@pytest.mark.parametrize("name", ["l2fc_pvalue_dotplot_protein_metabolite", "l2fc_pvalue_dotplot_gex"])
@pytest.mark.parametrize("missing", MISSING)
def test_legacy_analyte_labels(name, missing):
    frame = pd.DataFrame({"id": ["a", "b"], "label": [missing, "  VALID  "],
                          "fallback": ["FALLBACK", "ignored"], "p": [.01, .02], "effect": [-1., 2.]})
    before = frame.copy(deep=True)
    fn = getattr(adtl, name)
    kwargs = dict(index_column="id", analyte_label_column="label", comparison_column=None,
                  pval_col="p", l2fc_col="effect")
    fn(frame, ["b", "a"], **kwargs)
    old = plt.gcf()
    fn(frame, ["b", "a"], feature_label_fallback="fallback", **kwargs)
    new = plt.gcf()
    assert geometry(old) == geometry(new)
    # Pandas 3 string conversion preserves nulls, so legacy seaborn drops
    # those rows. Fallback must not change that pre-existing selection.
    if len(old.axes[0].get_yticklabels()) == 2:
        assert "FALLBACK" in [t.get_text() for t in new.axes[0].get_yticklabels()]
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="feature_label_fallback"):
        fn(frame, ["a", "b"], feature_label_fallback="absent", **kwargs)


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_resolver_extension_dtypes_and_duplicate_index(dtype):
    frame = pd.DataFrame({"label": pd.Series([None, "  valid  ", "NaN"], dtype=dtype),
                          "fallback": ["same", "ignored", "same"]})
    frame.index = ["a", "a", "b"]
    before = frame.copy(deep=True)
    assert _resolve_feature_labels(frame, "label", "fallback").tolist() == ["same", "  valid  ", "same"]
    pd.testing.assert_frame_equal(frame, before)


def test_ranked_volcano_tie_selection_uses_original_not_fallback_text():
    frame = pd.DataFrame({"gene_names": ["", "Z", "A"], "fallback": ["ZZZZ", "x", "x"],
                          "log2FoldChange": [-1., 1., 1.5], "pvalue": [.01] * 3})
    for layout in ["inline", "ranked_columns"]:
        kwargs = dict(label_top_features=True, n_top_features=1, label_layout=layout, xlimit=2, ylimit=4)
        old = adtl.volcano_plot_generic(frame, **kwargs)
        new = adtl.volcano_plot_generic(frame, feature_label_fallback="fallback", **kwargs)
        assert geometry(old.figure) == geometry(new.figure)
        positions = lambda ax: [getattr(t, "xy", t.get_position()) for t in ax.texts]
        assert positions(old) == positions(new)


def test_forest_grouped_duplicate_display_labels_keep_feature_rows():
    frame = pd.DataFrame({"id": ["a", "b", "a", "b"], "group": ["x", "x", "y", "y"],
                          "effect": [1., 2., 3., 4.], "low": [0., 1., 2., 3.], "high": [2., 3., 4., 5.],
                          "label": [" ", "", " ", ""], "fallback": ["same"] * 4})
    kwargs = dict(var_df=frame, feature_list=["b", "a"], feature_id_col="id", group_col="group",
                  estimate_col="effect", ci_low_col="low", ci_high_col="high", feature_label_col="label", show=False)
    old = adtl.forest(**kwargs)
    new = adtl.forest(**kwargs, feature_label_fallback="fallback")
    assert geometry(old[0]) == geometry(new[0])
    pd.testing.assert_frame_equal(old[2].drop(columns="feature_label"), new[2].drop(columns="feature_label"))
    assert new[2].feature_label.tolist() == ["same"] * 4


@pytest.mark.parametrize("name", VOLCANOS)
def test_volcano_default_and_both_missing(name):
    frame = pd.DataFrame({"gene_names": [" ", "VALID"], "fallback": [None, "ignored"],
                          "log2FoldChange": [-1., 1.], "pvalue": [.01, .02], "padj": [.01, .02]})
    fn = getattr(adtl, name)
    kwargs = dict(label_top_features=True, n_top_features=2, xlimit=2, ylimit=4)
    old = fn(frame, **kwargs)
    for fallback in [None, "fallback"]:
        new = fn(frame, feature_label_fallback=fallback, **kwargs)
        assert geometry(old.figure) == geometry(new.figure)
        assert [t.get_text() for t in old.texts] == [t.get_text() for t in new.texts]


@pytest.mark.parametrize("name", ["l2fc_pvalue_dotplot_protein_metabolite", "l2fc_pvalue_dotplot_gex"])
def test_legacy_analyte_duplicate_fallback_and_default(name):
    frame = pd.DataFrame({"id": ["a", "b"], "label": [" ", "nan"],
                          "fallback": ["same", "same"], "p": [.01, .02], "effect": [-1., 2.]})
    fn = getattr(adtl, name)
    kwargs = dict(index_column="id", analyte_label_column="label", comparison_column=None,
                  pval_col="p", l2fc_col="effect")
    fn(frame, ["b", "a"], **kwargs); old = plt.gcf()
    fn(frame, ["b", "a"], feature_label_fallback=None, **kwargs); explicit = plt.gcf()
    assert geometry(old) == geometry(explicit)
    fn(frame, ["b", "a"], feature_label_fallback="fallback", **kwargs); new = plt.gcf()
    assert geometry(old) == geometry(new)
    assert [t.get_text() for t in new.axes[0].get_yticklabels()] == ["same", "same"]


@pytest.mark.parametrize("name", COLUMNS)
def test_anndata_var_fallback_source(name):
    import anndata
    fixture = column_fixtures.ColumnPlotRendererTests(); fixture.setUp()
    var = fixture.var_df.copy()
    var["feature_label"] = ["", "VALID"]
    var["fallback"] = ["FROM_VAR", "ignored"]
    data = anndata.AnnData(X=fixture.x_df.to_numpy(), obs=fixture.obs_df.copy(), var=var)
    data.layers["salmon_effective_TPM"] = data.X.copy()
    kwargs = dict(adata=data, feature_list=fixture.features, feature_label_vars_col="feature_label",
                  feature_label_fallback="fallback", figsize=(6, 4))
    if "comparison_order" in inspect.signature(getattr(adtl, name)).parameters:
        kwargs["comparison_order"] = ["control", "drug"]
    before = data.var.copy(deep=True)
    fig = getattr(adtl, name)(**kwargs)[0]
    assert any(t.get_text() == "FROM_VAR" for t in fig.findobj(Text))
    pd.testing.assert_frame_equal(data.var, before)


@pytest.mark.parametrize("name", ["l2fc_pvalue_dotplot_protein_metabolite", "l2fc_pvalue_dotplot_gex"])
def test_legacy_shared_original_category_retains_all_fallback_labels(name):
    frame = pd.DataFrame({"id": ["a", "b"], "label": ["", ""],
                          "fallback": ["Alpha", "Beta"], "p": [.01, .02], "effect": [-1., 2.]})
    fn = getattr(adtl, name)
    kwargs = dict(index_column="id", analyte_label_column="label", comparison_column=None,
                  pval_col="p", l2fc_col="effect")
    fn(frame, ["a", "b"], **kwargs); old = plt.gcf()
    fn(frame, ["a", "b"], feature_label_fallback="fallback", **kwargs); new = plt.gcf()
    assert geometry(old) == geometry(new)
    assert new.axes[0].get_yticklabels()[0].get_text() == "Alpha / Beta"


def test_forest_rowwise_conflict_keeps_existing_validation():
    frame = pd.DataFrame({"id": ["a", "a"], "group": ["x", "y"],
                          "effect": [1., 2.], "low": [0., 1.], "high": [2., 3.],
                          "label": ["Alpha", None], "fallback": ["id-a", "id-a"]})
    kwargs = dict(var_df=frame, feature_list=["a"], feature_id_col="id", group_col="group",
                  estimate_col="effect", ci_low_col="low", ci_high_col="high", feature_label_col="label", show=False)
    assert adtl.forest(**kwargs)[2].feature_label.tolist() == ["Alpha", "Alpha"]
    with pytest.raises(ValueError, match="Feature labels must agree"):
        adtl.forest(**kwargs, feature_label_fallback="fallback")
