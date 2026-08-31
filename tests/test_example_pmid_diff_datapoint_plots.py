from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "example_PMID_33969320" / "config" / "config.yaml"
SCRIPT_PATH = (
    REPO_ROOT
    / "example_PMID_33969320"
    / "scripts"
    / "make_diff_datapoint_plots.py"
)


def test_example_pmid_config_uses_unified_effect_panel_schema():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        plot_cfg = yaml.safe_load(handle)["diff_datapoint_plots_params"]

    assert set(plot_cfg) == {
        "repo_results_dir",
        "datapoints_effect_panels_column_calls_defaults",
        "datapoints_effect_panels_column_calls",
    }
    defaults = plot_cfg["datapoints_effect_panels_column_calls_defaults"]
    runs = plot_cfg["datapoints_effect_panels_column_calls"]
    assert defaults["comparison_col"] == "COVID"
    assert defaults["comparison_col_order"] == ["0", "1"]
    assert defaults["top_n_features"] == 15
    assert defaults["column_title_y"] == 0.9
    assert defaults["figsize"] == [15, 18]
    assert defaults["fig_title"] == (
        "OLINK proteomics: top 15 differential features\n"
        "D0 comparison; independent t-test"
    )
    assert defaults["col_wspace"] == 0.07
    assert defaults["effect_panel_defaults"]["effect_column"] == (
        "l2fc_COVID_over_NOT_D0"
    )
    assert defaults["effect_panel_defaults"]["annotate_xy"] == [0.98, 0.82]
    assert defaults["effect_panel_defaults"]["annotate_fontsize"] == 11
    assert runs["COVID_over_NOT_D0"]["sortby_col"] == (
        "ttest_ind_pvals_COVID_over_NOT_D0"
    )
    assert runs["COVID_over_NOT_D0"]["effect_panels"][0][
        "pvalue_column"
    ] == "ttest_ind_pvals_COVID_over_NOT_D0"
    assert runs["COVID_over_NOT_D0"]["effect_panels"][0][
        "pvalue_label"
    ] == "-log10(RAW independent t-test p-value)"
    assert runs["COVID_over_NOT_D0_FDR"]["sortby_col"] == (
        "ttest_ind_pvals_FDR_COVID_over_NOT_D0"
    )
    assert runs["COVID_over_NOT_D0_FDR"]["effect_panels"][0][
        "pvalue_column"
    ] == "ttest_ind_pvals_FDR_COVID_over_NOT_D0"
    assert runs["COVID_over_NOT_D0_FDR"]["effect_panels"][0][
        "pvalue_label"
    ] == "-log10(FDR-adjusted independent t-test p-value)"
    assert all(run["run"] is True for run in runs.values())

    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert script_source.count("adtl.datapoints_effect_panels_column(") == 1
    for legacy_name in (
        "barh_l2fc_dotplot_column(",
        "barh_dotplot_dotplot_column(",
        "barh_dotplot_dotplot_dotplot_column(",
        "barh_4X_dotplot_column(",
    ):
        assert legacy_name not in script_source


def test_named_runs_skip_disabled_and_preserve_falsey_overrides(
    caplog, monkeypatch, tmp_path
):
    spec = importlib.util.spec_from_file_location(
        "example_pmid_make_diff_datapoint_plots", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PACKAGE_ROOT = tmp_path
    adata = anndata.AnnData(
        X=np.array([[1.0], [2.0]]),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "b"])},
            index=["sample_1", "sample_2"],
        ),
        var=pd.DataFrame(
            {"effect": [1.0], "pvalue": [0.01]},
            index=["feature_1"],
        ),
    )
    read_h5ad = Mock(return_value=adata)
    figure = plt.figure()
    renderer = Mock(return_value=(figure, np.empty((1, 2), dtype=object)))
    monkeypatch.setattr(module.anndata, "read_h5ad", read_h5ad)
    monkeypatch.setattr(
        module.adtl, "datapoints_effect_panels_column", renderer
    )
    caplog.set_level("INFO", logger=module.__name__)

    plot_cfg = {
        "datapoints_effect_panels_column_calls_defaults": {
            "run": True,
            "adata_h5ad_path": "input.h5ad",
            "file_name": "default.png",
            "feature_list": ["feature_1"],
            "comparison_col": "group",
            "comparison_col_order": ["a", "b"],
            "comparison_axis_label": "group",
            "hue_palette_color_list": ["#111111", "#222222"],
            "figsize": [4, 3],
            "include_stripplot": True,
            "point_jitter": 0.2,
            "use_tight_layout": True,
            "effect_panel_defaults": {
                "effect_mode": "pvalue",
                "effect_column": "effect",
                "pvalue_column": "pvalue",
                "effect_reference_value": 1,
                "legend": True,
            },
        },
        "datapoints_effect_panels_column_calls": {
            "disabled": {"run": False},
            "enabled": {
                "file_name": "enabled.png",
                "comparison_axis_label": "",
                "include_stripplot": False,
                "point_jitter": 0.0,
                "use_tight_layout": False,
                "effect_panels": [
                    {
                        "effect_reference_value": 0,
                        "legend": False,
                    }
                ],
            },
        },
    }
    module.run_datapoint_plots(plot_cfg)

    assert (
        "Skipping datapoints_effect_panels_column 'disabled' because run=false"
        in caplog.messages
    )
    read_h5ad.assert_called_once_with(tmp_path / "input.h5ad")
    renderer.assert_called_once()
    call_kwargs = renderer.call_args.kwargs
    assert call_kwargs["comparison_axis_label"] == ""
    assert call_kwargs["include_stripplot"] is False
    assert call_kwargs["point_jitter"] == 0.0
    assert call_kwargs["use_tight_layout"] is False
    assert call_kwargs["effect_panels"][0]["effect_reference_value"] == 0
    assert call_kwargs["effect_panels"][0]["legend"] is False
    assert call_kwargs["file_name"] == str(tmp_path / "enabled.png")
    assert not plt.fignum_exists(figure.number)
