#!/usr/bin/env python3
"""Generate configured differential datapoint and effect-panel plots."""

from __future__ import annotations

import logging
import sys
from collections import ChainMap
from datetime import datetime
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import pandas as pd
import yaml


EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = EXAMPLE_ROOT.parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

import adata_science_tools as adtl


CONFIG_PATH = EXAMPLE_ROOT / "config" / "config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as handle:
    CFG = yaml.safe_load(handle) or {}

PLOT_CFG = CFG["diff_datapoint_plots_params"]
OUTPUT_DIR = PACKAGE_ROOT / Path(PLOT_CFG["repo_results_dir"])
LOG_DIR = OUTPUT_DIR / "logs"
SCRIPT_LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILENAME = f"{Path(__file__).stem}_{datetime.now():%Y%m%d_%H%M%S}.log"
RESULTS_LOG_FILE = LOG_DIR / LOG_FILENAME
SCRIPT_LOG_FILE = SCRIPT_LOG_DIR / LOG_FILENAME
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure the existing results and script-local log destinations."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(RESULTS_LOG_FILE),
            logging.FileHandler(SCRIPT_LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    LOGGER.info("Logging to %s", RESULTS_LOG_FILE)
    LOGGER.info("Logging to %s", SCRIPT_LOG_FILE)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").propagate = True
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)


def run_datapoint_plots(plot_cfg: dict) -> None:
    """Render each enabled named run from the unified plotting config."""
    default_params = plot_cfg.get(
        "datapoints_effect_panels_column_calls_defaults", {}
    )
    plot_runs = plot_cfg.get("datapoints_effect_panels_column_calls", {})

    for plot_key, run_params in plot_runs.items():
        params = dict(ChainMap(run_params, default_params))
        if not params.get("run", False):
            LOGGER.info(
                "Skipping datapoints_effect_panels_column '%s' because run=false",
                plot_key,
            )
            continue

        LOGGER.info("Starting diff datapoint plot for run: %s", plot_key)
        effect_panel_defaults = params.get("effect_panel_defaults", {})
        effect_panels = [
            dict(ChainMap(panel, effect_panel_defaults))
            for panel in params["effect_panels"]
        ]
        primary_effect_column = effect_panels[0]["effect_column"]

        adata_path = PACKAGE_ROOT / Path(params["adata_h5ad_path"])
        LOGGER.info("Loading AnnData from %s", adata_path)
        adata = anndata.read_h5ad(adata_path)
        LOGGER.info("Loaded %s", adata)
        LOGGER.info("AnnData var columns: %s", adata.var.columns.tolist())

        LOGGER.info(
            "Filtering features with non-null values in %s",
            primary_effect_column,
        )
        adata_filtered = adata[:, ~adata.var[primary_effect_column].isna()].copy()
        LOGGER.info("Filtered AnnData var shape: %s", adata_filtered.var.shape)

        feature_list = params.get("feature_list")
        if not feature_list:
            var_mask_col = params.get("var_mask_col")
            if var_mask_col:
                LOGGER.info("Filtering features where %s is true", var_mask_col)
                adata_filtered = adata_filtered[
                    :, adata_filtered.var[var_mask_col]
                ].copy()

            filter_column = params.get("lessthan_filterby_col")
            filter_threshold = params.get("lessthan_filterby_threshold")
            if filter_column and filter_threshold is not None:
                LOGGER.info(
                    "Filtering features where %s <= %s",
                    filter_column,
                    filter_threshold,
                )
                adata_filtered = adata_filtered[
                    :, adata_filtered.var[filter_column] <= filter_threshold
                ].copy()

            for prefix in params.get(
                "list_of_startswith_str_to_filter_features", []
            ):
                LOGGER.info("Removing features starting with %s", prefix)
                adata_filtered = adata_filtered[
                    :, ~adata_filtered.var.index.str.startswith(prefix)
                ].copy()
            for suffix in params.get(
                "list_of_endswith_str_to_filter_features", []
            ):
                LOGGER.info("Removing features ending with %s", suffix)
                adata_filtered = adata_filtered[
                    :, ~adata_filtered.var.index.str.endswith(suffix)
                ].copy()

            sortby_col = params["sortby_col"]
            ascending = params.get("ascending", True)
            top_n_features = params.get("top_n_features", 15)
            LOGGER.info(
                "Selecting top %s features by %s (ascending=%s)",
                top_n_features,
                sortby_col,
                ascending,
            )
            feature_list = (
                adata_filtered.var.sort_values(
                    by=sortby_col,
                    ascending=ascending,
                )
                .head(top_n_features)
                .index.tolist()
            )
        else:
            feature_list = list(feature_list)
        LOGGER.info(
            "Selected %s ordered features: %s", len(feature_list), feature_list
        )

        comparison_col = params["comparison_col"]
        comparison_order = params.get("comparison_col_order")
        if comparison_order is not None:
            LOGGER.info(
                "Ordering %s categories as %s",
                comparison_col,
                comparison_order,
            )
            adata.obs[comparison_col] = pd.Categorical(
                adata.obs[comparison_col],
                categories=comparison_order,
                ordered=True,
            )
        palette_order = (
            comparison_order
            if comparison_order is not None
            else list(pd.unique(adata.obs[comparison_col]))
        )
        palette_colors = params.get("hue_palette_color_list")
        distribution_palette = (
            dict(zip(palette_order, palette_colors))
            if palette_colors is not None
            else None
        )

        file_name = PACKAGE_ROOT / Path(params["file_name"])
        file_name.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving %s to %s", plot_key, file_name)
        figure, _ = adtl.datapoints_effect_panels_column(
            adata=adata,
            layer=params.get("layer"),
            feature_list=feature_list,
            orientation=params.get("orientation", "horizontal"),
            effect_panels=effect_panels,
            comparison_col=comparison_col,
            comparison_order=comparison_order,
            feature_label_vars_col=params.get("feature_label_vars_col"),
            feature_label_char_limit=params.get("feature_label_char_limit"),
            feature_labels_as_ylabels=params.get(
                "feature_labels_as_ylabels", False
            ),
            feature_label_x=params.get("feature_label_x", -0.02),
            feature_label_fontsize=params.get("feature_label_fontsize"),
            remove_group_tick_labels=params.get(
                "remove_group_tick_labels", False
            ),
            comparison_axis_label=params.get("comparison_axis_label"),
            distribution_kind=params.get("distribution_kind", "bar"),
            include_stripplot=params.get("include_stripplot", True),
            distribution_palette=distribution_palette,
            point_color_column=params.get("point_color_column"),
            point_shape_column=params.get("point_shape_column"),
            point_palette=params.get("point_palette"),
            point_markers=params.get("point_markers"),
            point_jitter=params.get("point_jitter"),
            point_size=params.get("point_size"),
            pvalue_cutoff=params.get("pvalue_cutoff", 0.1),
            share_pvalue_scale=params.get("share_pvalue_scale", False),
            effect_reference_value=params.get("effect_reference_value", 0),
            effect_marker_size=params.get("effect_marker_size", 5),
            effect_color=params.get("effect_color", "black"),
            share_distribution_axis=params.get(
                "share_distribution_axis", False
            ),
            distribution_axis_limits=params.get("distribution_axis_limits"),
            share_effect_x=params.get("share_effect_x", False),
            effect_xlim=params.get("effect_xlim"),
            figsize=tuple(params["figsize"]),
            width_ratios=params.get("width_ratios", (3.0, 1.0)),
            fig_title=params.get("fig_title"),
            fig_title_y=params.get("fig_title_y", 0.995),
            fig_title_fontsize=params.get("fig_title_fontsize"),
            distribution_title=params.get("distribution_title"),
            column_title_y=params.get("column_title_y"),
            column_title_fontsize=params.get("column_title_fontsize"),
            distribution_axis_label=params.get(
                "distribution_axis_label", "Expression"
            ),
            effect_axis_label=params.get(
                "effect_axis_label", "log2FoldChange"
            ),
            tick_label_fontsize=params.get("tick_label_fontsize"),
            legend_fontsize=params.get("legend_fontsize"),
            numeric_tick_format=params.get("numeric_tick_format"),
            axis_labels_outer_only=params.get("axis_labels_outer_only", False),
            row_hspace=params.get("row_hspace"),
            col_wspace=params.get("col_wspace"),
            legend=params.get("legend", True),
            distribution_legend=params.get("distribution_legend"),
            distribution_legend_loc=params.get(
                "distribution_legend_loc", "upper center"
            ),
            distribution_legend_bbox_to_anchor=params.get(
                "distribution_legend_bbox_to_anchor"
            ),
            distribution_legend_frameon=params.get(
                "distribution_legend_frameon", False
            ),
            tight_layout_rect=params.get("tight_layout_rect"),
            use_tight_layout=params.get("use_tight_layout", True),
            footer=params.get("footer"),
            savefig=params.get("savefig", True),
            file_name=str(file_name),
        )
        plt.close(figure)


if __name__ == "__main__":
    configure_logging()
    LOGGER.info("Using adata_science_tools from %s", adtl.__file__)
    LOGGER.info("Starting make_diff_datapoint_plots.py")
    run_datapoint_plots(PLOT_CFG)
    LOGGER.info("Finished make_diff_datapoint_plots.py")
