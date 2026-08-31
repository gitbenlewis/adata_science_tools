"""Generate deterministic PNG assets for every exported plotting renderer.

Run from the repository parent so the local package is importable::

    python -m adata_science_tools.example_plotting_gallery.generate_gallery \
        --output-dir /tmp/adata-science-tools-gallery
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from functools import cached_property
import logging
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any
from unittest.mock import patch
import warnings

import matplotlib

matplotlib.use("Agg", force=True)

import anndata as ad
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
LOGGER = logging.getLogger(__name__)

if __package__ in {None, ""}:
    package_parent = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_parent))
    import adata_science_tools as adtl
    from adata_science_tools.example_plotting_gallery.manifest import (
        GalleryCase,
        MANIFEST_BY_NAME,
        RENDERER_MANIFEST,
        RendererSpec,
    )
    from adata_science_tools.example_plotting_gallery.simulated_data import (
        make_composition_frame,
        make_continuous_effect_frames,
        make_independent_group_adata,
        make_longitudinal_frame,
        make_meta_forest_rows,
        make_ols_model_results,
        make_paired_adata,
        make_ranked_inputs,
        make_residual_diagnostic_frame,
        make_survival_frames,
        run_independent_diff_test,
    )
else:
    from .. import (
        fit_smf_ols_models_and_summarize_adata,
        pairwise_spearman_corr_matrix,
    )
    from .. import _plotting as adtl
    from .manifest import (
        GalleryCase,
        MANIFEST_BY_NAME,
        RENDERER_MANIFEST,
        RendererSpec,
    )
    from .simulated_data import (
        make_composition_frame,
        make_continuous_effect_frames,
        make_independent_group_adata,
        make_longitudinal_frame,
        make_meta_forest_rows,
        make_ols_model_results,
        make_paired_adata,
        make_ranked_inputs,
        make_residual_diagnostic_frame,
        make_survival_frames,
        run_independent_diff_test,
    )

if __package__ in {None, ""}:
    fit_smf_ols_models_and_summarize_adata = (
        adtl.fit_smf_ols_models_and_summarize_adata
    )
    pairwise_spearman_corr_matrix = adtl.pairwise_spearman_corr_matrix


FEATURES = ("feature_positive", "feature_negative", "feature_null")
GROUP_COLORS = {"control": "#4477AA", "case": "#CC6677"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class GalleryGenerationError(RuntimeError):
    """A renderer case failed before producing its declared PNG."""


class GalleryInputs:
    """Lazily build deterministic, reusable inputs for selected gallery cases."""

    @cached_property
    def independent(self) -> ad.AnnData:
        return make_independent_group_adata()

    @cached_property
    def diff_bundle(self) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
        return run_independent_diff_test(self.independent)

    @cached_property
    def ols_unadjusted(self) -> pd.DataFrame:
        return make_ols_model_results(self.independent)

    @cached_property
    def ols_adjusted(self) -> pd.DataFrame:
        return fit_smf_ols_models_and_summarize_adata(
            self.independent,
            feature_columns=list(FEATURES),
            predictors=["condition_indicator", "age_centered"],
            model_name="gallery_adjusted",
            add_adata_var_column_key_list=[
                "feature_label",
                "truth_class",
                "expected_direction",
            ],
            save_table=False,
            save_model_spec_yaml=False,
            save_result_to_adata_uns_as_dict=False,
            include_fdr=True,
        )

    @cached_property
    def column_adata(self) -> ad.AnnData:
        _, annotated, results = self.diff_bundle
        result_index = results.reindex(annotated.var_names)
        adjusted = self.ols_adjusted.reindex(annotated.var_names)
        gallery_adata = annotated.copy()
        gallery_adata.var["log2FoldChange"] = result_index["effect"]
        gallery_adata.var["pvalue"] = result_index["pvalue"]
        gallery_adata.var["log2FoldChange_alt"] = result_index["effect"]
        gallery_adata.var["pvalue_alt"] = result_index["mannwhitneyu_pvalue"]
        gallery_adata.var["log2FoldChange_alt2"] = adjusted[
            "gallery_adjusted_Coef_condition_indicator"
        ]
        gallery_adata.var["pvalue_alt2"] = adjusted[
            "gallery_adjusted_P>|t|_condition_indicator"
        ]
        gallery_adata.var["log2FoldChange_alt3"] = adjusted[
            "gallery_adjusted_Coef_age_centered"
        ]
        gallery_adata.var["pvalue_alt3"] = adjusted[
            "gallery_adjusted_P>|t|_age_centered"
        ]
        return gallery_adata

    @cached_property
    def independent_frame(self) -> pd.DataFrame:
        return self.independent.to_df().join(self.independent.obs)

    @cached_property
    def forest_grouped(self) -> pd.DataFrame:
        columns = {
            "unadjusted": (
                self.ols_unadjusted,
                "gallery_ols",
            ),
            "age_adjusted": (
                self.ols_adjusted,
                "gallery_adjusted",
            ),
        }
        frames = []
        for model, (results, prefix) in columns.items():
            selected = results.reindex(FEATURES)
            frames.append(
                pd.DataFrame(
                    {
                        "feature": selected.index,
                        "feature_label": selected["feature_label"].astype(str),
                        "model": model,
                        "estimate": selected[
                            f"{prefix}_Coef_condition_indicator"
                        ],
                        "ci_low": selected[
                            f"{prefix}_CI_low_condition_indicator"
                        ],
                        "ci_high": selected[
                            f"{prefix}_CI_high_condition_indicator"
                        ],
                        "pvalue": selected[
                            f"{prefix}_P>|t|_condition_indicator"
                        ],
                        "n_total": selected[f"{prefix}_nobs"],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    @cached_property
    def forest_log_ratio(self) -> pd.DataFrame:
        source = self.independent[:, list(FEATURES)].copy()
        source.layers["log_abundance"] = np.log(np.asarray(source.X, dtype=float))
        results = fit_smf_ols_models_and_summarize_adata(
            source,
            layer="log_abundance",
            feature_columns=list(FEATURES),
            predictors=["condition_indicator"],
            model_name="gallery_log_ratio",
            add_adata_var_column_key_list=["feature_label"],
            save_table=False,
            save_model_spec_yaml=False,
            save_result_to_adata_uns_as_dict=False,
            include_fdr=True,
        )
        return results

    @cached_property
    def paired(self) -> ad.AnnData:
        paired = make_paired_adata()
        paired.layers["norm"] = np.asarray(paired.X).copy()
        return paired

    @cached_property
    def paired_source_summary(self) -> ad.AnnData:
        source = self.paired
        pre_mask = source.obs["condition"].astype(str).eq("pre")
        post_mask = source.obs["condition"].astype(str).eq("post")
        pre = source[pre_mask, :].to_df()
        post = source[post_mask, :].to_df()
        pre.index = source.obs.loc[pre_mask, "subject_id"].astype(str)
        post.index = source.obs.loc[post_mask, "subject_id"].astype(str)
        subjects = sorted(set(pre.index).intersection(post.index))
        pre = pre.loc[subjects]
        post = post.loc[subjects]
        post_obs = source.obs.loc[post_mask].copy()
        post_obs.index = post_obs["subject_id"].astype(str)
        obs = post_obs.loc[subjects, ["subject_id", "cohort"]].copy()
        summary = ad.AnnData(
            X=post.to_numpy() - pre.to_numpy(),
            obs=obs,
            var=source.var.copy(),
        )
        summary.obsm["pre_values"] = pre.copy()
        summary.obsm["post_values"] = post.copy()
        summary.uns["ref_vs_target_adata"] = {
            "source": "make_paired_adata",
            "operation": "post_minus_pre",
        }
        return summary

    @cached_property
    def longitudinal(self) -> pd.DataFrame:
        return make_longitudinal_frame()

    @cached_property
    def survival(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return make_survival_frames()

    @cached_property
    def continuous(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return make_continuous_effect_frames()

    @cached_property
    def meta_rows(self) -> pd.DataFrame:
        return make_meta_forest_rows()

    @cached_property
    def ranked(self) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
        return make_ranked_inputs()

    @cached_property
    def library_rank_correlation(self) -> pd.DataFrame:
        ranked_lists, _, _ = self.ranked
        return pairwise_spearman_corr_matrix(ranked_lists)

    @cached_property
    def composition(self) -> pd.DataFrame:
        return make_composition_frame()

    @cached_property
    def residuals(self) -> pd.DataFrame:
        return make_residual_diagnostic_frame(self.independent)

    @cached_property
    def pooled_diff_results(self) -> pd.DataFrame:
        frames = []
        for replicate, seed in enumerate(range(1729, 1737), start=1):
            _, _, results = run_independent_diff_test(
                n_per_group=18,
                random_seed=seed,
            )
            frame = results.copy()
            frame["feature"] = [
                f"{feature}_r{replicate:02d}" for feature in frame["feature"]
            ]
            frame["gene_names"] = frame["feature"]
            frame["replicate"] = f"replicate_{replicate:02d}"
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @cached_property
    def legacy_diff_results(self) -> pd.DataFrame:
        results = self.diff_bundle[2].copy()
        legacy = pd.DataFrame(
            {
                "comparison": "case_vs_control",
                "feature_id": results["feature"].astype(str),
                "feature_label": results["feature_label"].astype(str),
                "display_name": results["feature_label"].astype(str),
                "pvalue": results["pvalue"],
                "effect": results["effect"],
            }
        )
        return legacy


def _figure_from_result(result: Any) -> Figure | None:
    """Find a Matplotlib figure in a renderer's heterogeneous return value."""

    if isinstance(result, Figure):
        return result
    if isinstance(result, Axes):
        return result.figure
    result_figure = getattr(result, "fig", None)
    if isinstance(result_figure, Figure):
        return result_figure
    result_figure = getattr(result, "figure", None)
    if isinstance(result_figure, Figure):
        return result_figure
    if isinstance(result, Mapping):
        for value in result.values():
            figure = _figure_from_result(value)
            if figure is not None:
                return figure
    elif isinstance(result, Sequence) and not isinstance(
        result, (str, bytes, bytearray)
    ):
        for value in result:
            figure = _figure_from_result(value)
            if figure is not None:
                return figure
    return None


def _invoke_case(
    spec: RendererSpec,
    case: GalleryCase,
    inputs: GalleryInputs,
    asset_path: Path,
) -> Any:
    """Invoke the exact exported renderer named by one manifest case."""

    renderer = getattr(adtl, spec.name)
    case_key = (spec.name, case.case_id)

    if case_key == ("adata_histograms", "subgroup_kde"):
        return renderer(
            adata=inputs.independent,
            var_names=["feature_positive", "feature_negative"],
            subset_obs_key="condition",
            subset_order=["control", "case"],
            subset_palette=GROUP_COLORS,
            show_all_obs_hist=True,
            bins=14,
            kde=True,
            kde_fill=True,
            kde_fill_alpha=0.12,
            add_zero_line=False,
            x_reference_lines=[
                {
                    "value": 12.0,
                    "label": "Reference abundance",
                    "linestyle": ":",
                    "color": "0.25",
                }
            ],
            title="Simulated feature distributions by condition",
            xlabel="Simulated abundance",
            legend_fontsize=8,
            legend_loc="upper center",
            legend_bbox_to_anchor=(0.5, -0.22),
            ncols=2,
            figsize=(9, 4.5),
            show=False,
        )

    if case_key == ("adata_histograms", "feature_group_collapse"):
        return renderer(
            adata=inputs.independent,
            var_groupby_key="feature_group",
            var_names=["signal", "reference"],
            collapse_mode="aggregate",
            collapse_func="mean",
            subset_obs_key="batch",
            subset_order=["batch_1", "batch_2", "batch_3"],
            show_all_obs_hist=True,
            bins=12,
            kde=True,
            add_zero_line=False,
            add_mean_line=True,
            title="Collapsed distributions for feature groups",
            xlabel="Mean simulated abundance",
            legend_fontsize=8,
            legend_loc="upper center",
            legend_bbox_to_anchor=(0.5, -0.22),
            ncols=2,
            figsize=(9, 4.5),
            show=False,
        )

    if spec.name in {
        "barh_column",
        "barh_l2fc_dotplot_column",
        "barh_dotplot_dotplot_column",
        "barh_dotplot_dotplot_dotplot_column",
        "barh_4X_dotplot_column",
    }:
        column_adata = inputs.column_adata
        common = {
            "adata": column_adata,
            "layer": None,
            "feature_list": list(FEATURES),
            "feature_label_vars_col": "feature_label",
            "comparison_col": "condition",
            "comparison_order": ["control", "case"],
            "feature_label_fontsize": 10,
            "tick_label_fontsize": 9,
            "legend_fontsize": 9,
            "fig_title_fontsize": 14,
            "savefig": False,
        }
        if case_key == ("barh_column", "grouped_expression"):
            common.pop("layer")
            return renderer(
                **common,
                include_stripplot=True,
                figsize=(7.5, 5.5),
                fig_title="Simulated feature abundance by condition",
                barh_subplot_xlabel="Simulated abundance",
                barh_legend_bbox_to_anchor=(0.5, -0.02),
            )
        if case_key == ("barh_l2fc_dotplot_column", "two_panel"):
            return renderer(
                **common,
                figsize=(10, 6.5),
                fig_title="Expression and case-versus-control effect",
                fig_title_y=0.99,
                subfig_title_y=0.96,
                subfig_title_fontsize=12,
                barh_figure_plot_title="Observed abundance",
                barh_subplot_xlabel="Simulated abundance",
                dotplot_figure_plot_title="Welch t-test",
                dotplot_subplot_xlabel="log2(case/control)",
                dotplot_annotate=True,
                sizes=(20, 500),
                dotplot_sharex=True,
                tight_layout_rect_arg=(0, 0.22, 1, 1),
                dotplot_legend_bbox_to_anchor=(0.5, 0.11),
                barh_legend_bbox_to_anchor=(0.5, 0.11),
            )
        if case_key == ("barh_dotplot_dotplot_column", "three_panel"):
            return renderer(
                **common,
                figsize=(13, 6.5),
                fig_title="Expression with parametric and rank-test summaries",
                fig_title_y=0.99,
                subfig_title_y=0.96,
                subfig_title_fontsize=12,
                barh_figure_plot_title="Observed abundance",
                barh_subplot_xlabel="Simulated abundance",
                dotplot_figure_plot_title="Welch t-test",
                dotplot_subplot_xlabel="log2(case/control)",
                dotplot2_figure_plot_title="Mann–Whitney U",
                dotplot2_subplot_xlabel="log2(case/control)",
                sizes=(20, 500),
                dotplot2_sizes=(20, 500),
                dotplot_sharex=True,
                dotplot2_sharex=True,
                tight_layout_rect_arg=(0, 0.22, 1, 1),
                barh_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot2_legend_bbox_to_anchor=(0.5, 0.11),
            )
        if case_key == (
            "barh_dotplot_dotplot_dotplot_column",
            "four_panel",
        ):
            return renderer(
                **common,
                figsize=(16, 6.5),
                fig_title="Expression with unadjusted and adjusted summaries",
                fig_title_y=0.99,
                subfig_title_y=0.96,
                subfig_title_fontsize=12,
                barh_figure_plot_title="Observed abundance",
                barh_subplot_xlabel="Simulated abundance",
                dotplot_figure_plot_title="Welch t-test",
                dotplot_subplot_xlabel="log2(case/control)",
                dotplot2_figure_plot_title="Mann–Whitney U",
                dotplot2_subplot_xlabel="log2(case/control)",
                dotplot3_figure_plot_title="Age-adjusted OLS",
                dotplot3_subplot_xlabel="Case coefficient",
                sizes=(20, 500),
                dotplot2_sizes=(20, 500),
                dotplot3_sizes=(20, 500),
                dotplot_sharex=True,
                dotplot2_sharex=True,
                dotplot3_sharex=True,
                tight_layout_rect_arg=(0, 0.22, 1, 1),
                barh_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot2_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot3_legend_bbox_to_anchor=(0.5, 0.11),
            )
        if case_key == ("barh_4X_dotplot_column", "five_panel"):
            return renderer(
                **common,
                figsize=(19, 8.5),
                fig_title="Expression with four inferential views",
                fig_title_y=0.99,
                subfig_title_y=0.96,
                subfig_title_fontsize=12,
                barh_figure_plot_title="Observed abundance",
                barh_subplot_xlabel="Simulated abundance",
                dotplot_figure_plot_title="Welch t-test",
                dotplot_subplot_xlabel="log2(case/control)",
                dotplot2_figure_plot_title="Mann–Whitney U",
                dotplot2_subplot_xlabel="log2(case/control)",
                dotplot3_figure_plot_title="Age-adjusted OLS",
                dotplot3_subplot_xlabel="Case coefficient",
                dotplot4_figure_plot_title="Age association",
                dotplot4_subplot_xlabel="Age coefficient",
                sizes=(20, 500),
                dotplot2_sizes=(20, 500),
                dotplot3_sizes=(20, 500),
                dotplot4_sizes=(20, 500),
                dotplot_sharex=True,
                dotplot2_sharex=True,
                dotplot3_sharex=True,
                dotplot4_sharex=True,
                use_single_dotplot_colormap=True,
                use_tight_layout=True,
                tight_layout_rect_arg=(0, 0.22, 1, 1),
                barh_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot2_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot3_legend_bbox_to_anchor=(0.5, 0.11),
                dotplot4_legend_bbox_to_anchor=(0.5, 0.11),
            )

    if case_key == ("category_composition", "percent_annotated"):
        return renderer(
            inputs.composition,
            x="group",
            category="category",
            x_order=["control", "treated"],
            category_order=["lymphoid", "myeloid", "stromal"],
            palette={
                "lymphoid": "#4477AA",
                "myeloid": "#EE6677",
                "stromal": "#228833",
            },
            normalize="percent",
            annotate=True,
            title="Simulated cell-category composition",
            xlabel="Sample group",
            ylabel="Composition (%)",
            legend_title="Category",
            legend_kwargs={
                "loc": "center left",
                "bbox_to_anchor": (1.02, 0.5),
            },
            figsize=(7, 5),
            show=False,
        )

    if case_key == ("continuous_effect_plot", "observed_categories"):
        curve, observed = inputs.continuous
        return renderer(
            curve,
            x="x",
            estimate="estimate",
            ci_lower="ci_lower",
            ci_upper="ci_upper",
            observed_df=observed,
            observed_x="x",
            observed_y="value",
            observed_category="category",
            observed_order=["cohort_a", "cohort_b"],
            observed_styles={
                "cohort_a": {
                    "marker": "o",
                    "label": "Cohort A",
                    "facecolor": "#4477AA",
                },
                "cohort_b": {
                    "marker": "s",
                    "filled": False,
                    "label": "Cohort B",
                    "facecolor": "#CC6677",
                },
            },
            xscale="log",
            y_reference_lines=[
                {
                    "value": 1.0,
                    "label": "Reference",
                    "linestyle": "--",
                    "color": "0.3",
                }
            ],
            xlabel="Simulated exposure",
            ylabel="Precomputed effect",
            title="Continuous exposure–response curve",
            annotation="Precomputed curve and confidence interval",
            legend_kwargs={
                "loc": "center left",
                "bbox_to_anchor": (1.02, 0.5),
            },
            figsize=(7.5, 5.5),
            show=False,
        )

    if case_key == (
        "geneset_enrichemnt_ol_ven_M_n_N_x",
        "replacement_smoke",
    ):
        return renderer(
            M_set=[f"feature_{index:02d}" for index in range(1, 13)],
            n_set=["feature_01", "feature_02", "feature_03", "feature_04"],
            N_set=["feature_03", "feature_04", "feature_05", "feature_06"],
            plot_title="Legacy enrichment API",
        )

    if case_key == ("geneset_enrichment_venn", "universe_filtered"):
        return renderer(
            universe=[f"feature_{index:02d}" for index in range(1, 13)],
            geneset=[
                "feature_01",
                "feature_02",
                "feature_03",
                "feature_04",
                "outside_geneset",
            ],
            hits=[
                "feature_03",
                "feature_04",
                "feature_05",
                "feature_06",
                "outside_hits",
            ],
            dataset_label="Selected features",
            geneset_label="Reference pathway",
            plot_title="Synthetic gene-set enrichment",
        )

    if case_key == ("corr_dotplot", "subgroup_marginals"):
        return renderer(
            df=inputs.independent_frame.copy(),
            column_key_x="feature_positive",
            column_key_y="feature_negative",
            hue="condition",
            subset_key="batch",
            palette=GROUP_COLORS,
            subset_palette=["#4477AA", "#EE6677", "#228833"],
            method="pearson",
            show_all_obs_fit=True,
            show_x_marginal_hist=True,
            show_y_marginal_hist=True,
            show_all_obs_x_hist=True,
            show_all_obs_y_hist=True,
            x_marginal_hist_bins=12,
            y_marginal_hist_bins=12,
            dot_size=55,
            title_fontsize=14,
            stats_fontsize=9,
            axis_label_fontsize=11,
            tick_label_fontsize=9,
            legend_fontsize=8,
            fit_legend_bbox_to_anchor=(1.18, 1.0),
            hue_legend_bbox_to_anchor=(1.18, 0.45),
            axes_lines=False,
            xlabel="Known positive-effect feature",
            ylabel="Known negative-effect feature",
            axes_title="Feature correlation with batch-specific fits",
            figsize=(8.5, 6),
            show=False,
        )

    if case_key == ("corr_dotplot", "log1p_identity"):
        return renderer(
            df=inputs.independent_frame.copy(),
            column_key_x="feature_positive",
            column_key_y="feature_negative",
            method="spearman",
            show_identity_line=True,
            identity_limits="data",
            identity_line_style={
                "color": "#CC6677",
                "linestyle": "--",
                "linewidth": 1.2,
            },
            xscale="log1p",
            yscale="log1p",
            axes_lines=False,
            x_reference_lines=[
                {"value": 12.0, "label": "x reference", "linestyle": ":"}
            ],
            y_reference_lines=[
                {"value": 12.0, "label": "y reference", "linestyle": ":"}
            ],
            dot_size=48,
            title_fontsize=14,
            stats_fontsize=9,
            axis_label_fontsize=11,
            tick_label_fontsize=9,
            legend_fontsize=8,
            xlabel="Positive-effect feature (log1p scale)",
            ylabel="Negative-effect feature (log1p scale)",
            axes_title="Synchronized nonlinear axes",
            figsize=(7, 5.5),
            show=False,
        )

    if case_key == ("corr_dotplot_dev", "replacement_smoke"):
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            return renderer(
                df=inputs.independent_frame.copy(),
                column_key_x="feature_positive",
                column_key_y="feature_negative",
                hue="condition",
                palette=GROUP_COLORS,
                dot_size=45,
                title_fontsize=12,
                stats_fontsize=8,
                axis_label_fontsize=10,
                tick_label_fontsize=9,
                legend_fontsize=8,
                axes_title="Compatibility wrapper; use corr_dotplot",
                figsize=(7, 5),
                show=False,
            )

    if case_key == ("datapoints", "grouped_markers"):
        return renderer(
            adata=inputs.independent,
            var_names=["feature_positive", "feature_negative"],
            x_by_obs_key="condition",
            x_order=["control", "case"],
            subset_obs_key="condition",
            subset_order=["control", "case"],
            subset_palette=GROUP_COLORS,
            marker_by_obs_key="batch",
            marker_order=["batch_1", "batch_2", "batch_3"],
            marker_styles={
                "batch_1": {"marker": "o", "label": "Batch 1"},
                "batch_2": {
                    "marker": "s",
                    "filled": False,
                    "label": "Batch 2",
                },
                "batch_3": {"marker": "^", "label": "Batch 3"},
            },
            group_annotations=[
                {
                    "metric": "mean",
                    "position": "axes_top",
                    "format": "mean={value:.1f}",
                }
            ],
            legend_metrics=("count", "mean"),
            legend_scope="figure",
            legend_loc="center left",
            legend_bbox_to_anchor=(1.01, 0.5),
            y_reference_lines=[
                {"value": 12.0, "label": "Reference", "linestyle": ":"}
            ],
            point_size=42,
            jitter_amount=0.14,
            random_seed=2026,
            title="Simulated feature values by condition",
            ylabel="Simulated abundance",
            ncols=2,
            figsize=(10, 4.5),
            show=False,
        )

    if case_key == ("datapoints", "feature_group_collapse"):
        return renderer(
            adata=inputs.independent,
            var_groupby_key="feature_group",
            var_names=["signal", "reference"],
            collapse_mode="aggregate",
            collapse_func="mean",
            x_by_obs_key="condition",
            x_order=["control", "case"],
            subset_obs_key="condition",
            subset_order=["control", "case"],
            subset_palette=GROUP_COLORS,
            violinplot=True,
            boxplot=False,
            point_size=38,
            jitter_amount=0.12,
            random_seed=2026,
            title="Collapsed feature-group values",
            ylabel="Mean simulated abundance",
            legend_scope="figure",
            legend_loc="center left",
            legend_bbox_to_anchor=(1.01, 0.5),
            ncols=2,
            figsize=(9, 4.5),
            show=False,
        )

    if case_key == ("forest", "grouped_estimates"):
        return renderer(
            var_df=inputs.forest_grouped,
            feature_list=list(FEATURES),
            feature_id_col="feature",
            feature_label_col="feature_label",
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col="pvalue",
            total_observations_col="n_total",
            group_col="model",
            group_order=["unadjusted", "age_adjusted"],
            group_labels={
                "unadjusted": "Unadjusted",
                "age_adjusted": "Age adjusted",
            },
            effect_type="coefficient",
            effect_label="Case coefficient",
            pvalue_color_mode="continuous",
            group_palette={
                "unadjusted": "#4477AA",
                "age_adjusted": "#CC6677",
            },
            ci_clip="arrows",
            xlims=(-4.5, 4.5),
            x_reference_lines=[
                {"value": 0.0, "label": "No group effect", "linestyle": "--"}
            ],
            xlabel="Case-versus-control OLS coefficient",
            title="Unadjusted and age-adjusted OLS estimates",
            figsize=(10, 5.5),
            show=False,
        )

    if case_key == ("forest", "ratio_scale"):
        return renderer(
            var_df=inputs.forest_log_ratio,
            feature_list=list(FEATURES),
            feature_label_col="feature_label",
            estimate_col="gallery_log_ratio_Coef_condition_indicator",
            ci_low_col="gallery_log_ratio_CI_low_condition_indicator",
            ci_high_col="gallery_log_ratio_CI_high_condition_indicator",
            pvalue_col="gallery_log_ratio_P>|t|_condition_indicator",
            total_observations_col="gallery_log_ratio_nobs",
            effect_type="log_ratio",
            effect_label="Case/control ratio",
            annotate=True,
            show_pvalue_legend=False,
            show_size_legend=False,
            xlims=(0.65, 1.5),
            ci_clip="arrows",
            xlabel="Case/control geometric-mean ratio",
            title="OLS estimates on log abundance",
            figsize=(11.5, 5.5),
            show=False,
        )

    if case_key == ("kaplan_meier_plot", "grouped_risk_censor"):
        curve, risk, censor = inputs.survival
        return renderer(
            curve,
            risk,
            censor_df=censor,
            group_order=["standard", "intensive"],
            palette={"standard": "#4477AA", "intensive": "#CC6677"},
            legend_labels={
                "standard": "Standard care",
                "intensive": "Intensive care",
            },
            title="Precomputed survival curves",
            xlabel="Follow-up time",
            ylabel="Survival probability",
            legend_title="Treatment",
            figsize=(8, 6.5),
            show=False,
        )

    if case_key == ("l2fc_dotplot_column", "multi_feature"):
        return renderer(
            adata=inputs.column_adata,
            feature_list=list(FEATURES),
            feature_label_vars_col="feature_label",
            figsize=(7.5, 6.5),
            fig_title="Case-versus-control differential effects",
            fig_title_y=0.99,
            subfig_title_fontsize=13,
            feature_label_fontsize=10,
            tick_label_fontsize=9,
            legend_fontsize=9,
            sizes=(20, 700),
            dotplot_figure_plot_title="Welch t-test",
            dotplot_subplot_xlabel="log2(case/control)",
            dotplot_set_xaxis_lims=(-0.5, 0.5),
            dotplot_annotate=True,
            dotplot_legend_bbox_to_anchor=(0.5, 0.07),
            tight_layout_rect_arg=[0, 0.20, 1, 1],
            savefig=False,
        )

    if case_key == ("datapoints_effect_panels_column", "horizontal_pvalue"):
        return renderer(
            adata=inputs.column_adata,
            feature_list=list(FEATURES),
            feature_label_vars_col="feature_label",
            orientation="horizontal",
            effect_mode="pvalue",
            comparison_col="condition",
            comparison_order=["control", "case"],
            distribution_kind="violin",
            include_stripplot=True,
            distribution_palette=GROUP_COLORS,
            point_color_column="batch",
            point_palette={
                "batch_1": "#0072B2",
                "batch_2": "#E69F00",
                "batch_3": "#009E73",
            },
            figsize=(10, 6.5),
            fig_title="Horizontal distributions with p-value effects",
            distribution_axis_label="Simulated abundance",
            effect_axis_label="log2(case/control)",
            tight_layout_rect=(0, 0.12, 1, 0.91),
        )

    replacement_case_settings = {
        "horizontal_one_effect": (
            1,
            (10, 6.5),
            "Expression and case-versus-control effect",
            9,
        ),
        "horizontal_two_effects": (
            2,
            (15, 6.5),
            "Expression with parametric and rank-test summaries",
            8,
        ),
        "horizontal_three_effects": (
            3,
            (19, 6.5),
            "Expression with unadjusted and adjusted summaries",
            8,
        ),
        "horizontal_four_effects": (
            4,
            (19, 8.5),
            "Expression with four inferential views",
            9,
        ),
    }
    if (
            spec.name == "datapoints_effect_panels_column"
            and case.case_id in replacement_case_settings
    ):
        panel_count, figsize, fig_title, legend_fontsize = (
            replacement_case_settings[case.case_id]
        )
        effect_panels = [
            {
                "title": "Welch t-test",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "effect_axis_label": "log2(case/control)",
                "pvalue_label": "-log10(p-value)",
                "legend_bins": 4,
                "legend_bbox_to_anchor": (0.5, -0.04),
            },
            {
                "title": "Mann–Whitney U",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt",
                "pvalue_column": "pvalue_alt",
                "effect_axis_label": "log2(case/control)",
                "pvalue_label": "-log10(p-value)",
                "legend_bins": 4,
                "legend_bbox_to_anchor": (0.5, -0.04),
            },
            {
                "title": "Age-adjusted OLS",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt2",
                "pvalue_column": "pvalue_alt2",
                "effect_axis_label": "Case coefficient",
                "pvalue_label": "-log10(p-value)",
                "legend_bins": 4,
                "legend_bbox_to_anchor": (0.5, -0.04),
            },
            {
                "title": "Age association",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt3",
                "pvalue_column": "pvalue_alt3",
                "effect_axis_label": "Age coefficient",
                "pvalue_label": "-log10(p-value)",
                "legend_bins": 4,
                "legend_bbox_to_anchor": (0.5, -0.04),
            },
        ][:panel_count]
        if panel_count == 1:
            effect_panels[0].update({
                "annotate": True,
                "annotate_labels": ("l2fc: ", "p:"),
                "annotate_fontsize": 9,
            })
        elif panel_count in {2, 3}:
            legend_x_positions = np.linspace(0.15, 0.85, panel_count)
            for panel, legend_x in zip(effect_panels, legend_x_positions):
                panel["legend_bbox_to_anchor"] = (float(legend_x), -0.04)
        return renderer(
            adata=inputs.column_adata,
            feature_list=list(FEATURES),
            feature_label_vars_col="feature_label",
            orientation="horizontal",
            comparison_col="condition",
            comparison_order=["control", "case"],
            distribution_kind="bar",
            include_stripplot=True,
            distribution_palette={"control": "#1f77b4", "case": "#ff7f0e"},
            effect_panels=effect_panels,
            share_pvalue_scale=panel_count == 4,
            share_effect_x=True,
            distribution_legend=True,
            feature_label_char_limit=25,
            feature_labels_as_ylabels=True,
            feature_label_x=-0.02,
            feature_label_fontsize=10,
            remove_group_tick_labels=True,
            comparison_axis_label="",
            axis_labels_outer_only=True,
            figsize=figsize,
            width_ratios=(1.5, 1.0),
            fig_title=fig_title,
            fig_title_y=0.99,
            fig_title_fontsize=14,
            distribution_title="Observed abundance",
            column_title_fontsize=12,
            distribution_axis_label="Simulated abundance",
            tick_label_fontsize=9,
            legend_fontsize=legend_fontsize,
            numeric_tick_format="{x:g}",
            use_tight_layout=True,
            distribution_legend_loc="lower center",
            distribution_legend_bbox_to_anchor=(0.5, -0.04),
            distribution_legend_frameon=True,
            tight_layout_rect=(0, 0.12, 1, 0.91),
        )

    if case_key == ("datapoints_effect_panels_column", "vertical_interval"):
        fixture_dir = REPO_ROOT / "example_plotting_gallery" / "data"
        expression_df = pd.read_csv(fixture_dir / "synthetic_expression.csv")
        effects_df = pd.read_csv(fixture_dir / "synthetic_effects.csv")
        obs_df = (
            expression_df.drop_duplicates("sample_id")
            .set_index("sample_id")[["response_group", "subtype", "cohort"]]
        )
        x_df = expression_df.pivot(
            index="sample_id",
            columns="feature",
            values="gtpm",
        ).reindex(obs_df.index)
        return renderer(
            x_df=x_df,
            obs_df=obs_df,
            var_df=effects_df.set_index("feature"),
            feature_list=["GENE_A", "GENE_B", "GENE_C"],
            orientation="vertical",
            effect_mode="interval",
            comparison_col="response_group",
            comparison_order=["NonResponder", "Responder"],
            distribution_kind="box",
            include_stripplot=True,
            distribution_palette={
                "NonResponder": "#eeeeee",
                "Responder": "#eeeeee",
            },
            point_color_column="subtype",
            point_shape_column="cohort",
            point_palette={
                "Subtype A": "#0072B2",
                "Subtype B": "#E69F00",
                "Subtype C": "#009E73",
            },
            point_markers={"Cohort 1": "o", "Cohort 2": "s"},
            effect_column="adjusted_log2fc",
            ci_low_column="ci_low",
            ci_high_column="ci_high",
            effect_marker_size=5,
            effect_color="black",
            figsize=(12, 7.5),
            fig_title=(
                "SYNTHETIC EXAMPLE: response-associated expression panel"
            ),
            fig_title_y=0.99,
            distribution_axis_label="Synthetic gTPM",
            comparison_axis_label="",
            effect_axis_label="Adjusted log2FC\nResponder / NonResponder",
            distribution_legend_loc="upper center",
            distribution_legend_bbox_to_anchor=(0.5, 0.99),
            distribution_legend_frameon=False,
            footer=(
                "All values, identifiers, groups, and effect estimates are "
                "synthetic; intervals are supplied independently of the "
                "expression table."
            ),
            tight_layout_rect=(0, 0.03, 1, 0.91),
        )

    if case_key == ("vbar_l2fc_dotplot_column", "synthetic_response_panel"):
        fixture_dir = REPO_ROOT / "example_plotting_gallery" / "data"
        expression_df = pd.read_csv(fixture_dir / "synthetic_expression.csv")
        effects_df = pd.read_csv(fixture_dir / "synthetic_effects.csv")
        return renderer(
            expression_df=expression_df,
            effects_df=effects_df,
            feature_list=["GENE_A", "GENE_B", "GENE_C"],
            comparison_order=["NonResponder", "Responder"],
            point_color_column="subtype",
            point_shape_column="cohort",
            distribution_kind="box",
            point_palette={
                "Subtype A": "#0072B2",
                "Subtype B": "#E69F00",
                "Subtype C": "#009E73",
            },
            point_markers={"Cohort 1": "o", "Cohort 2": "s"},
            figsize=(12, 7.5),
            fig_title="SYNTHETIC EXAMPLE: response-associated expression panel",
            fig_title_y=0.99,
            value_axis_label="Synthetic gTPM",
            effect_axis_label="Adjusted log2FC\nResponder / NonResponder",
            footer=(
                "All values, identifiers, groups, and effect estimates are synthetic; "
                "intervals are supplied independently of the expression table."
            ),
            legend_bbox_to_anchor=(0.5, 0.99),
            tight_layout_rect_arg=[0, 0.03, 1, 0.91],
            savefig=False,
        )

    if case_key == ("l2fc_dotplot_single", "single_axis"):
        fig, axes = renderer(
            adata=inputs.column_adata,
            feature_list=list(FEATURES),
            feature_label_vars_col="feature_label",
            figsize=(7.5, 4.5),
            fig_title="Differential effect overview",
            feature_label_fontsize=10,
            tick_label_fontsize=9,
            legend_fontsize=9,
            sizes=(20, 700),
            dotplot_subplot_xlabel="log2(case/control)",
            dotplot_set_xaxis_lims=(-0.5, 0.5),
            dotplot_annotate=False,
            dotplot_legend_bbox_to_anchor=(0.5, 0.05),
            tight_layout_rect_arg=(0, 0.23, 1, 1),
        )
        axes.get_legend().set_bbox_to_anchor(
            (0.5, 0.05),
            transform=fig.transFigure,
        )
        return fig, axes

    if spec.name in {
        "l2fc_pvalue_dotplot_gex",
        "l2fc_pvalue_dotplot_protein_metabolite",
    } and case.case_id == "replacement_smoke":
        legacy = inputs.legacy_diff_results
        return renderer(
            legacy,
            legacy["feature_id"].tolist(),
            index_column="feature_id",
            analyte_label_column="feature_label",
            analyte_label="display_name",
            comparison_column="comparison",
            comparison="case_vs_control",
            pval_col="pvalue",
            l2fc_col="effect",
            pval_label="p-value",
            x_axis_label="log2(case/control)",
            pvalue_cutoff=0.1,
            sizes=(30, 500),
            figsize=(7, 4.5),
            bbox_to_anchor=(0.5, -0.48),
            plot_title="Legacy effect dotplot; use l2fc_dotplot_single",
            savefig=True,
            file_name=str(asset_path),
        )

    if case_key == ("longitudinal_trajectories", "markers_and_gaps"):
        return renderer(
            inputs.longitudinal,
            x="visit",
            y="value",
            display_y="display_value",
            subject="subject",
            x_order=["baseline", "week_4", "week_12"],
            line_eligible="eligible",
            connect="adjacent",
            line_color_by="group",
            point_color_by="group",
            color_order=["control", "intervention"],
            palette={"control": "#4477AA", "intervention": "#CC6677"},
            marker_by="response_class",
            marker_order=["stable", "responder"],
            marker_styles={
                "stable": {"marker": "o", "label": "Stable"},
                "responder": {
                    "marker": "^",
                    "filled": False,
                    "label": "Responder",
                },
            },
            x_jitter=0.04,
            random_seed=2026,
            y_reference_lines=[
                {"value": 10.0, "label": "Clinical guide", "linestyle": ":"}
            ],
            xlabel="Visit",
            ylabel="Simulated outcome",
            title="Subject trajectories with an intentional visit gap",
            color_legend_kwargs={
                "bbox_to_anchor": (1.0, 1.0),
                "borderaxespad": 0.0,
                "fontsize": 9,
            },
            marker_legend_kwargs={
                "bbox_to_anchor": (1.0, 0.55),
                "borderaxespad": 0.0,
                "fontsize": 9,
            },
            figsize=(8.5, 5.5),
            show=False,
        )

    if case_key == ("meta_forest", "study_summary_prediction"):
        return renderer(
            inputs.meta_rows,
            label_col="study",
            estimate_col="effect",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            prediction_low_col="prediction_low",
            prediction_high_col="prediction_high",
            weight_col="weight",
            sample_size_col="n",
            study_size_by="weight",
            table_columns={
                "Year": "year",
                "N": "n",
                "Heterogeneity": "heterogeneity",
            },
            effect_scale="additive",
            x_reference_lines=[
                {"value": 0.0, "label": "No effect", "linestyle": "--"}
            ],
            xlabel="Precomputed standardized effect",
            title="Precomputed subgroup meta-analysis",
            figsize=(11, 6.5),
            show=False,
        )

    if case_key == ("paired_datapoints", "paired_groups"):
        return renderer(
            adata=inputs.paired,
            var_groupby_key="feature_group",
            var_names=["responsive", "reference"],
            collapse_mode="aggregate",
            collapse_func="mean",
            groupby_key="condition",
            groupby_key_ref_value="pre",
            groupby_key_target_value="post",
            pair_by_key="subject_id",
            subset_obs_key="cohort",
            subset_order=["cohort_a", "cohort_b"],
            subset_palette=["#4477AA", "#CC6677"],
            legend=True,
            legend_scope="figure",
            legend_loc="center left",
            legend_bbox_to_anchor=(1.01, 0.5),
            random_seed=2026,
            point_size=48,
            jitter_amount=0.12,
            title="Paired pre/post changes by cohort",
            xlabel="Condition",
            ylabel="Mean simulated abundance",
            ncols=2,
            figsize=(9, 4.5),
            show=False,
        )

    if case_key == ("paired_datapoints", "slope_colored_lines"):
        return renderer(
            adata=inputs.paired,
            var_names=["paired_increase", "paired_decrease", "paired_null"],
            layer="linear_mean",
            groupby_key="condition",
            groupby_key_ref_value="pre",
            groupby_key_target_value="post",
            pair_by_key="subject_id",
            line_color_by_slope=True,
            slope_color_threshold=0.05,
            line_alpha=0.75,
            line_width=1.4,
            jitter_amount=0.08,
            random_seed=2026,
            point_size=36,
            boxplot=False,
            title="Paired changes colored by direction",
            xlabel="Condition",
            ylabel="Simulated abundance",
            subplot_title_var_col="feature_label",
            ncols=3,
            figsize=(10.5, 3.8),
            show=False,
        )

    if case_key in {
        ("paired_datapoints", "difference_axis"),
        ("paired_datapoints", "log2fc_axis"),
    }:
        paired = inputs.paired
        post_mask = paired.obs["condition"].astype(str).eq("post").to_numpy()
        subject_ids = pd.Index(pd.unique(paired.obs["subject_id"].astype(str)))
        # Replace the fixed simulation effects with deterministic subject-specific changes.
        subject_changes = pd.DataFrame(
            {
                "positive_slopes": np.linspace(0.75, 2.75, len(subject_ids)),
                "negative_slopes": -np.linspace(0.75, 2.5, len(subject_ids)),
                "approximately_flat": np.resize(
                    np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2]),
                    len(subject_ids),
                ),
            },
            index=subject_ids,
        )
        row_changes = subject_changes.loc[
            paired.obs["subject_id"].astype(str)
        ].to_numpy()
        varied_values = np.asarray(paired.layers["linear_mean"], dtype=float).copy()
        varied_values[post_mask] += (
            row_changes[post_mask]
            - paired.var["true_paired_effect"].to_numpy(dtype=float)
        )
        # Duplicate the direction variables so stack mode can show them separately and together.
        individual_variables = list(subject_changes.columns)
        combined_variables = [
            f"all_{variable}" for variable in individual_variables
        ]
        gallery_df = pd.DataFrame(
            np.concatenate([varied_values, varied_values], axis=1),
            index=paired.obs_names,
            columns=individual_variables + combined_variables,
        ).join(paired.obs[["condition", "subject_id", "cohort"]])
        gallery_var_df = pd.DataFrame(
            {
                "gallery_panel": [
                    "Positive slopes",
                    "Negative slopes",
                    "Approximately flat",
                    "All directions",
                    "All directions",
                    "All directions",
                ],
            },
            index=individual_variables + combined_variables,
        )
        if case_key[1] == "difference_axis":
            derived_axis_options = {
                "paired_difference_label": "post - pre",
                "paired_difference_ylabel": "Paired difference",
                "paired_difference_ylims": (-3.0, 3.0),
                "boxplot": True,
                "title": "Varied paired slopes and signed differences",
                "xlabel": "Condition and derived change",
            }
        else:
            derived_axis_options = {
                "paired_difference_mode": "log2fc",
                "paired_difference_label": "log2(post / pre)",
                "paired_difference_ylabel": "Paired log2FC (post / pre)",
                "paired_difference_ylims": (-0.4, 0.4),
                "boxplot": False,
                "violinplot": True,
                "title": "Varied paired slopes and log2 fold changes",
                "xlabel": "Condition and derived log2 fold change",
            }
        return renderer(
            df=gallery_df,
            var_df=gallery_var_df,
            var_names=[
                "Positive slopes",
                "Negative slopes",
                "Approximately flat",
                "All directions",
            ],
            var_groupby_key="gallery_panel",
            collapse_mode="stack",
            groupby_key="condition",
            groupby_key_ref_value="pre",
            groupby_key_target_value="post",
            pair_by_key="subject_id",
            subset_obs_key="cohort",
            subset_order=["cohort_a", "cohort_b"],
            subset_palette=["#4477AA", "#CC6677"],
            show_paired_difference=True,
            line_color_by_slope=True,
            slope_color_threshold=0.05,
            line_alpha=0.5,
            line_width=1.1,
            jitter_amount=0.08,
            random_seed=2026,
            point_size=26,
            point_alpha=0.7,
            legend=True,
            legend_scope="figure",
            legend_loc="center left",
            legend_bbox_to_anchor=(1.01, 0.5),
            ylabel="Simulated abundance",
            ncols=4,
            figsize=(16, 4.2),
            show=False,
            **derived_axis_options,
        )

    if case_key in {
        ("paired_datapoints", "difference_summary_legend"),
        ("paired_datapoints", "log2fc_summary_legend"),
    }:
        if case_key[1] == "difference_summary_legend":
            summary_options = {
                "paired_difference_mode": "difference",
                "paired_difference_label": "post - pre",
                "paired_difference_ylabel": "Paired difference",
                "paired_difference_ylims": (-3.0, 3.0),
                "title": "Baseline, post, and raw-difference summaries",
                "xlabel": "Condition and raw paired difference",
            }
        else:
            summary_options = {
                "paired_difference_mode": "log2fc",
                "paired_difference_label": "log2(post / pre)",
                "paired_difference_ylabel": "Paired log2FC (post / pre)",
                "paired_difference_ylims": (-0.45, 0.45),
                "title": "Baseline, post, and log2FC summaries",
                "xlabel": "Condition and paired log2 fold change",
            }
        return renderer(
            adata=inputs.paired,
            var_names=["paired_decrease"],
            groupby_key="condition",
            groupby_key_ref_value="pre",
            groupby_key_target_value="post",
            pair_by_key="subject_id",
            subset_obs_key="cohort",
            subset_order=["cohort_a", "cohort_b"],
            subset_palette=["#4477AA", "#CC6677"],
            show_paired_difference=True,
            legend=True,
            legend_metrics=("count", "mean", "sem"),
            legend_metric_formats={
                "count": "n={value:d}",
                "mean": "mean={value:.2f}",
                "sem": "SEM={value:.2f}",
            },
            legend_scope="figure",
            legend_loc="center left",
            legend_bbox_to_anchor=(1.01, 0.5),
            random_seed=2026,
            point_size=42,
            jitter_amount=0.08,
            subplot_title_var_col="feature_label",
            ylabel="Simulated abundance",
            ncols=1,
            figsize=(7.5, 4.5),
            show=False,
            **summary_options,
        )

    if case_key == ("paired_datapoints", "precomputed_pair_values"):
        return renderer(
            adata=inputs.paired_source_summary,
            var_names=["paired_increase", "paired_decrease"],
            ref_values_obsm_key="pre_values",
            target_values_obsm_key="post_values",
            groupby_key_ref_value="pre",
            groupby_key_target_value="post",
            pair_by_key="subject_id",
            subset_obs_key="cohort",
            subset_order=["cohort_a", "cohort_b"],
            subset_palette=["#4477AA", "#CC6677"],
            legend=True,
            legend_scope="figure",
            legend_loc="center left",
            legend_bbox_to_anchor=(1.01, 0.5),
            random_seed=2026,
            point_size=48,
            jitter_amount=0.12,
            title="Preserved pre/post source matrices",
            xlabel="Condition",
            ylabel="Simulated abundance",
            subplot_title_var_col="feature_label",
            ncols=2,
            figsize=(9, 4.5),
            show=False,
        )

    if case_key == (
        "plot_column_of_bar_h_2groups_GEX_adata",
        "replacement_smoke",
    ):
        column_adata = inputs.column_adata
        return renderer(
            adata=None,
            layer=None,
            x_df=column_adata.to_df(),
            var_df=column_adata.var,
            obs_df=column_adata.obs,
            feature_list=list(FEATURES),
            feature_label_vars_col="feature_label",
            comparison_col="condition",
            comparison_order=["control", "case"],
            figsize=(7.5, 5.5),
            fig_title="Legacy API; use barh_column",
            fig_title_fontsize=14,
            feature_label_fontsize=10,
            tick_label_fontsize=9,
            legend_fontsize=9,
            subplot_xlabel="Simulated abundance",
            barh_legend_bbox_to_anchor=(0.5, -0.02),
            savefig=False,
        )

    if case_key == (
        "plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata",
        "replacement_smoke",
    ):
        column_adata = inputs.column_adata
        return renderer(
            adata=None,
            layer=None,
            x_df=column_adata.to_df(),
            var_df=column_adata.var,
            obs_df=column_adata.obs,
            feature_list=list(FEATURES),
            feature_label_vars_col="feature_label",
            comparison_col="condition",
            comparison_order=["control", "case"],
            figsize=(10, 6),
            fig_title="Legacy API; use barh_l2fc_dotplot_column",
            fig_title_y=0.99,
            subfig_title_y=0.96,
            fig_title_fontsize=14,
            subfig_title_fontsize=12,
            feature_label_fontsize=10,
            tick_label_fontsize=9,
            legend_fontsize=9,
            barh_subplot_xlabel="Simulated abundance",
            dotplot_subplot_xlabel="log2(case/control)",
            sizes=(20, 500),
            dotplot_sharex=True,
            tight_layout_rect_arg=(0, 0.22, 1, 1),
            barh_legend_bbox_to_anchor=(0.5, 0.1),
            dotplot_legend_bbox_to_anchor=(0.5, 0.1),
            savefig=False,
        )

    if case_key == ("plot_columns", "multi_metric"):
        return renderer(
            df=inputs.independent_frame.copy(),
            columns2plot=["age", "feature_positive", "feature_negative"],
            columns2plot_titles=[
                "Age",
                "Positive effect",
                "Negative effect",
            ],
            y_groupby="condition",
            figsize=(15, 5),
            sharex=False,
            sharey=False,
            swarm_size=5,
            suptitle_fontsize=18,
            subplot_title_fontsize=14,
            y_label_fontsize=11,
            y_tick_label_fontsize=9,
        )

    if spec.name == "plot_heatmap":
        if case.case_id == "clustered":
            return renderer(
                inputs.library_rank_correlation,
                title="Clustered pairwise rank correlation",
                cluster=True,
                annot=True,
                cmap="vlag",
                figsize=(6, 5.5),
                show=False,
            )
        if case.case_id == "fixed_order":
            return renderer(
                inputs.library_rank_correlation,
                title="Pairwise rank correlation in source order",
                cluster=False,
                annot=True,
                cmap="vlag",
                figsize=(6, 5),
                show=False,
            )

    if case_key == ("plot_paired_point_anndata", "replacement_smoke"):
        np.random.seed(20260727)
        return renderer(
            inputs.paired,
            feature_name="paired_increase",
            x_col="condition",
            feature_name_label_col="feature_label",
            layer="norm",
            Hue="cohort",
            subplotby="cohort",
            analyte_label="Simulated abundance",
            subject_col="subject_id",
            connect_lines=True,
            jitter_amount=0.08,
            legend=True,
            figsize=(8, 6),
            savefig=True,
            file_name=str(asset_path),
        )

    if case_key == ("plot_rank_heatmap", "rank_hexbin"):
        ranked_lists, _, _ = inputs.ranked
        return renderer(
            ranked_lists["method_a"],
            ranked_lists["method_c"],
            extra_title="Method A versus Method C",
            x_label="Rank in method A",
            y_label="Rank in method C",
            gridsize=3,
            figsize=(7, 5.5),
            show_diagonal=True,
        )

    if case_key == ("plot_rank_scatter", "rank_agreement"):
        ranked_lists, _, _ = inputs.ranked
        return renderer(
            ranked_lists["method_a"],
            ranked_lists["method_b"],
            extra_title="Method A versus Method B",
            x_label="Rank in method A",
            y_label="Rank in method B",
            figsize=(7, 5.5),
            show_diagonal=True,
        )

    if case_key == ("plot_rank_scatter_density", "rank_density"):
        ranked_lists, _, _ = inputs.ranked
        return renderer(
            ranked_lists["method_a"],
            ranked_lists["method_c"],
            extra_title="Method A versus Method C",
            x_label="Rank in method A",
            y_label="Rank in method C",
            dot_size=55,
            cmap="viridis",
            figsize=(7, 5.5),
            show_diagonal=True,
        )

    if case_key == ("qqplot", "differential_pvalues"):
        return renderer(
            inputs.pooled_diff_results,
            pvalue_column="pvalue",
            source="df",
            title="QQ plot of library-derived differential-test p-values",
            pvalue_column_plot_label="Welch t-test p-value",
            figsize=(6, 5.5),
            show=False,
            return_points=True,
            annotate_lambda=True,
        )

    if case_key == ("qqplot_pvalues", "replacement_smoke"):
        return renderer(
            inputs.pooled_diff_results,
            pvalue_column="pvalue",
            source="df",
            title="Legacy API; use qqplot",
            pvalue_column_plot_label="Welch t-test p-value",
            figsize=(6, 5),
            show=False,
            return_points=True,
            annotate_lambda=True,
        )

    if case_key == ("ranked_waterfall", "direction_colored"):
        _, ranked_frame, _ = inputs.ranked
        waterfall = ranked_frame.loc[
            ranked_frame["list_name"].eq("method_a")
        ].copy()
        return renderer(
            waterfall,
            value="value",
            label="feature",
            color_by="category",
            color_order=["negative", "positive"],
            palette={"negative": "#4477AA", "positive": "#CC6677"},
            ascending=True,
            tie_breaker="source_rank",
            y_reference_lines=[
                {"value": 0.0, "label": "No effect", "linestyle": "--"}
            ],
            xlabel="Simulated feature",
            ylabel="Simulated effect",
            title="Stable ranking of simulated feature effects",
            tick_rotation=45,
            figsize=(9, 5),
            show=False,
        )

    if case_key == ("residual_diagnostic", "log_fitted"):
        residuals = inputs.residuals.loc[
            inputs.residuals["feature"].eq("feature_positive")
        ].copy()
        return renderer(
            residuals,
            x="expected",
            residual="residual",
            x_transform="log10",
            y_reference_lines=[
                {"value": 0.0, "label": "Zero residual", "linestyle": "--"}
            ],
            point_color="#4477AA",
            xlabel="Expected abundance (log10)",
            ylabel="Observed minus expected",
            title="Expectation-model residual diagnostic",
            figsize=(7, 4.5),
            show=False,
        )

    if case_key == ("show_colors", "categorical_palette"):
        return renderer(
            colors=[
                "#4477AA",
                "#EE6677",
                "#228833",
                "#CCBB44",
                "#66CCEE",
                "#AA3377",
            ],
            title_text="Gallery categorical palette",
            save_plot=False,
        )

    if case_key == ("show_tol_colors", "tol_palette"):
        return renderer()

    if case_key == ("spearman_cor_dotplot", "spearman_fit"):
        return renderer(
            df=inputs.independent_frame.copy(),
            column_key_x="feature_positive",
            column_key_y="feature_negative",
            hue="condition",
            palette=GROUP_COLORS,
            dot_size=52,
            title_fontsize=14,
            stats_fontsize=9,
            axis_label_fontsize=11,
            tick_label_fontsize=9,
            legend_fontsize=8,
            axes_lines=False,
            axes_title="Spearman compatibility wrapper",
            xlabel="Positive-effect feature",
            ylabel="Negative-effect feature",
            figsize=(7.5, 5.5),
            show=False,
        )

    if case_key == ("spearman_cor_dotplot_2", "dual_hue"):
        frame = inputs.independent_frame.copy()
        return renderer(
            frame,
            "feature_positive",
            "feature_negative",
            "condition",
            "batch",
            figsize=(11, 5),
            axes_lines=False,
        )

    if case_key == ("timeseries_paired_datapoints", "faceted_time_series"):
        np.random.seed(20260727)
        return renderer(
            inputs.paired,
            feature_name="paired_increase",
            x_col="condition",
            feature_name_label_col="feature_label",
            layer="norm",
            Hue="cohort",
            subplotby="cohort",
            analyte_label="Simulated abundance",
            subject_col="subject_id",
            connect_lines=True,
            jitter_amount=0.08,
            legend=True,
            figsize=(8, 6),
            savefig=True,
            file_name=str(asset_path),
        )

    if case_key == ("venn_plot_2list", "two_set_overlap"):
        return renderer(
            list1=["STAT1", "IRF1", "CXCL10", "ISG15"],
            list2=["IRF1", "CXCL10", "MKI67", "PCNA"],
            set_label_list=["Signature A", "Signature B"],
            plot_title="Synthetic two-set feature overlap",
            show_plot=True,
            return_df=False,
        )

    if case_key == ("venn_plot_3list", "three_set_overlap"):
        return renderer(
            list1=["A_only", "AB", "AC", "ABC"],
            list2=["B_only", "AB", "BC", "ABC"],
            list3=["C_only", "AC", "BC", "ABC"],
            set_label_list=["Signature A", "Signature B", "Signature C"],
            plot_title="Synthetic three-set feature overlap",
            show_plot=True,
            return_df=False,
        )

    if spec.name == "volcano_plot_generic":
        pooled = inputs.pooled_diff_results
        if case.case_id == "significance":
            axes = renderer(
                pooled,
                l2fc_col="effect",
                pvalue_col="pvalue",
                set_xlabel="log2(case/control)",
                set_ylabel="-log10(p-value)",
                title_text="Library-derived differential tests",
                comparison_label="Case versus control across simulations",
                log2FoldChange_threshold=0.1,
                pvalue_threshold=0.05,
                xlimit=0.55,
                ylimit=19.0,
                figsize=(9, 6),
                legend_bbox_to_anchor=(1.24, 1),
                title_fontsize=14,
                axis_label_and_tick_fontsize=10,
                legend_fontsize=9,
                label_top_features=True,
                label_top_features_fontsize=7,
                feature_label_col="gene_names",
                n_top_features=2,
                dot_size_shrink_factor=4,
                deg_count_types=("total", "up", "down"),
                show_deg_counts_in_legend=True,
                label_threshold_regions=True,
                save_deg_counts_csv=False,
                savefig=False,
            )
            lower_region_y = -np.log10(0.05) * 0.75
            for annotation in axes.texts:
                if annotation.get_gid() in {
                    "volcano_threshold_region_lower_left",
                    "volcano_threshold_region_lower_center",
                    "volcano_threshold_region_lower_right",
                }:
                    annotation.set_y(lower_region_y)
                    annotation.set_verticalalignment("top")
                    annotation.set_fontsize(9)
                    annotation.set_text(annotation.get_text().replace("\n", "  "))
            return axes
        if case.case_id == "ranked_columns":
            return renderer(
                pooled,
                l2fc_col="effect",
                pvalue_col="padj",
                set_xlabel="log2(case/control)",
                set_ylabel="-log10(FDR-adjusted p-value)",
                title_text="Ranked differential-test labels",
                comparison_label="Case versus control across simulations",
                log2FoldChange_threshold=0.1,
                pvalue_threshold=0.05,
                xlimit=0.55,
                ylimit=19.0,
                figsize=(9, 6),
                legend_bbox_to_anchor=(1.24, 1),
                title_fontsize=14,
                axis_label_and_tick_fontsize=10,
                legend_fontsize=9,
                label_top_features=True,
                label_top_features_fontsize=7,
                label_features_char_limit=24,
                feature_label_col="gene_names",
                n_top_features=10,
                dot_size_shrink_factor=4,
                label_layout="ranked_columns",
                savefig=False,
            )
        if case.case_id == "feature_class":
            return renderer(
                pooled,
                l2fc_col="effect",
                pvalue_col="pvalue",
                set_xlabel="log2(case/control)",
                set_ylabel="-log10(p-value)",
                title_text="Differential tests by simulated truth class",
                comparison_label="Case versus control across simulations",
                hue_column="truth_class",
                log2FoldChange_threshold=0.1,
                pvalue_threshold=0.05,
                xlimit=0.55,
                ylimit=19.0,
                figsize=(9, 6),
                legend_bbox_to_anchor=(1.24, 1),
                title_fontsize=14,
                axis_label_and_tick_fontsize=10,
                legend_fontsize=9,
                label_top_features=True,
                only_label_hue_dots=True,
                label_top_features_fontsize=7,
                feature_label_col="gene_names",
                n_top_features=2,
                dot_size_shrink_factor=4,
                savefig=False,
            )

    if case_key == (
        "volcano_plot_sns_single_comparison_generic",
        "replacement_smoke",
    ):
        return renderer(
            inputs.pooled_diff_results,
            l2fc_col="effect",
            padj_col="padj",
            set_xlabel="log2(case/control)",
            set_ylabel="-log10(adjusted p-value)",
            title_text="Legacy API; use volcano_plot_generic",
            comparison_label="Case versus control",
            log2FoldChange_threshold=0.1,
            pvalue_threshold=0.05,
            xlimit=0.55,
            figsize=(8, 5.5),
            legend_bbox_to_anchor=(1.65, 1),
            label_top_features=True,
            feature_label_col="gene_names",
            n_top_features=2,
            dot_size_shrink_factor=4,
            savefig=False,
        )

    raise KeyError(f"No generator invocation is defined for {case_key!r}.")


def _select_cases(
    renderer_names: Iterable[str] | None,
    case_ids: Iterable[str] | None,
) -> list[tuple[RendererSpec, GalleryCase]]:
    requested_renderers = (
        set(MANIFEST_BY_NAME) if renderer_names is None else set(renderer_names)
    )
    unknown_renderers = requested_renderers.difference(MANIFEST_BY_NAME)
    if unknown_renderers:
        raise ValueError(f"Unknown renderer name(s): {sorted(unknown_renderers)}.")

    requested_cases = None if case_ids is None else set(case_ids)
    selected = [
        (spec, case)
        for spec in RENDERER_MANIFEST
        if spec.name in requested_renderers
        for case in spec.cases
        if requested_cases is None or case.case_id in requested_cases
    ]
    if requested_cases is not None:
        observed_case_ids = {case.case_id for _, case in selected}
        unknown_cases = requested_cases.difference(observed_case_ids)
        if unknown_cases:
            raise ValueError(
                "Unknown case id(s) for the selected renderer set: "
                f"{sorted(unknown_cases)}."
            )
    if not selected:
        raise ValueError("No gallery cases matched the requested selection.")
    return selected


def _generate_selected(
    output_dir: Path,
    selected: Sequence[tuple[RendererSpec, GalleryCase]],
    *,
    continue_on_error: bool,
) -> tuple[list[Path], list[tuple[str, str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = GalleryInputs()
    generated: list[Path] = []
    failures: list[tuple[str, str]] = []
    default_rng = np.random.default_rng

    def gallery_default_rng(seed: Any = None) -> np.random.Generator:
        """Seed otherwise-unseeded NumPy Generator users during rendering."""

        return default_rng(20260727 if seed is None else seed)

    for spec, case in selected:
        case_name = f"{spec.name}.{case.case_id}"
        asset_path = output_dir / case.asset
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=asset_path.parent,
            prefix=f".{asset_path.stem}.",
            suffix=".png",
            delete=False,
        ) as temporary_file:
            temporary_asset_path = Path(temporary_file.name)
        open_before = set(plt.get_fignums())
        random_state = np.random.get_state()
        result_figure: Figure | None = None
        try:
            np.random.seed(20260727)
            with (
                patch.object(np.random, "default_rng", gallery_default_rng),
                patch.object(plt, "show"),
                plt.rc_context(
                    {
                        "figure.facecolor": "white",
                        "axes.facecolor": "white",
                        "savefig.facecolor": "white",
                        "font.size": 10,
                    }
                ),
            ):
                result = _invoke_case(
                    spec,
                    case,
                    inputs,
                    temporary_asset_path,
                )
                result_figure = _figure_from_result(result)
                if temporary_asset_path.stat().st_size == 0:
                    if result_figure is None:
                        new_figure_numbers = [
                            number
                            for number in plt.get_fignums()
                            if number not in open_before
                        ]
                        if new_figure_numbers:
                            result_figure = plt.figure(new_figure_numbers[-1])
                    if result_figure is None:
                        raise GalleryGenerationError(
                            f"{case_name} returned no figure and did not save "
                            f"{case.asset!r}."
                        )
                    result_figure.savefig(
                        temporary_asset_path,
                        dpi=140,
                        bbox_inches="tight",
                        facecolor="white",
                    )
            with temporary_asset_path.open("rb") as generated_file:
                png_signature = generated_file.read(len(PNG_SIGNATURE))
            if png_signature != PNG_SIGNATURE:
                raise GalleryGenerationError(
                    f"{case_name} did not produce a valid PNG for "
                    f"{asset_path}."
                )
            temporary_asset_path.chmod(0o644)
            temporary_asset_path.replace(asset_path)
            generated.append(asset_path)
        except Exception as exc:
            message = f"{case_name} failed: {type(exc).__name__}: {exc}"
            if not continue_on_error:
                raise GalleryGenerationError(message) from exc
            failures.append((case_name, traceback.format_exc()))
        finally:
            temporary_asset_path.unlink(missing_ok=True)
            np.random.set_state(random_state)
            if result_figure is not None:
                plt.close(result_figure)
            for figure_number in set(plt.get_fignums()).difference(open_before):
                plt.close(figure_number)

    return generated, failures


def generate_gallery(
    output_dir: str | Path,
    *,
    renderer_names: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """Generate selected gallery assets, raising on the first failed renderer.

    Parameters
    ----------
    output_dir
        Directory that receives the manifest-declared PNG filenames.
    renderer_names
        Optional exported renderer names. By default all 45 are invoked.
    case_ids
        Optional manifest case IDs, applied within the selected renderer set.
    """

    selected = _select_cases(renderer_names, case_ids)
    generated, failures = _generate_selected(
        Path(output_dir),
        selected,
        continue_on_error=False,
    )
    if failures:
        raise AssertionError("Strict gallery generation unexpectedly retained failures.")
    return tuple(generated)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic plotting-gallery PNG assets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="YAML configuration file (default: repository config/config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the plotting_gallery_params output directory.",
    )
    parser.add_argument(
        "--renderer",
        dest="renderer_names",
        action="append",
        choices=sorted(MANIFEST_BY_NAME),
        help="Generate only this renderer; repeat to select multiple renderers.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="Generate only this case ID within the selected renderer set.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override plotting_gallery_params continue_on_error.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    args = _build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    gallery_cfg = cfg["plotting_gallery_params"]

    if args.output_dir is None:
        output_dir = REPO_ROOT / Path(gallery_cfg["output_dir"])
    else:
        output_dir = args.output_dir
    log_dir = REPO_ROOT / Path(gallery_cfg["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"generate_gallery_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.captureWarnings(True)

    renderer_names = (
        args.renderer_names
        if args.renderer_names is not None
        else gallery_cfg.get("renderer_names")
    )
    case_ids = (
        args.case_ids if args.case_ids is not None else gallery_cfg.get("case_ids")
    )
    continue_on_error = (
        args.continue_on_error
        if args.continue_on_error is not None
        else bool(gallery_cfg.get("continue_on_error", False))
    )

    LOGGER.info("Loaded plotting gallery configuration from %s", args.config)
    LOGGER.info("Writing gallery assets to %s", output_dir)
    LOGGER.info("Writing gallery logs to %s", log_path)
    selected = _select_cases(renderer_names, case_ids)
    generated, failures = _generate_selected(
        output_dir,
        selected,
        continue_on_error=continue_on_error,
    )
    for asset_path in generated:
        LOGGER.info("Generated %s", asset_path)
    if failures:
        LOGGER.error(
            f"Generated {len(generated)} of {len(selected)} gallery cases; "
            f"{len(failures)} failed."
        )
        for case_name, failure_traceback in failures:
            LOGGER.error("[%s]\n%s", case_name, failure_traceback)
        return 1
    LOGGER.info("Generated %d gallery cases in %s.", len(generated), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
