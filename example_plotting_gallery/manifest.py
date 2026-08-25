"""Declarative coverage manifest for the plotting example gallery.

The manifest is intentionally independent of the plotting package so it can be
used by coverage tests without importing Matplotlib or optional scientific
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal


RendererStatus = Literal["maintained", "compatibility", "deprecated"]
CaseTier = Literal["major", "smoke"]


@dataclass(frozen=True)
class GalleryCase:
    """One deterministic renderer invocation and its expected gallery asset."""

    case_id: str
    title: str
    asset: str
    features: tuple[str, ...]
    tier: CaseTier = "major"
    canonical_asset: str | None = None


@dataclass(frozen=True)
class RendererSpec:
    """Gallery metadata for one package-defined exported renderer."""

    name: str
    module: str
    status: RendererStatus
    provenance: str
    cases: tuple[GalleryCase, ...]
    replacement: str | None = None
    notes: str | None = None


RENDERER_MANIFEST: tuple[RendererSpec, ...] = (
    RendererSpec(
        name="adata_histograms",
        module="_plotting._histograms",
        status="maintained",
        provenance="make_independent_group_adata; simulated measurements and observed sample groups",
        cases=(
            GalleryCase(
                case_id="subgroup_kde",
                title="Distributions by treatment",
                asset="adata_histograms__subgroup_kde.png",
                features=(
                    "observation subgroup overlays",
                    "all-observation context",
                    "KDE and mean/reference lines",
                ),
            ),
            GalleryCase(
                case_id="feature_group_collapse",
                title="Grouped feature distributions",
                asset="adata_histograms__feature_group_collapse.png",
                features=(
                    "variable-metadata grouping",
                    "within-observation aggregation",
                    "shared panel scale",
                ),
            ),
        ),
    ),
    RendererSpec(
        name="barh_4X_dotplot_column",
        module="_plotting._column_plots",
        status="compatibility",
        provenance=(
            "make_independent_group_adata plus run_independent_diff_test and "
            "fit_smf_ols_models_and_summarize_adata; library-derived "
            "differential and adjusted OLS results"
        ),
        cases=(
            GalleryCase(
                case_id="five_panel",
                title="Expression with four inferential summaries",
                asset="barh_4X_dotplot_column__five_panel.png",
                features=(
                    "grouped expression bars",
                    "four effect/p-value encodings",
                    "shared dotplot color scale",
                ),
                canonical_asset=(
                    "datapoints_effect_panels_column__horizontal_four_effects.png"
                ),
            ),
        ),
        replacement="datapoints_effect_panels_column",
    ),
    RendererSpec(
        name="barh_column",
        module="_plotting._column_plots",
        status="maintained",
        provenance="make_independent_group_adata; simulated measurements and observed treatment groups",
        cases=(
            GalleryCase(
                case_id="grouped_expression",
                title="Feature abundance by treatment",
                asset="barh_column__grouped_expression.png",
                features=("grouped horizontal bars", "sample strip overlay", "ordered treatment legend"),
            ),
        ),
    ),
    RendererSpec(
        name="barh_dotplot_dotplot_column",
        module="_plotting._column_plots",
        status="compatibility",
        provenance="make_independent_group_adata plus run_independent_diff_test; library-derived differential results",
        cases=(
            GalleryCase(
                case_id="three_panel",
                title="Expression with two differential summaries",
                asset="barh_dotplot_dotplot_column__three_panel.png",
                features=("grouped expression bars", "two effect/p-value columns", "feature-aligned rows"),
                canonical_asset=(
                    "datapoints_effect_panels_column__horizontal_two_effects.png"
                ),
            ),
        ),
        replacement="datapoints_effect_panels_column",
    ),
    RendererSpec(
        name="barh_dotplot_dotplot_dotplot_column",
        module="_plotting._column_plots",
        status="compatibility",
        provenance=(
            "make_independent_group_adata plus run_independent_diff_test and "
            "fit_smf_ols_models_and_summarize_adata; library-derived "
            "differential and adjusted OLS results"
        ),
        cases=(
            GalleryCase(
                case_id="four_panel",
                title="Expression with three differential summaries",
                asset="barh_dotplot_dotplot_dotplot_column__four_panel.png",
                features=("grouped expression bars", "three effect/p-value columns", "feature-aligned rows"),
                canonical_asset=(
                    "datapoints_effect_panels_column__horizontal_three_effects.png"
                ),
            ),
        ),
        replacement="datapoints_effect_panels_column",
    ),
    RendererSpec(
        name="barh_l2fc_dotplot_column",
        module="_plotting._column_plots",
        status="compatibility",
        provenance="make_independent_group_adata plus run_independent_diff_test; library-derived differential results",
        cases=(
            GalleryCase(
                case_id="two_panel",
                title="Expression and differential effect",
                asset="barh_l2fc_dotplot_column__two_panel.png",
                features=("grouped expression bars", "effect direction", "p-value size and threshold ring"),
                canonical_asset=(
                    "datapoints_effect_panels_column__horizontal_one_effect.png"
                ),
            ),
        ),
        replacement="datapoints_effect_panels_column",
    ),
    RendererSpec(
        name="datapoints_effect_panels_column",
        module="_plotting._column_plots",
        status="maintained",
        provenance=(
            "make_independent_group_adata plus run_independent_diff_test for "
            "the exploratory and one- through four-effect cases, with "
            "fit_smf_ols_models_and_summarize_adata supplying adjusted OLS "
            "panels in the three- and four-effect cases; "
            "synthetic_expression.csv plus "
            "synthetic_effects.csv for the supplied-interval case"
        ),
        cases=(
            GalleryCase(
                case_id="horizontal_pvalue",
                title="Horizontal distributions with p-value effects",
                asset="datapoints_effect_panels_column__horizontal_pvalue.png",
                features=(
                    "horizontal grouped distributions",
                    "effect direction",
                    "p-value size and threshold ring",
                ),
            ),
            GalleryCase(
                case_id="horizontal_one_effect",
                title="Horizontal distributions with one effect summary",
                asset=(
                    "datapoints_effect_panels_column__horizontal_one_effect.png"
                ),
                features=(
                    "barh_l2fc_dotplot_column replacement",
                    "grouped expression bars",
                    "one effect/p-value panel",
                ),
            ),
            GalleryCase(
                case_id="horizontal_two_effects",
                title="Horizontal distributions with two effect summaries",
                asset=(
                    "datapoints_effect_panels_column__horizontal_two_effects.png"
                ),
                features=(
                    "barh_dotplot_dotplot_column replacement",
                    "two ordered effect/p-value panels",
                    "independent p-value scales",
                ),
            ),
            GalleryCase(
                case_id="horizontal_three_effects",
                title="Horizontal distributions with three effect summaries",
                asset=(
                    "datapoints_effect_panels_column__horizontal_three_effects.png"
                ),
                features=(
                    "barh_dotplot_dotplot_dotplot_column replacement",
                    "three ordered effect/p-value panels",
                    "unadjusted and adjusted summaries",
                ),
            ),
            GalleryCase(
                case_id="horizontal_four_effects",
                title="Horizontal distributions with four effect summaries",
                asset="datapoints_effect_panels_column__horizontal_four_effects.png",
                features=(
                    "one shared distribution column",
                    "four ordered p-value effect panels",
                    "shared p-value color and size scale",
                ),
            ),
            GalleryCase(
                case_id="vertical_interval",
                title="Vertical distributions with supplied intervals",
                asset="datapoints_effect_panels_column__vertical_interval.png",
                features=(
                    "vertical boxplots with observation overlays",
                    "subtype color and cohort shape",
                    "supplied effect confidence intervals",
                ),
            ),
        ),
    ),
    RendererSpec(
        name="vbar_l2fc_dotplot_column",
        module="_plotting._column_plots",
        status="compatibility",
        provenance=(
            "synthetic_expression.csv plus synthetic_effects.csv; all values, "
            "identifiers, groups, and supplied intervals are synthetic"
        ),
        cases=(
            GalleryCase(
                case_id="synthetic_response_panel",
                title="Synthetic response-associated expression panel",
                asset="vbar_l2fc_dotplot_column__synthetic_response_panel.png",
                features=(
                    "vertical boxplots with observation overlays",
                    "subtype color and cohort shape",
                    "supplied effect confidence intervals",
                ),
                canonical_asset=(
                    "datapoints_effect_panels_column__vertical_interval.png"
                ),
            ),
        ),
        replacement="datapoints_effect_panels_column",
    ),
    RendererSpec(
        name="category_composition",
        module="_plotting._tabular_plots",
        status="maintained",
        provenance="make_composition_frame; deterministic simulated sample categories",
        cases=(
            GalleryCase(
                case_id="percent_annotated",
                title="Response composition by cohort",
                asset="category_composition__percent_annotated.png",
                features=("percentage normalization", "explicit category order", "segment annotations"),
            ),
        ),
    ),
    RendererSpec(
        name="continuous_effect_plot",
        module="_plotting._analytical_plots",
        status="maintained",
        provenance=(
            "make_continuous_effect_frames; deterministic precomputed curve "
            "and interval with seeded simulated observations"
        ),
        cases=(
            GalleryCase(
                case_id="observed_categories",
                title="Continuous exposure effect",
                asset="continuous_effect_plot__observed_categories.png",
                features=("confidence band", "log-scaled exposure", "categorized observed markers"),
            ),
        ),
    ),
    RendererSpec(
        name="corr_dotplot",
        module="_plotting._corr_dotplots",
        status="maintained",
        provenance="make_independent_group_adata; deterministic simulated covariates and feature measurements",
        cases=(
            GalleryCase(
                case_id="subgroup_marginals",
                title="Correlation with subgroup fits and marginals",
                asset="corr_dotplot__subgroup_marginals.png",
                features=(
                    "subgroup regression fits",
                    "independent point color",
                    "x and y marginal histograms",
                ),
            ),
            GalleryCase(
                case_id="log1p_identity",
                title="Correlation on synchronized log1p axes",
                asset="corr_dotplot__log1p_identity.png",
                features=("log1p axis transforms", "identity line", "custom reference lines"),
            ),
        ),
    ),
    RendererSpec(
        name="corr_dotplot_dev",
        module="_plotting._corr_dotplots",
        status="deprecated",
        provenance="make_independent_group_adata; deterministic compatibility smoke input",
        replacement="corr_dotplot",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Deprecated correlation wrapper",
                asset="corr_dotplot_dev__replacement_smoke.png",
                features=("legacy invocation", "replacement parity"),
                tier="smoke",
            ),
        ),
        notes="Deprecated wrapper retained for callers that require the legacy axes-dictionary return.",
    ),
    RendererSpec(
        name="datapoints",
        module="_plotting._datapoints",
        status="maintained",
        provenance="make_independent_group_adata; simulated measurements and observed sample metadata",
        cases=(
            GalleryCase(
                case_id="grouped_markers",
                title="Feature values with independent marker and color encodings",
                asset="datapoints__grouped_markers.png",
                features=(
                    "observation-group x axis",
                    "subset color",
                    "marker category",
                    "summary annotation",
                ),
            ),
            GalleryCase(
                case_id="feature_group_collapse",
                title="Collapsed feature-group datapoints",
                asset="datapoints__feature_group_collapse.png",
                features=("variable-metadata grouping", "aggregate collapse", "panel by variable group"),
            ),
        ),
    ),
    RendererSpec(
        name="forest",
        module="_plotting._forest",
        status="maintained",
        provenance="make_independent_group_adata plus fit_smf_ols_models_and_summarize_adata; library-derived OLS estimates, confidence intervals, p-values, and sample counts",
        cases=(
            GalleryCase(
                case_id="grouped_estimates",
                title="Grouped model estimates",
                asset="forest__grouped_estimates.png",
                features=("dodged estimate groups", "confidence-interval clipping", "p-value and sample-size encodings"),
            ),
            GalleryCase(
                case_id="ratio_scale",
                title="Ratio-scale effects",
                asset="forest__ratio_scale.png",
                features=("ratio null at one", "logarithmic x axis", "auditable annotation table"),
            ),
        ),
    ),
    RendererSpec(
        name="geneset_enrichemnt_ol_ven_M_n_N_x",
        module="_plotting._venn_plots",
        status="deprecated",
        provenance="fixed synthetic universe, hit, and gene-set identifiers passed through the legacy enrichment API",
        replacement="geneset_enrichment_venn",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy gene-set enrichment overlap",
                asset="geneset_enrichemnt_ol_ven_M_n_N_x__replacement_smoke.png",
                features=("legacy invocation", "hypergeometric p-value", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="geneset_enrichment_venn",
        module="_plotting._venn_plots",
        status="maintained",
        provenance="fixed synthetic universe, hit, and gene-set identifiers; library-derived upper-tail hypergeometric p-value",
        cases=(
            GalleryCase(
                case_id="universe_filtered",
                title="Gene-set enrichment overlap",
                asset="geneset_enrichment_venn__universe_filtered.png",
                features=("universe filtering", "overlap count", "hypergeometric p-value"),
            ),
        ),
    ),
    RendererSpec(
        name="kaplan_meier_plot",
        module="_plotting._analytical_plots",
        status="maintained",
        provenance="make_survival_frames; deterministic precomputed curves, risk counts, and censor positions",
        cases=(
            GalleryCase(
                case_id="grouped_risk_censor",
                title="Survival curves with numbers at risk",
                asset="kaplan_meier_plot__grouped_risk_censor.png",
                features=("grouped confidence bands", "censor markers", "aligned risk table"),
            ),
        ),
    ),
    RendererSpec(
        name="l2fc_dotplot_column",
        module="_plotting._column_plots",
        status="maintained",
        provenance="run_independent_diff_test; library-derived differential results",
        cases=(
            GalleryCase(
                case_id="multi_feature",
                title="Differential effects by feature",
                asset="l2fc_dotplot_column__multi_feature.png",
                features=("feature-aligned axes", "effect direction", "p-value size and threshold ring"),
            ),
        ),
    ),
    RendererSpec(
        name="l2fc_dotplot_single",
        module="_plotting._column_plots",
        status="maintained",
        provenance="run_independent_diff_test; library-derived differential results",
        cases=(
            GalleryCase(
                case_id="single_axis",
                title="Differential effect overview",
                asset="l2fc_dotplot_single__single_axis.png",
                features=("single-axis feature ordering", "effect direction", "p-value size and threshold ring"),
            ),
        ),
    ),
    RendererSpec(
        name="l2fc_pvalue_dotplot_gex",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="run_independent_diff_test reshaped to the legacy table contract",
        replacement="l2fc_dotplot_single",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy gene-expression effect dotplot",
                asset="l2fc_pvalue_dotplot_gex__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="l2fc_pvalue_dotplot_protein_metabolite",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="run_independent_diff_test reshaped to the legacy table contract",
        replacement="l2fc_dotplot_single",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy protein/metabolite effect dotplot",
                asset="l2fc_pvalue_dotplot_protein_metabolite__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="longitudinal_trajectories",
        module="_plotting._longitudinal",
        status="maintained",
        provenance="make_longitudinal_frame; deterministic repeated-measure observations with an intentional visit gap",
        cases=(
            GalleryCase(
                case_id="markers_and_gaps",
                title="Longitudinal trajectories with visit gaps",
                asset="longitudinal_trajectories__markers_and_gaps.png",
                features=("adjacent-only connections", "independent color and marker channels", "exact versus displayed values"),
            ),
        ),
    ),
    RendererSpec(
        name="meta_forest",
        module="_plotting._meta_forest",
        status="maintained",
        provenance="make_meta_forest_rows; deterministic precomputed study, heading, summary, and prediction rows",
        cases=(
            GalleryCase(
                case_id="study_summary_prediction",
                title="Meta-analysis summary",
                asset="meta_forest__study_summary_prediction.png",
                features=("study weights", "summary diamonds", "prediction intervals", "side table"),
            ),
        ),
    ),
    RendererSpec(
        name="paired_datapoints",
        module="_plotting._datapoints",
        status="maintained",
        provenance="make_paired_adata; deterministic paired subjects, conditions, and treatment strata",
        cases=(
            GalleryCase(
                case_id="paired_groups",
                title="Paired changes by treatment",
                asset="paired_datapoints__paired_groups.png",
                features=("subject pairing", "subset colors", "feature-group aggregation"),
            ),
            GalleryCase(
                case_id="slope_colored_lines",
                title="Paired lines colored by relative change",
                asset="paired_datapoints__slope_colored_lines.png",
                features=(
                    "symmetric average-relative change",
                    "positive, negative, and approximately-flat line colors",
                    "5% flat threshold",
                ),
            ),
            GalleryCase(
                case_id="difference_axis",
                title="Varied paired slopes and signed differences",
                asset="paired_datapoints__difference_axis.png",
                features=(
                    "varied slope magnitudes",
                    "combined positive, negative, and flat panel",
                    "raw signed paired differences",
                    "zero-centered symmetric secondary y axes",
                ),
            ),
            GalleryCase(
                case_id="log2fc_axis",
                title="Varied paired slopes and log2 fold changes",
                asset="paired_datapoints__log2fc_axis.png",
                features=(
                    "varied slope magnitudes",
                    "combined positive, negative, and flat panel",
                    "post-over-baseline log2 fold changes",
                    "zero-centered symmetric secondary y axes",
                ),
            ),
            GalleryCase(
                case_id="precomputed_pair_values",
                title="Paired values from preserved source matrices",
                asset="paired_datapoints__precomputed_pair_values.png",
                features=("obsm reference/target values", "one row per preserved pair", "feature panels"),
            ),
        ),
    ),
    RendererSpec(
        name="plot_column_of_bar_h_2groups_GEX_adata",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="make_independent_group_adata; deterministic compatibility smoke input",
        replacement="barh_column",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy grouped expression bars",
                asset="plot_column_of_bar_h_2groups_GEX_adata__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="make_independent_group_adata plus run_independent_diff_test; deterministic compatibility smoke input",
        replacement="barh_l2fc_dotplot_column",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy expression and effect composite",
                asset="plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="plot_columns",
        module="_plotting._row_plots",
        status="maintained",
        provenance="make_independent_group_adata; simulated covariates and observed treatment groups",
        cases=(
            GalleryCase(
                case_id="multi_metric",
                title="Grouped covariate summaries",
                asset="plot_columns__multi_metric.png",
                features=("row of metric panels", "group means and confidence intervals", "sample swarm overlays"),
            ),
        ),
    ),
    RendererSpec(
        name="plot_heatmap",
        module="_plotting._corr_dotplots",
        status="maintained",
        provenance="make_ranked_inputs plus pairwise_spearman_corr_matrix; deterministic library-derived rank-correlation matrix",
        cases=(
            GalleryCase(
                case_id="clustered",
                title="Clustered rank-correlation heatmap",
                asset="plot_heatmap__clustered.png",
                features=("hierarchical row/column ordering", "annotated correlation cells"),
            ),
            GalleryCase(
                case_id="fixed_order",
                title="Fixed-order rank-correlation heatmap",
                asset="plot_heatmap__fixed_order.png",
                features=("caller-preserved order", "annotated correlation cells"),
            ),
        ),
    ),
    RendererSpec(
        name="plot_paired_point_anndata",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="make_paired_adata; deterministic compatibility smoke input",
        replacement="timeseries_paired_datapoints",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy paired time-series datapoints",
                asset="plot_paired_point_anndata__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="plot_rank_heatmap",
        module="_plotting._corr_dotplots",
        status="maintained",
        provenance="make_ranked_inputs; deterministic ranked feature lists",
        cases=(
            GalleryCase(
                case_id="rank_hexbin",
                title="Rank agreement density",
                asset="plot_rank_heatmap__rank_hexbin.png",
                features=("common-item ranks", "hexbin count encoding", "identity reference"),
            ),
        ),
    ),
    RendererSpec(
        name="plot_rank_scatter",
        module="_plotting._corr_dotplots",
        status="maintained",
        provenance="make_ranked_inputs; deterministic ranked feature lists",
        cases=(
            GalleryCase(
                case_id="rank_agreement",
                title="Rank agreement",
                asset="plot_rank_scatter__rank_agreement.png",
                features=("common-item ranks", "Spearman summary", "identity reference"),
            ),
        ),
    ),
    RendererSpec(
        name="plot_rank_scatter_density",
        module="_plotting._corr_dotplots",
        status="maintained",
        provenance="make_ranked_inputs; deterministic non-collinear ranked feature lists",
        cases=(
            GalleryCase(
                case_id="rank_density",
                title="Rank agreement with local density",
                asset="plot_rank_scatter_density__rank_density.png",
                features=("common-item ranks", "KDE density color", "identity reference"),
            ),
        ),
    ),
    RendererSpec(
        name="qqplot",
        module="_plotting._plots",
        status="maintained",
        provenance="run_independent_diff_test; library-derived p-values",
        cases=(
            GalleryCase(
                case_id="differential_pvalues",
                title="Differential-test p-value QQ plot",
                asset="qqplot__differential_pvalues.png",
                features=("observed versus expected quantiles", "identity reference", "genomic inflation annotation"),
            ),
        ),
    ),
    RendererSpec(
        name="qqplot_pvalues",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="run_independent_diff_test; library-derived p-values passed through the legacy API",
        replacement="qqplot",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy p-value QQ plot",
                asset="qqplot_pvalues__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
    RendererSpec(
        name="ranked_waterfall",
        module="_plotting._tabular_plots",
        status="maintained",
        provenance="make_ranked_inputs; deterministic feature effects and directions",
        cases=(
            GalleryCase(
                case_id="direction_colored",
                title="Ranked feature effects",
                asset="ranked_waterfall__direction_colored.png",
                features=("stable effect ranking", "direction color", "zero reference"),
            ),
        ),
    ),
    RendererSpec(
        name="residual_diagnostic",
        module="_plotting._tabular_plots",
        status="maintained",
        provenance="make_independent_group_adata plus calculate_expectations, predict_expectation, and excess_expectation; deterministic library-derived fitted values and residuals",
        cases=(
            GalleryCase(
                case_id="log_fitted",
                title="Residuals versus fitted abundance",
                asset="residual_diagnostic__log_fitted.png",
                features=("caller-supplied residuals", "log10 fitted-value transform", "zero reference"),
            ),
        ),
    ),
    RendererSpec(
        name="show_colors",
        module="_plotting._utils",
        status="maintained",
        provenance="library palette constants; no simulated biological data",
        cases=(
            GalleryCase(
                case_id="categorical_palette",
                title="Categorical plotting palette",
                asset="show_colors__categorical_palette.png",
                features=("caller-supplied palette", "hex labels"),
            ),
        ),
    ),
    RendererSpec(
        name="show_tol_colors",
        module="_plotting._utils",
        status="maintained",
        provenance="library default Paul Tol palette; no simulated biological data",
        cases=(
            GalleryCase(
                case_id="tol_palette",
                title="Paul Tol plotting palette",
                asset="show_tol_colors__tol_palette.png",
                features=("library default palette", "hex labels"),
            ),
        ),
    ),
    RendererSpec(
        name="spearman_cor_dotplot",
        module="_plotting._corr_dotplots",
        status="compatibility",
        provenance="make_independent_group_adata; deterministic simulated covariates and feature measurements",
        replacement="corr_dotplot",
        cases=(
            GalleryCase(
                case_id="spearman_fit",
                title="Spearman correlation compatibility API",
                asset="spearman_cor_dotplot__spearman_fit.png",
                features=("forced Spearman method", "subgroup point colors", "fit and correlation summary"),
            ),
        ),
        notes="Maintained compatibility wrapper that delegates to corr_dotplot with method='spearman'.",
    ),
    RendererSpec(
        name="spearman_cor_dotplot_2",
        module="_plotting._corr_dotplots",
        status="maintained",
        provenance="make_independent_group_adata; deterministic simulated covariates with two categorical encodings",
        cases=(
            GalleryCase(
                case_id="dual_hue",
                title="Correlation under two categorical encodings",
                asset="spearman_cor_dotplot_2__dual_hue.png",
                features=("side-by-side hue encodings", "shared linear fit", "Spearman summary"),
            ),
        ),
    ),
    RendererSpec(
        name="timeseries_paired_datapoints",
        module="_plotting._plots",
        status="maintained",
        provenance="make_paired_adata; deterministic repeated measures, treatment groups, and source layer",
        cases=(
            GalleryCase(
                case_id="faceted_time_series",
                title="Paired time-series datapoints",
                asset="timeseries_paired_datapoints__faceted_time_series.png",
                features=("subject connections", "categorical time order", "faceted cohort panels"),
            ),
        ),
    ),
    RendererSpec(
        name="venn_plot_2list",
        module="_plotting._venn_plots",
        status="maintained",
        provenance="fixed synthetic feature identifiers with deterministic two-set membership",
        cases=(
            GalleryCase(
                case_id="two_set_overlap",
                title="Two feature-set overlap",
                asset="venn_plot_2list__two_set_overlap.png",
                features=("exclusive regions", "shared region", "set totals"),
            ),
        ),
    ),
    RendererSpec(
        name="venn_plot_3list",
        module="_plotting._venn_plots",
        status="maintained",
        provenance="fixed synthetic feature identifiers spanning all seven three-set regions",
        cases=(
            GalleryCase(
                case_id="three_set_overlap",
                title="Three feature-set overlap",
                asset="venn_plot_3list__three_set_overlap.png",
                features=("seven exclusive regions", "three-way overlap", "set totals"),
            ),
        ),
    ),
    RendererSpec(
        name="volcano_plot_generic",
        module="_plotting._plots",
        status="maintained",
        provenance="run_independent_diff_test; library-derived effects and p-values",
        cases=(
            GalleryCase(
                case_id="significance",
                title="Differential-test volcano plot",
                asset="volcano_plot_generic__significance.png",
                features=("nested significance levels", "effect and p-value thresholds", "top-feature labels"),
            ),
            GalleryCase(
                case_id="ranked_columns",
                title="Ranked volcano labels in side columns",
                asset="volcano_plot_generic__ranked_columns.png",
                features=(
                    "p-value-ranked labels",
                    "signed side columns",
                    "leader lines to plotted points",
                ),
            ),
            GalleryCase(
                case_id="feature_class",
                title="Volcano plot with feature-class highlighting",
                asset="volcano_plot_generic__feature_class.png",
                features=("custom hue column", "background context layer", "selected-feature labels"),
            ),
        ),
    ),
    RendererSpec(
        name="volcano_plot_sns_single_comparison_generic",
        module="_plotting._plots_depreciated",
        status="deprecated",
        provenance="run_independent_diff_test; library-derived effects and adjusted p-values passed through the legacy API",
        replacement="volcano_plot_generic",
        cases=(
            GalleryCase(
                case_id="replacement_smoke",
                title="Legacy adjusted-p volcano plot",
                asset="volcano_plot_sns_single_comparison_generic__replacement_smoke.png",
                features=("legacy invocation", "replacement guidance"),
                tier="smoke",
            ),
        ),
    ),
)


EXPECTED_RENDERER_NAMES = frozenset(
    {
        "adata_histograms",
        "barh_4X_dotplot_column",
        "barh_column",
        "barh_dotplot_dotplot_column",
        "barh_dotplot_dotplot_dotplot_column",
        "barh_l2fc_dotplot_column",
        "category_composition",
        "continuous_effect_plot",
        "corr_dotplot",
        "corr_dotplot_dev",
        "datapoints",
        "datapoints_effect_panels_column",
        "forest",
        "geneset_enrichemnt_ol_ven_M_n_N_x",
        "geneset_enrichment_venn",
        "kaplan_meier_plot",
        "l2fc_dotplot_column",
        "l2fc_dotplot_single",
        "l2fc_pvalue_dotplot_gex",
        "l2fc_pvalue_dotplot_protein_metabolite",
        "longitudinal_trajectories",
        "meta_forest",
        "paired_datapoints",
        "plot_column_of_bar_h_2groups_GEX_adata",
        "plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata",
        "plot_columns",
        "plot_heatmap",
        "plot_paired_point_anndata",
        "plot_rank_heatmap",
        "plot_rank_scatter",
        "plot_rank_scatter_density",
        "qqplot",
        "qqplot_pvalues",
        "ranked_waterfall",
        "residual_diagnostic",
        "show_colors",
        "show_tol_colors",
        "spearman_cor_dotplot",
        "spearman_cor_dotplot_2",
        "timeseries_paired_datapoints",
        "venn_plot_2list",
        "venn_plot_3list",
        "volcano_plot_generic",
        "volcano_plot_sns_single_comparison_generic",
        "vbar_l2fc_dotplot_column",
    }
)

EXCLUDED_PUBLIC_CALLABLES = {
    "compare_ranked_lists": "analysis helper returning overlap and rank statistics, not a renderer",
    "pairwise_spearman_corr_matrix": "analysis helper returning a correlation matrix, not a renderer",
    "linregress": "SciPy callable leaked by a star import",
    "spearmanr": "SciPy callable leaked by a star import",
}

RENDERER_NAMES = frozenset(spec.name for spec in RENDERER_MANIFEST)
MANIFEST_BY_NAME = {spec.name: spec for spec in RENDERER_MANIFEST}


def validate_manifest() -> None:
    """Raise ``ValueError`` when the gallery coverage contract is inconsistent."""

    if len(RENDERER_MANIFEST) != 45:
        raise ValueError(f"Expected 45 renderer entries, found {len(RENDERER_MANIFEST)}.")
    if len(RENDERER_NAMES) != len(RENDERER_MANIFEST):
        raise ValueError("Renderer names must be unique.")
    if RENDERER_NAMES != EXPECTED_RENDERER_NAMES:
        missing = sorted(EXPECTED_RENDERER_NAMES - RENDERER_NAMES)
        unexpected = sorted(RENDERER_NAMES - EXPECTED_RENDERER_NAMES)
        raise ValueError(f"Renderer coverage mismatch; missing={missing}, unexpected={unexpected}.")

    assets: set[str] = set()
    case_keys: set[tuple[str, str]] = set()
    for spec in RENDERER_MANIFEST:
        if not spec.cases:
            raise ValueError(f"Renderer {spec.name!r} must define at least one case.")
        if spec.status == "deprecated" and not spec.replacement:
            raise ValueError(f"Deprecated renderer {spec.name!r} must name a replacement.")
        for case in spec.cases:
            key = (spec.name, case.case_id)
            if key in case_keys:
                raise ValueError(f"Duplicate gallery case: {key!r}.")
            case_keys.add(key)
            path = PurePosixPath(case.asset)
            if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
                raise ValueError(f"Gallery asset must be a relative PNG path: {case.asset!r}.")
            if case.asset in assets:
                raise ValueError(f"Gallery asset paths must be unique: {case.asset!r}.")
            assets.add(case.asset)

    for spec in RENDERER_MANIFEST:
        for case in spec.cases:
            if case.canonical_asset is not None and case.canonical_asset not in assets:
                raise ValueError(
                    f"Canonical asset {case.canonical_asset!r} for "
                    f"{spec.name}.{case.case_id} is not declared."
                )


validate_manifest()
