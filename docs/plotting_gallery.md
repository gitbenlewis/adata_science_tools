# Deterministic plotting gallery

This gallery is a repository-owned visual catalog for the plotting functions
exported by `adata_science_tools._plotting`. The coverage contract and asset
filenames come from
[`example_plotting_gallery/manifest.py`](../example_plotting_gallery/manifest.py):
45 renderers and 53 cases, split across maintained, compatibility, and
deprecated APIs.

The examples are for rendering and API coverage. They are not benchmark
datasets or evidence for biological conclusions.

## Data and analysis provenance

The inputs in
[`example_plotting_gallery/simulated_data.py`](../example_plotting_gallery/simulated_data.py)
are deterministic. Seeded builders create balanced independent groups, paired
subjects, repeated measurements, continuous observations, and composition
categories. Other builders return fixed survival curves and meta-analysis
rows. These precomputed tables, and the deterministic continuous-effect curve
and interval, are plotting inputs rather than estimates produced by the
library.

The `vbar_l2fc_dotplot_column` response-panel case and the vertical-interval
`datapoints_dotplot_column` case read the fixed `synthetic_expression.csv` and
`synthetic_effects.csv` fixtures. Every sample, feature, group assignment,
abundance, effect, and interval is synthetic. The effect intervals are supplied
plotting values and are not estimated from the expression table. Response group,
subtype, and cohort remain constant for each synthetic sample across feature
rows. The latter case reshapes the long expression fixture into the preferred
API's aligned wide expression, observation-metadata, and feature-metadata
tables without changing any values.

Some examples deliberately consume analysis results produced by public library
functions:

| Library-derived result | Gallery consumers |
| --- | --- |
| `diff_test` effects and p-values, via `run_independent_diff_test` | The horizontal p-value `datapoints_dotplot_column` case, `barh_l2fc_dotplot_column`, `barh_dotplot_dotplot_column`, `barh_dotplot_dotplot_dotplot_column`, `barh_4X_dotplot_column`, `l2fc_dotplot_column`, `l2fc_dotplot_single`, `qqplot`, and `volcano_plot_generic`; the deprecated `l2fc_pvalue_dotplot_gex`, `l2fc_pvalue_dotplot_protein_metabolite`, `plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata`, `qqplot_pvalues`, and `volcano_plot_sns_single_comparison_generic` cases reuse the same results through their legacy contracts. |
| `fit_smf_ols_models_and_summarize_adata` coefficients, confidence intervals, p-values, and sample counts | Both `forest` cases, plus the adjusted OLS panels in `barh_dotplot_dotplot_dotplot_column` and `barh_4X_dotplot_column`. |
| `calculate_expectations`, `predict_expectation`, and `excess_expectation` fitted values and residuals | `residual_diagnostic`. |
| `pairwise_spearman_corr_matrix` rank-correlation matrix | Both `plot_heatmap` cases. The three `plot_rank_*` renderers instead calculate their displayed rank summaries from the deterministic ranked lists passed to them. |
| `scipy.stats.hypergeom` upper-tail probability from fixed synthetic universe, hit, and gene-set identifiers | `geneset_enrichment_venn` and the deprecated `geneset_enrichemnt_ol_ven_M_n_N_x`. The maintained API also demonstrates filtering out-of-universe identifiers; every legacy-case identifier is already contained in its supplied universe. |

`run_independent_diff_test` never assigns a sorted or filtered result table
directly to `AnnData.var`. It starts from a copy of the original `AnnData` and
left-joins only result columns on the original variable index. This preserves
the matrix-variable alignment and original variable order; a feature omitted
by `diff_test`, such as the all-zero fixture, remains in `var` with missing
analysis values.

## Coverage boundaries

The manifest labels 35 renderers as maintained, 1 as a compatibility API, and 9
as deprecated. Compatibility and deprecated entries remain in the catalog so
their current call paths and recommended replacements are visible; their
screenshots do not change their support status.

## Regenerate the assets

The `plotting_gallery_params` block in
[`config/config.yaml`](../config/config.yaml) controls the asset directory, log
directory, selected renderer names, selected case IDs, and whether generation
continues after an error. Null renderer and case selections generate all
manifest cases. From the repository root, run:

```bash
bash scripts/000_generate_plotting_gallery.bash
```

The runner sources `config/local_env.sh`, activates `POSTPROCESS_ENV`, configures
headless Matplotlib, writes a runner log under `scripts/logs`, and leaves
analysis and output selection to the Python entry point and YAML config.

## Catalog

The entries, order, case IDs, titles, statuses, replacements, and filenames
below are derived from `RENDERER_MANIFEST`; the manifest remains authoritative
if coverage changes.

### Maintained renderers (35)

| Renderer | Gallery cases |
| --- | --- |
| `adata_histograms`<br><small>`_plotting._histograms`</small> | <a href="assets/plotting_gallery/adata_histograms__subgroup_kde.png"><img src="assets/plotting_gallery/adata_histograms__subgroup_kde.png" alt="Distributions by treatment" width="260"></a><br>`subgroup_kde` — Distributions by treatment<br><br><a href="assets/plotting_gallery/adata_histograms__feature_group_collapse.png"><img src="assets/plotting_gallery/adata_histograms__feature_group_collapse.png" alt="Grouped feature distributions" width="260"></a><br>`feature_group_collapse` — Grouped feature distributions |
| `barh_4X_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/barh_4X_dotplot_column__five_panel.png"><img src="assets/plotting_gallery/barh_4X_dotplot_column__five_panel.png" alt="Expression with four inferential summaries" width="260"></a><br>`five_panel` — Expression with four inferential summaries |
| `barh_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/barh_column__grouped_expression.png"><img src="assets/plotting_gallery/barh_column__grouped_expression.png" alt="Feature abundance by treatment" width="260"></a><br>`grouped_expression` — Feature abundance by treatment |
| `barh_dotplot_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/barh_dotplot_dotplot_column__three_panel.png"><img src="assets/plotting_gallery/barh_dotplot_dotplot_column__three_panel.png" alt="Expression with two differential summaries" width="260"></a><br>`three_panel` — Expression with two differential summaries |
| `barh_dotplot_dotplot_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/barh_dotplot_dotplot_dotplot_column__four_panel.png"><img src="assets/plotting_gallery/barh_dotplot_dotplot_dotplot_column__four_panel.png" alt="Expression with three differential summaries" width="260"></a><br>`four_panel` — Expression with three differential summaries |
| `barh_l2fc_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/barh_l2fc_dotplot_column__two_panel.png"><img src="assets/plotting_gallery/barh_l2fc_dotplot_column__two_panel.png" alt="Expression and differential effect" width="260"></a><br>`two_panel` — Expression and differential effect |
| `datapoints_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/datapoints_dotplot_column__horizontal_pvalue.png"><img src="assets/plotting_gallery/datapoints_dotplot_column__horizontal_pvalue.png" alt="Horizontal distributions with p-value effects" width="260"></a><br>`horizontal_pvalue` — Horizontal distributions with p-value effects<br><br><a href="assets/plotting_gallery/datapoints_dotplot_column__vertical_interval.png"><img src="assets/plotting_gallery/datapoints_dotplot_column__vertical_interval.png" alt="Vertical distributions with supplied intervals" width="260"></a><br>`vertical_interval` — Vertical distributions with supplied intervals |
| `category_composition`<br><small>`_plotting._tabular_plots`</small> | <a href="assets/plotting_gallery/category_composition__percent_annotated.png"><img src="assets/plotting_gallery/category_composition__percent_annotated.png" alt="Response composition by cohort" width="260"></a><br>`percent_annotated` — Response composition by cohort |
| `continuous_effect_plot`<br><small>`_plotting._analytical_plots`</small> | <a href="assets/plotting_gallery/continuous_effect_plot__observed_categories.png"><img src="assets/plotting_gallery/continuous_effect_plot__observed_categories.png" alt="Continuous exposure effect" width="260"></a><br>`observed_categories` — Continuous exposure effect |
| `corr_dotplot`<br><small>`_plotting._corr_dotplots`</small> | <a href="assets/plotting_gallery/corr_dotplot__subgroup_marginals.png"><img src="assets/plotting_gallery/corr_dotplot__subgroup_marginals.png" alt="Correlation with subgroup fits and marginals" width="260"></a><br>`subgroup_marginals` — Correlation with subgroup fits and marginals<br><br><a href="assets/plotting_gallery/corr_dotplot__log1p_identity.png"><img src="assets/plotting_gallery/corr_dotplot__log1p_identity.png" alt="Correlation on synchronized log1p axes" width="260"></a><br>`log1p_identity` — Correlation on synchronized log1p axes |
| `datapoints`<br><small>`_plotting._datapoints`</small> | <a href="assets/plotting_gallery/datapoints__grouped_markers.png"><img src="assets/plotting_gallery/datapoints__grouped_markers.png" alt="Feature values with independent marker and color encodings" width="260"></a><br>`grouped_markers` — Feature values with independent marker and color encodings<br><br><a href="assets/plotting_gallery/datapoints__feature_group_collapse.png"><img src="assets/plotting_gallery/datapoints__feature_group_collapse.png" alt="Collapsed feature-group datapoints" width="260"></a><br>`feature_group_collapse` — Collapsed feature-group datapoints |
| `forest`<br><small>`_plotting._forest`</small> | <a href="assets/plotting_gallery/forest__grouped_estimates.png"><img src="assets/plotting_gallery/forest__grouped_estimates.png" alt="Grouped model estimates" width="260"></a><br>`grouped_estimates` — Grouped model estimates<br><br><a href="assets/plotting_gallery/forest__ratio_scale.png"><img src="assets/plotting_gallery/forest__ratio_scale.png" alt="Ratio-scale effects" width="260"></a><br>`ratio_scale` — Ratio-scale effects |
| `geneset_enrichment_venn`<br><small>`_plotting._venn_plots`</small> | <a href="assets/plotting_gallery/geneset_enrichment_venn__universe_filtered.png"><img src="assets/plotting_gallery/geneset_enrichment_venn__universe_filtered.png" alt="Gene-set enrichment overlap" width="260"></a><br>`universe_filtered` — Gene-set enrichment overlap |
| `kaplan_meier_plot`<br><small>`_plotting._analytical_plots`</small> | <a href="assets/plotting_gallery/kaplan_meier_plot__grouped_risk_censor.png"><img src="assets/plotting_gallery/kaplan_meier_plot__grouped_risk_censor.png" alt="Survival curves with numbers at risk" width="260"></a><br>`grouped_risk_censor` — Survival curves with numbers at risk |
| `l2fc_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/l2fc_dotplot_column__multi_feature.png"><img src="assets/plotting_gallery/l2fc_dotplot_column__multi_feature.png" alt="Differential effects by feature" width="260"></a><br>`multi_feature` — Differential effects by feature |
| `l2fc_dotplot_single`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/l2fc_dotplot_single__single_axis.png"><img src="assets/plotting_gallery/l2fc_dotplot_single__single_axis.png" alt="Differential effect overview" width="260"></a><br>`single_axis` — Differential effect overview |
| `vbar_l2fc_dotplot_column`<br><small>`_plotting._column_plots`</small> | <a href="assets/plotting_gallery/vbar_l2fc_dotplot_column__synthetic_response_panel.png"><img src="assets/plotting_gallery/vbar_l2fc_dotplot_column__synthetic_response_panel.png" alt="Synthetic response-associated expression panel" width="260"></a><br>`synthetic_response_panel` — Synthetic response-associated expression panel |
| `longitudinal_trajectories`<br><small>`_plotting._longitudinal`</small> | <a href="assets/plotting_gallery/longitudinal_trajectories__markers_and_gaps.png"><img src="assets/plotting_gallery/longitudinal_trajectories__markers_and_gaps.png" alt="Longitudinal trajectories with visit gaps" width="260"></a><br>`markers_and_gaps` — Longitudinal trajectories with visit gaps |
| `meta_forest`<br><small>`_plotting._meta_forest`</small> | <a href="assets/plotting_gallery/meta_forest__study_summary_prediction.png"><img src="assets/plotting_gallery/meta_forest__study_summary_prediction.png" alt="Meta-analysis summary" width="260"></a><br>`study_summary_prediction` — Meta-analysis summary |
| `paired_datapoints`<br><small>`_plotting._datapoints`</small> | <a href="assets/plotting_gallery/paired_datapoints__paired_groups.png"><img src="assets/plotting_gallery/paired_datapoints__paired_groups.png" alt="Paired changes by treatment" width="260"></a><br>`paired_groups` — Paired changes by treatment<br><br><a href="assets/plotting_gallery/paired_datapoints__precomputed_pair_values.png"><img src="assets/plotting_gallery/paired_datapoints__precomputed_pair_values.png" alt="Paired values from preserved source matrices" width="260"></a><br>`precomputed_pair_values` — Paired values from preserved source matrices |
| `plot_columns`<br><small>`_plotting._row_plots`</small> | <a href="assets/plotting_gallery/plot_columns__multi_metric.png"><img src="assets/plotting_gallery/plot_columns__multi_metric.png" alt="Grouped covariate summaries" width="260"></a><br>`multi_metric` — Grouped covariate summaries |
| `plot_heatmap`<br><small>`_plotting._corr_dotplots`</small> | <a href="assets/plotting_gallery/plot_heatmap__clustered.png"><img src="assets/plotting_gallery/plot_heatmap__clustered.png" alt="Clustered rank-correlation heatmap" width="260"></a><br>`clustered` — Clustered rank-correlation heatmap<br><br><a href="assets/plotting_gallery/plot_heatmap__fixed_order.png"><img src="assets/plotting_gallery/plot_heatmap__fixed_order.png" alt="Fixed-order rank-correlation heatmap" width="260"></a><br>`fixed_order` — Fixed-order rank-correlation heatmap |
| `plot_rank_heatmap`<br><small>`_plotting._corr_dotplots`</small> | <a href="assets/plotting_gallery/plot_rank_heatmap__rank_hexbin.png"><img src="assets/plotting_gallery/plot_rank_heatmap__rank_hexbin.png" alt="Rank agreement density" width="260"></a><br>`rank_hexbin` — Rank agreement density |
| `plot_rank_scatter`<br><small>`_plotting._corr_dotplots`</small> | <a href="assets/plotting_gallery/plot_rank_scatter__rank_agreement.png"><img src="assets/plotting_gallery/plot_rank_scatter__rank_agreement.png" alt="Rank agreement" width="260"></a><br>`rank_agreement` — Rank agreement |
| `plot_rank_scatter_density`<br><small>`_plotting._corr_dotplots`</small> | <a href="assets/plotting_gallery/plot_rank_scatter_density__rank_density.png"><img src="assets/plotting_gallery/plot_rank_scatter_density__rank_density.png" alt="Rank agreement with local density" width="260"></a><br>`rank_density` — Rank agreement with local density |
| `qqplot`<br><small>`_plotting._plots`</small> | <a href="assets/plotting_gallery/qqplot__differential_pvalues.png"><img src="assets/plotting_gallery/qqplot__differential_pvalues.png" alt="Differential-test p-value QQ plot" width="260"></a><br>`differential_pvalues` — Differential-test p-value QQ plot |
| `ranked_waterfall`<br><small>`_plotting._tabular_plots`</small> | <a href="assets/plotting_gallery/ranked_waterfall__direction_colored.png"><img src="assets/plotting_gallery/ranked_waterfall__direction_colored.png" alt="Ranked feature effects" width="260"></a><br>`direction_colored` — Ranked feature effects |
| `residual_diagnostic`<br><small>`_plotting._tabular_plots`</small> | <a href="assets/plotting_gallery/residual_diagnostic__log_fitted.png"><img src="assets/plotting_gallery/residual_diagnostic__log_fitted.png" alt="Residuals versus fitted abundance" width="260"></a><br>`log_fitted` — Residuals versus fitted abundance |
| `show_colors`<br><small>`_plotting._utils`</small> | <a href="assets/plotting_gallery/show_colors__categorical_palette.png"><img src="assets/plotting_gallery/show_colors__categorical_palette.png" alt="Categorical plotting palette" width="260"></a><br>`categorical_palette` — Categorical plotting palette |
| `show_tol_colors`<br><small>`_plotting._utils`</small> | <a href="assets/plotting_gallery/show_tol_colors__tol_palette.png"><img src="assets/plotting_gallery/show_tol_colors__tol_palette.png" alt="Paul Tol plotting palette" width="260"></a><br>`tol_palette` — Paul Tol plotting palette |
| `spearman_cor_dotplot_2`<br><small>`_plotting._corr_dotplots`</small> | <a href="assets/plotting_gallery/spearman_cor_dotplot_2__dual_hue.png"><img src="assets/plotting_gallery/spearman_cor_dotplot_2__dual_hue.png" alt="Correlation under two categorical encodings" width="260"></a><br>`dual_hue` — Correlation under two categorical encodings |
| `timeseries_paired_datapoints`<br><small>`_plotting._plots`</small> | <a href="assets/plotting_gallery/timeseries_paired_datapoints__faceted_time_series.png"><img src="assets/plotting_gallery/timeseries_paired_datapoints__faceted_time_series.png" alt="Paired time-series datapoints" width="260"></a><br>`faceted_time_series` — Paired time-series datapoints |
| `venn_plot_2list`<br><small>`_plotting._venn_plots`</small> | <a href="assets/plotting_gallery/venn_plot_2list__two_set_overlap.png"><img src="assets/plotting_gallery/venn_plot_2list__two_set_overlap.png" alt="Two feature-set overlap" width="260"></a><br>`two_set_overlap` — Two feature-set overlap |
| `venn_plot_3list`<br><small>`_plotting._venn_plots`</small> | <a href="assets/plotting_gallery/venn_plot_3list__three_set_overlap.png"><img src="assets/plotting_gallery/venn_plot_3list__three_set_overlap.png" alt="Three feature-set overlap" width="260"></a><br>`three_set_overlap` — Three feature-set overlap |
| `volcano_plot_generic`<br><small>`_plotting._plots`</small> | <a href="assets/plotting_gallery/volcano_plot_generic__significance.png"><img src="assets/plotting_gallery/volcano_plot_generic__significance.png" alt="Differential-test volcano plot" width="260"></a><br>`significance` — Differential-test volcano plot<br><br><a href="assets/plotting_gallery/volcano_plot_generic__feature_class.png"><img src="assets/plotting_gallery/volcano_plot_generic__feature_class.png" alt="Volcano plot with feature-class highlighting" width="260"></a><br>`feature_class` — Volcano plot with feature-class highlighting |

### Compatibility renderers (1)

| Renderer | Gallery cases |
| --- | --- |
| `spearman_cor_dotplot`<br><small>`_plotting._corr_dotplots`</small><br>Replacement: `corr_dotplot` | <a href="assets/plotting_gallery/spearman_cor_dotplot__spearman_fit.png"><img src="assets/plotting_gallery/spearman_cor_dotplot__spearman_fit.png" alt="Spearman correlation compatibility API" width="260"></a><br>`spearman_fit` — Spearman correlation compatibility API |

### Deprecated renderers (9)

| Renderer | Gallery cases |
| --- | --- |
| `corr_dotplot_dev`<br><small>`_plotting._corr_dotplots`</small><br>Replacement: `corr_dotplot` | <a href="assets/plotting_gallery/corr_dotplot_dev__replacement_smoke.png"><img src="assets/plotting_gallery/corr_dotplot_dev__replacement_smoke.png" alt="Deprecated correlation wrapper" width="260"></a><br>`replacement_smoke` — Deprecated correlation wrapper |
| `geneset_enrichemnt_ol_ven_M_n_N_x`<br><small>`_plotting._venn_plots`</small><br>Replacement: `geneset_enrichment_venn` | <a href="assets/plotting_gallery/geneset_enrichemnt_ol_ven_M_n_N_x__replacement_smoke.png"><img src="assets/plotting_gallery/geneset_enrichemnt_ol_ven_M_n_N_x__replacement_smoke.png" alt="Legacy gene-set enrichment overlap" width="260"></a><br>`replacement_smoke` — Legacy gene-set enrichment overlap |
| `l2fc_pvalue_dotplot_gex`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `l2fc_dotplot_single` | <a href="assets/plotting_gallery/l2fc_pvalue_dotplot_gex__replacement_smoke.png"><img src="assets/plotting_gallery/l2fc_pvalue_dotplot_gex__replacement_smoke.png" alt="Legacy gene-expression effect dotplot" width="260"></a><br>`replacement_smoke` — Legacy gene-expression effect dotplot |
| `l2fc_pvalue_dotplot_protein_metabolite`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `l2fc_dotplot_single` | <a href="assets/plotting_gallery/l2fc_pvalue_dotplot_protein_metabolite__replacement_smoke.png"><img src="assets/plotting_gallery/l2fc_pvalue_dotplot_protein_metabolite__replacement_smoke.png" alt="Legacy protein/metabolite effect dotplot" width="260"></a><br>`replacement_smoke` — Legacy protein/metabolite effect dotplot |
| `plot_column_of_bar_h_2groups_GEX_adata`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `barh_column` | <a href="assets/plotting_gallery/plot_column_of_bar_h_2groups_GEX_adata__replacement_smoke.png"><img src="assets/plotting_gallery/plot_column_of_bar_h_2groups_GEX_adata__replacement_smoke.png" alt="Legacy grouped expression bars" width="260"></a><br>`replacement_smoke` — Legacy grouped expression bars |
| `plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `barh_l2fc_dotplot_column` | <a href="assets/plotting_gallery/plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata__replacement_smoke.png"><img src="assets/plotting_gallery/plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata__replacement_smoke.png" alt="Legacy expression and effect composite" width="260"></a><br>`replacement_smoke` — Legacy expression and effect composite |
| `plot_paired_point_anndata`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `timeseries_paired_datapoints` | <a href="assets/plotting_gallery/plot_paired_point_anndata__replacement_smoke.png"><img src="assets/plotting_gallery/plot_paired_point_anndata__replacement_smoke.png" alt="Legacy paired time-series datapoints" width="260"></a><br>`replacement_smoke` — Legacy paired time-series datapoints |
| `qqplot_pvalues`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `qqplot` | <a href="assets/plotting_gallery/qqplot_pvalues__replacement_smoke.png"><img src="assets/plotting_gallery/qqplot_pvalues__replacement_smoke.png" alt="Legacy p-value QQ plot" width="260"></a><br>`replacement_smoke` — Legacy p-value QQ plot |
| `volcano_plot_sns_single_comparison_generic`<br><small>`_plotting._plots_depreciated`</small><br>Replacement: `volcano_plot_generic` | <a href="assets/plotting_gallery/volcano_plot_sns_single_comparison_generic__replacement_smoke.png"><img src="assets/plotting_gallery/volcano_plot_sns_single_comparison_generic__replacement_smoke.png" alt="Legacy adjusted-p volcano plot" width="260"></a><br>`replacement_smoke` — Legacy adjusted-p volcano plot |
