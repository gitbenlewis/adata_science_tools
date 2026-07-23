# Plotting API updates

This release implements the public plotting roadmap additively. Existing function defaults and tuple shapes remain unchanged unless an explicit new keyword requests new behavior.

## Compatibility summary

| Public API | Existing behavior retained | Additive controls or outputs |
|---|---|---|
| `corr_dotplot` | Filtering, correlation/fit values, subgroup fits, optional marginals, conditional axes return | Hidden fit artists, identity line, linear/log/log2/log1p scales, limits/padding, ordered axis references |
| `adata_histograms` | Input selection, collapse modes, filters, histograms/KDE, mean/zero lines, two-item return | Subgroup eligibility/metrics, mapping palettes, KDE tuning and underfill, styled and ordered references |
| `datapoints` | Input selection, filters, faceting, jitter, boxes/violins, metric legends, three-item return | Summary-only filters, markers, annotations, per-metric legend formats, log scale, ordered references |
| `ranked_waterfall` | New API | Stable ranked bars and returned ranked rows |
| `category_composition` | New API | Ordered stacked composition and returned wide table |
| `residual_diagnostic` | New API | Supplied-residual coordinates without model fitting |
| `longitudinal_trajectories` | New API; `paired_datapoints` remains unchanged | Multi-timepoint exact/display values and auditable segment tuples |
| `kaplan_meier_plot` | New precomputed-data API | Supplied post-step curves, confidence bands, censors, aligned risk table, and normalized audit tables without survival fitting |
| `continuous_effect_plot` | New precomputed-data API | Supplied effect curve, confidence band, optional styled observations, references, external-axis composition, and normalized audit tables without model fitting |

## Existing-function argument classification

Every argument is classified below. “Existing” means the argument and default predate this roadmap; “new” means this release added the keyword. A type expansion is called out separately.

### `corr_dotplot`

New arguments: `show_fit`, `show_identity_line`, `identity_line_label`, `identity_line_style`, `identity_limits`, `xscale`, `yscale`, `xlims`, `ylims`, `xlim_padding_fraction`, `ylim_padding_fraction`, `x_reference_lines`, and `y_reference_lines`.

`xscale` and `yscale` accept `"linear"`, base-10 `"log"`, base-2 `"log2"`, and `"log1p"` (forward `log1p`, inverse `expm1`). Nonlinear axes densely sample raw-coordinate fit and identity relations. Automatic `"log1p"` limits include active fit endpoints and reference values before transformed-space padding; explicit limits win, and explicit reference sequences replace legacy origins. Correlation and linear-fit calculations remain on the untransformed input values.

Existing arguments: `df`, `adata`, `layer`, `x_df`, `var_df`, `obs_df`, `column_key_x`, `column_key_y`, `hue`, `subset_key`, `figsize`, `xlabel`, `ylabel`, `axes_title`, `axes_lines`, `show_y_intercept`, `palette`, `subset_palette`, `dot_size`, `title_fontsize`, `stats_fontsize`, `axes_title_y`, `axis_label_fontsize`, `tick_label_fontsize`, `legend_fontsize`, `fit_legend_bbox_to_anchor`, `hue_legend_bbox_to_anchor`, `show_all_obs_fit`, `show_fit_legend`, `show_hue_legend`, `show_stats_text`, `nas2zeros`, `dropna`, `dropzeros`, `method`, `show_x_marginal_hist`, `show_y_marginal_hist`, `x_marginal_hist_bins`, `y_marginal_hist_bins`, `x_marginal_hist_fill`, `x_marginal_hist_KDE`, `y_marginal_hist_fill`, `y_marginal_hist_KDE`, `show_all_obs_x_hist`, `show_all_obs_y_hist`, `x_marginal_hist_height_ratio`, `y_marginal_hist_width_ratio`, and `show`.

### `adata_histograms`

New arguments: `subset_min_count`, `subset_small_group_policy`, `subset_legend_metrics`, `subset_label_format`, `zero_line_style`, `mean_line_style`, `x_reference_lines`, `kde_fill`, `kde_fill_alpha`, `kde_bw_method`, `kde_grid_points`, and `kde_clip`. The existing `subset_palette` argument now also accepts mappings.

Existing arguments: `adata`, `df`, `var_df`, `var_names`, `var_groupby_key`, `collapse_mode`, `collapse_func`, `ref_values_obsm_key`, `layer`, `use_raw`, `filter_vars_by_isin_lists`, `filter_obs_by_isin_lists`, `subset_obs_key`, `subset_order`, `palette`, `subset_palette`, `show_all_obs_hist`, `all_obs_color`, `all_obs_alpha`, `ncols`, `figsize`, `sharex`, `xlims`, `add_zero_line`, `add_mean_line`, `add_mean_to_legend`, `highlight_negative_mean_legend`, `bins`, `binwidth`, `binrange`, `stat`, `multiple`, `element`, `fill`, `kde`, `common_bins`, `common_norm`, `discrete`, `cumulative`, `alpha`, `color`, `xlabel`, `ylabel`, `title`, `subplot_title_var_col`, `title_fontsize`, `axis_label_fontsize`, `tick_label_fontsize`, `legend_fontsize`, `legend_loc`, `legend_bbox_to_anchor`, `legend`, `dropna`, `nas2zeros`, `dropzeros`, and `show`.

### `datapoints`

New arguments: `summary_filter_obs_by_isin_lists`, `marker_by_obs_key`, `marker_order`, `marker_styles`, `legend_metric_formats`, `group_annotations`, `yscale`, `y_reference_lines`, `append_marker_handles_to_legend`, and `append_reference_handles_to_legend`. The existing `subset_palette` argument now also accepts mappings.

Existing arguments: `input_data`, `adata`, `df`, `var_df`, `var_names`, `var_groupby_key`, `collapse_mode`, `collapse_func`, `layer`, `use_raw`, `filter_vars_by_isin_lists`, `filter_obs_by_isin_lists`, `subset_obs_key`, `subset_order`, `subplot_by_obs_key`, `subplot_by_var_key`, `subplot_by_var_missing_label`, `subplot_order`, `x_order`, `x_order_include_unobserved`, `x_by_obs_key`, `x_by_obs_missing_label`, `x_by_obs_multi_var_mode`, `palette`, `subset_palette`, `color`, `jitter_amount`, `random_seed`, `point_size`, `point_alpha`, `boxplot`, `boxplot_width`, `boxplot_showfliers`, `violinplot`, `violin_width`, `violin_alpha`, `legend_metrics`, `show_all_data_metrics`, `highlight_negative_mean_legend`, `ncols`, `figsize`, `sharey`, `ylims`, `add_zero_line`, `xlabel`, `ylabel`, `title`, `title_fontsize`, `axis_label_fontsize`, `tick_label_fontsize`, `legend_fontsize`, `legend_loc`, `legend_bbox_to_anchor`, `legend_scope`, `legend`, `dropna`, `nas2zeros`, `dropzeros`, `show`, `savefig`, `file_name`, `logger`, `log_level`, `allow_unused_params`, and `params`.

## New-API argument classification

All arguments of these new functions are new in this release:

- `ranked_waterfall`: `df`, `value`, `label`, `color_by`, `color_order`, `palette`, `ascending`, `tie_breaker`, `allow_duplicate_labels`, `y_reference_lines`, `bar_width`, `bar_alpha`, `xlabel`, `ylabel`, `title`, `tick_rotation`, `tick_fontsize`, `legend_title`, `legend_kwargs`, `figsize`, and `show`.

- `category_composition`: `df`, `x`, `category`, `x_order`, `category_order`, `palette`, `normalize`, `include_unobserved_x`, `include_unobserved_categories`, `missing_category`, `missing_label`, `annotate`, `annotation_format`, `xlabel`, `ylabel`, `title`, `legend_title`, `legend_kwargs`, `figsize`, and `show`.

- `residual_diagnostic`: `df`, `x`, `residual`, `x_transform`, `y_reference_lines`, `point_color`, `point_size`, `point_alpha`, `xlabel`, `ylabel`, `title`, `figsize`, `dropna`, and `show`.

- `longitudinal_trajectories`: `df`, `x`, `y`, `subject`, `x_order`, `display_y`, `line_eligible`, `connect`, `line_color_by`, `point_color_by`, `color_order`, `palette`, `marker_by`, `marker_order`, `marker_styles`, `line_color`, `line_width`, `line_alpha`, `point_size`, `point_alpha`, `x_jitter`, `random_seed`, `yscale`, `ylims`, `y_reference_lines`, `xlabel`, `ylabel`, `title`, `figsize`, `color_legend_title`, `marker_legend_title`, `color_legend_kwargs`, `marker_legend_kwargs`, `dropna_display`, and `show`.

- `kaplan_meier_plot`: `curve_df`, `risk_table_df`, `censor_df`, `time`, `survival`, `ci_lower`, `ci_upper`, `group`, `risk_time`, `risk_count`, `group_order`, `palette`, `ci_alpha`, `censor_marker`, `censor_size`, `xlabel`, `ylabel`, `title`, `legend_title`, `legend_labels`, `figsize`, and `show`.

- `continuous_effect_plot`: `curve_df`, `x`, `estimate`, `ci_lower`, `ci_upper`, `observed_df`, `observed_x`, `observed_y`, `observed_category`, `observed_order`, `observed_styles`, `line_color`, `ci_alpha`, `xscale`, `ylims`, `y_reference_lines`, `xlabel`, `ylabel`, `title`, `annotation`, `annotation_xy`, `ax`, `figsize`, and `show`.

## Shared reference contract

Reference entries accept `value` plus optional `label`, `color`, `linestyle`, `linewidth`, `alpha`, and `zorder`. Entries are drawn and added to legends in caller order. Exact numeric duplicates of applicable legacy or earlier configured lines are drawn once; nearby unequal thresholds remain distinct. Passing an explicit empty sequence draws no configured lines, while `None` preserves the applicable legacy fallback.
