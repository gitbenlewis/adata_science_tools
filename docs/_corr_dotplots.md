# `_corr_dotplots`

Correlation-scatter and rank-comparison plotting helpers from `_plotting/_corr_dotplots.py`.

This is the most test-backed plotting module in the package. Direct regression tests live in `tests/test_corr_dotplots.py`.

## Main entry points

- `corr_dotplot`
- `corr_dotplot_dev`
- `spearman_cor_dotplot`
- `spearman_cor_dotplot_2`
- `plot_rank_scatter`
- `plot_rank_heatmap`
- `plot_rank_scatter_density`
- `pairwise_spearman_corr_matrix`
- `compare_ranked_lists`
- `plot_heatmap`

## `corr_dotplot`

`corr_dotplot(...)` is the primary public API.

```python
import adata_science_tools as adtl

fig, ax, fit, corr_value, corr_pvalue = adtl.corr_dotplot(
    adata=adata,
    layer="log1p",
    column_key_x="geneA",
    column_key_y="geneB",
    hue="Treatment",
    subset_key="Batch",
    method="spearman",
    show=False,
)
```

### Supported input modes

- `df=...` with preassembled plotting columns
- `adata=...` with optional `layer`
- `x_df` plus `obs_df`, with optional `var_df` when feature columns need names

When `df` is provided, the `AnnData`-derived path is ignored.

### Required arguments

- `column_key_x`
- `column_key_y`

These must resolve to numeric columns after the final plotting table is assembled.

### Return value

`corr_dotplot(...)` returns:

- `fig`
- `axes`
- `fit`
- `corr_value`
- `corr_pvalue`

The returned `fit`, `corr_value`, and `corr_pvalue` always describe the overall filtered observations, even when `subset_key` is used to draw subgroup fits.

### Important behavior

- `method` must be `"pearson"` or `"spearman"`.
- `nas2zeros`, `dropna`, and `dropzeros` control x/y cleanup before statistics.
- `palette` colors `hue`-driven scatter points, while `subset_palette` colors `subset_key`-driven subgroup fit lines.
- `subset_key` draws one fit line per subgroup when fitting succeeds.
- `show_all_obs_fit=True` adds the overall fit line in subset mode.
- `show_fit_legend` and `show_hue_legend` can be toggled independently.
- `fit_legend_bbox_to_anchor` and `hue_legend_bbox_to_anchor` accept either 2-item or 4-item sequences.
- `show=False` closes the figure before returning so notebook backends do not auto-render it.
- There is no built-in `savefig` argument; callers save the returned figure themselves.

### Tested behaviors

The current regression tests lock in several details:

- Observation columns that collide with feature names are renamed to `<name>_obs` before concatenation.
- Public styling kwargs such as `dot_size`, `title_fontsize`, `stats_fontsize`, `axes_title_y`, `axis_label_fontsize`, `tick_label_fontsize`, and `legend_fontsize` affect the final figure.
- `show_stats_text=False` suppresses the footer text without suppressing the returned statistics.
- Non-categorical `subset_key` columns are supported.
- Subsets that cannot produce a fit are reported as `fit unavailable` in the footer and omitted from the fit legend.
- The fit legend title changes with method, for example `batch fit\nPearson_corr` versus `batch fit\nSpearman_corr`.

## `corr_dotplot_dev`

`corr_dotplot_dev(...)` is the experimental variant that adds optional top and right marginal histograms while mostly keeping the same filtered-data correlation and fit behavior as `corr_dotplot(...)`.

```python
fig, axes, fit, corr_value, corr_pvalue = adtl.corr_dotplot_dev(
    df=df,
    column_key_x="Age",
    column_key_y="CRP",
    hue="Outcome",
    subset_key="Batch",
    show_x_marginal_hist=True,
    show_y_marginal_hist=True,
    show=False,
)

main_ax = axes["main"]
top_hist_ax = axes["x_marginal"]
right_hist_ax = axes["y_marginal"]
```

Important behavior:

- It accepts the full `corr_dotplot(...)` keyword API, including `palette` for `hue` layers and `subset_palette` for `subset_key` layers, plus `show_x_marginal_hist`, `show_y_marginal_hist`, `x_marginal_hist_bins`, `y_marginal_hist_bins`, `x_marginal_hist_fill`, `x_marginal_hist_KDE`, `y_marginal_hist_fill`, `y_marginal_hist_KDE`, `show_all_obs_x_hist`, `show_all_obs_y_hist`, `x_marginal_hist_height_ratio`, and `y_marginal_hist_width_ratio`.
- It returns `(fig, axes_dict, fit, corr_value, corr_pvalue)` instead of a single scatter axes.
- `axes_dict` always contains `"main"`, `"x_marginal"`, and `"y_marginal"` keys, with disabled marginal panels set to `None`.
- Marginals use the same filtered `working_df` as the scatter, correlation, and fit layers.
- When `subset_key` is provided, marginal histograms follow `subset_key` rather than `hue`, so their subgroup colors follow `subset_palette`.
- When `subset_key` is provided but no valid subgroup values remain after filtering, `corr_dotplot_dev(...)` falls back to the overall fit, footer stats, and all-observation marginals even if `show_all_obs_fit`, `show_all_obs_x_hist`, or `show_all_obs_y_hist` were `False`.
- `corr_dotplot(...)` keeps its current behavior in that edge case and does not apply this dev-only fallback.
- Marginals default to `fill=True` and `KDE=True` and can be turned off independently for the x and y panels.
- When `show_x_marginal_hist=True`, `axes_title` is attached to `axes["x_marginal"]` so the title sits above the full composite plot; otherwise it remains on `axes["main"]`.
- `corr_dotplot_dev(method="spearman", ...)` is the supported Spearman path for the dev API; there is no separate `spearman_cor_dotplot_dev(...)` wrapper in this change.

### Repo example with simulated data

The repository now includes a config-driven example that uses `corr_dotplot_dev(...)` end to end:

- [`example_simulated_data/scripts/simulate_1_var_covar_age.py`](../example_simulated_data/scripts/simulate_1_var_covar_age.py) creates a baseline `AnnData` object with `obs` columns `Age` and `case_control` and one feature, `simulated_feature`.
- [`example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py`](../example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py) loads that `.h5ad` file and calls `corr_dotplot_dev(...)` with `column_key_x="Age"`, `column_key_y="simulated_feature"`, `hue="case_control"`, and `subset_key="case_control"`.
- The default example uses the explicit palette `['#AA4499', '#332288', '#1f77b4', '#661100', '#117733', '#ff7f0e']` from the YAML config rather than `adtl.palettes`.

The generated example figure is available at [`baseline.png`](../example_simulated_data/results/plot_dotplot_simulate_1_var_covar_age/baseline/baseline.png).

## `spearman_cor_dotplot`

`spearman_cor_dotplot(...)` is a backward-compatible wrapper around `corr_dotplot(...)`.

```python
fig, ax, fit, corr_value, corr_pvalue = adtl.spearman_cor_dotplot(
    df,
    "x",
    "y",
    "group",
    show=False,
)
```

Important behavior:

- It forces `method="spearman"` even if a different method is passed.
- It still supports the older positional calling style `df, column_key_x, column_key_y, hue, ...`.

## Secondary helpers

### `spearman_cor_dotplot_2`

Creates a two-panel scatter plot of the same x/y pair with two different hue columns. It returns `(figure, axes)`.

### Rank-comparison helpers

- `plot_rank_scatter`, `plot_rank_heatmap`, and `plot_rank_scatter_density` compare two ranked lists using the ranks of shared elements.
- They return `(correlation, p_value)`.
- If the lists share no common elements, they return `(None, None)`.

### `pairwise_spearman_corr_matrix`

Builds a pairwise Spearman correlation matrix across a dictionary of ranked lists.

### `compare_ranked_lists`

Computes the Spearman correlation between two ranked lists without plotting and returns `(correlation, p_value)`, or `(None, None)` when there is no overlap.

### `plot_heatmap`

Heatmap helper for numeric matrices or `DataFrame` input. This function is documented from code rather than regression tests; use it as a convenience helper rather than a heavily stabilized API.

## Example with plain `DataFrame`

```python
fig, ax, fit, corr_value, corr_pvalue = adtl.corr_dotplot(
    df=df,
    column_key_x="Age",
    column_key_y="CRP",
    hue="Outcome",
    show=False,
)
fig.savefig("results/age_crp_corr.png", dpi=300, bbox_inches="tight")
```
