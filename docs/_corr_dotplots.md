# `_corr_dotplots`

Correlation-scatter and rank-comparison plotting helpers from `_plotting/_corr_dotplots.py`.

This is the most test-backed plotting module in the package. Direct regression tests live in `tests/test_corr_dotplots.py`.

## Main entry points

- `corr_dotplot`
- `corr_dotplot_dev` (deprecated compatibility wrapper)
- `spearman_cor_dotplot`
- `spearman_cor_dotplot_2`
- `plot_rank_scatter`
- `plot_rank_heatmap`
- `plot_rank_scatter_density`
- `pairwise_spearman_corr_matrix`
- `compare_ranked_lists`
- `plot_heatmap`

## `corr_dotplot`

`corr_dotplot(...)` is the primary public API for correlation scatter plots, subgroup fit lines, and optional x/y marginal histograms.

### Full signature

```python
def corr_dotplot(
    df: pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    layer: str | None = None,
    x_df: Any | None = None,
    var_df: pd.DataFrame | None = None,
    obs_df: pd.DataFrame | None = None,
    column_key_x: str | None = None,
    column_key_y: str | None = None,
    hue: str | None = None,
    subset_key: str | None = None,
    figsize: tuple[float, float] = (20, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    axes_title: str | None = None,
    axes_lines: bool = True,
    show_y_intercept: bool = True,
    palette: Sequence[Any] | str | None = palettes.godsnot_102,
    subset_palette: Sequence[Any] | str | None = None,
    dot_size: float = 200,
    title_fontsize: int = 20,
    stats_fontsize: int | None = None,
    axes_title_y: float | None = None,
    axis_label_fontsize: int = 20,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    fit_legend_bbox_to_anchor: Sequence[float] | None = None,
    hue_legend_bbox_to_anchor: Sequence[float] | None = None,
    show_all_obs_fit: bool = False,
    show_fit: bool = True,
    show_fit_legend: bool = True,
    show_hue_legend: bool = True,
    show_stats_text: bool = True,
    show_identity_line: bool = False,
    identity_line_label: str | None = "Identity",
    identity_line_style: Mapping[str, Any] | None = None,
    identity_limits: Literal["shared_axes", "data"] = "shared_axes",
    nas2zeros: bool = False,
    dropna: bool = False,
    dropzeros: bool = False,
    method: Literal["spearman", "pearson"] = "pearson",
    show_x_marginal_hist: bool = False,
    show_y_marginal_hist: bool = False,
    x_marginal_hist_bins: int | Sequence[float] = 20,
    y_marginal_hist_bins: int | Sequence[float] = 20,
    x_marginal_hist_fill: bool = True,
    x_marginal_hist_KDE: bool = True,
    y_marginal_hist_fill: bool = True,
    y_marginal_hist_KDE: bool = True,
    show_all_obs_x_hist: bool = False,
    show_all_obs_y_hist: bool = False,
    x_marginal_hist_height_ratio: float = 0.18,
    y_marginal_hist_width_ratio: float = 0.18,
    xscale: str = "linear",
    yscale: str = "linear",
    xlims: Sequence[float] | None = None,
    ylims: Sequence[float] | None = None,
    xlim_padding_fraction: float | None = None,
    ylim_padding_fraction: float | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes | dict[str, plt.Axes | None],
    Any,
    float,
    float,
]:
```

```python
import adata_science_tools as adtl

fig, axes, fit, corr_value, corr_pvalue = adtl.corr_dotplot(
    adata=adata,
    layer="log1p",
    column_key_x="geneA",
    column_key_y="geneB",
    hue="Treatment",
    subset_key="Batch",
    method="spearman",
    show_x_marginal_hist=True,
    show_y_marginal_hist=True,
    show=False,
)

main_ax = axes["main"]
top_hist_ax = axes["x_marginal"]
right_hist_ax = axes["y_marginal"]
```

### Supported input modes

- `df=...` with preassembled plotting columns
- `adata=...` with an optional `layer`
- `x_df` plus `obs_df`, with optional `var_df` when feature columns need names

When `df` is provided, the AnnData-derived path is ignored. `column_key_x` and `column_key_y` are required and must resolve to numeric columns after the final plotting table is assembled.

### Return value

`corr_dotplot(...)` always returns `(fig, axes, fit, corr_value, corr_pvalue)`.

- With both marginal flags disabled, `axes` is the single scatter `Axes`, preserving the original public contract.
- With either marginal enabled, `axes` is a dictionary containing `"main"`, `"x_marginal"`, and `"y_marginal"`; a disabled marginal entry is `None`.
- `fit`, `corr_value`, and `corr_pvalue` always describe the overall filtered observations, including when subgroup fits are drawn.

### Correlation, filtering, and fit behavior

- `method` must be `"pearson"` or `"spearman"`.
- `nas2zeros`, `dropna`, and `dropzeros` control x/y cleanup before statistics.
- `palette` colors `hue`-driven scatter points, while `subset_palette` colors `subset_key`-driven fit and marginal layers.
- `subset_key` draws one fit line per subgroup when fitting succeeds.
- Subgroup order follows categorical order for categorical columns, ascending order for non-categorical numeric columns, and first-seen order for other columns.
- `show_all_obs_fit=True` adds the overall fit line in subset mode.
- `show_fit=False` hides fit artists without changing returned fit or correlation statistics.
- `show_identity_line=True` draws y=x over the shared visible interval or combined finite data range.
- `xscale` and `yscale` accept `"linear"`, `"log"`, `"log2"`, and `"log1p"`; enabled marginals use the same scale and limits as the main axes.
- `"log"` uses Matplotlib's default logarithmic base, while `"log2"` uses a base-2 logarithmic scale.
- `"log1p"` uses `log1p`/`expm1` function scaling, accepts finite values strictly greater than -1, and keeps the legacy origin line valid.
- On nonlinear axes, fit and identity artists sample their raw-coordinate linear relation densely; correlation and regression statistics still use the untransformed values.
- Automatic and fractionally padded `"log1p"` ranges include the data plus active fit endpoints and axis-reference values, including a legacy origin line, before applying transformed-space padding that stays above -1.
- Explicit limits still win; an explicit reference sequence replaces the legacy origin on that axis, and an empty sequence draws none. Reference specifications other than `None` must be sequences of mappings.
- Explicit limits take precedence over padding; padding is calculated in the configured transformed space for all three logarithmic modes.
- Data, limits, references, fit endpoints, and identity coordinates must satisfy each configured scale's domain.
- When identity coordinates span two transformed axes, the stricter lower domain bound applies.
- Reference entries are drawn in caller order and accept `value`, `label`, `color`, `linestyle`, `linewidth`, `alpha`, and `zorder`.
- `show_fit_legend`, `show_hue_legend`, and `show_stats_text` are independent display controls.
- `fit_legend_bbox_to_anchor` and `hue_legend_bbox_to_anchor` accept 2-item or 4-item sequences.
- If no valid subgroup values remain, `corr_dotplot(...)` falls back to the all-data fit and statistics; enabled marginals also show the all-data distributions.
- `show=False` closes the figure before returning so notebook backends do not auto-render it.

### Marginal histogram behavior

- The supported layouts are no marginals, x-only above the scatter, y-only to the right, and both marginals.
- Marginals always use the same filtered observations as the scatter and returned statistics.
- Without `subset_key`, each enabled marginal draws one overall histogram.
- With `subset_key`, marginal grouping follows `subset_key`, not `hue`, and uses `subset_palette`.
- `show_all_obs_x_hist` and `show_all_obs_y_hist` add muted all-observation overlays in subset mode.
- Bin arguments accept an integer count or explicit one-dimensional edge sequences. Explicit edges for an enabled marginal must contain at least two finite, strictly increasing values within the configured scale's domain.
- Fill and KDE overlays can be controlled independently for x and y.
- `x_marginal_hist_height_ratio` and `y_marginal_hist_width_ratio` control marginal panel size relative to the main panel.
- When an x marginal is enabled, `axes_title` belongs to `axes["x_marginal"]`; otherwise it belongs to the main axes.
- Default legends move beyond the right marginal when one is present; explicit legend anchors still win.

### Tested behaviors

Regression tests cover observation/feature name collisions, generic and sparse matrix extraction, full matrix-shape validation, styling controls, optional footer text, subgroup fit failures, independent legends, separate palettes, numeric subgroup ordering, all four layouts, grouped and all-observation marginals, filtering parity, integer and explicit bins, fill/KDE controls, subplot ratios, title ownership, footer spacing, legend placement, main-limit preservation, the empty-subset fallback, Spearman forwarding, early scale validation, and synchronized `log`, `log2`, and `log1p` axes.

### Repo example with simulated data

The config-driven example uses `corr_dotplot(...)` with both marginals enabled:

- [`example_simulated_data/config/config.yaml`](../example_simulated_data/config/config.yaml) contains the marginal controls.
- [`example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py`](../example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py) loads the simulated AnnData and creates the plot.
- The generated figure is available at [`baseline.png`](../example_simulated_data/results/plot_dotplot_simulate_1_var_covar_age/baseline/baseline.png).

## `corr_dotplot_dev`

`corr_dotplot_dev(...)` is deprecated. It emits `DeprecationWarning` and forwards all arguments to `corr_dotplot(...)`.

For compatibility, its second return value is always the three-key axes dictionary, including when both marginals are disabled. New code should use `corr_dotplot(...)` and follow the conditional return contract above.

## `spearman_cor_dotplot`

`spearman_cor_dotplot(...)` is a backward-compatible wrapper around `corr_dotplot(...)`.

### Full signature

```python
def spearman_cor_dotplot(*args, **kwargs):
```

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

#### Full signature

```python
def spearman_cor_dotplot_2(df, column_key_x, column_key_y, hue, hue_right, figsize=(20, 10)):
```

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
