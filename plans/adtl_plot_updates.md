# Public plotting feature roadmap for `adata_science_tools`

Status: stages 1 through 5 implemented

## Implementation outcome

1. All five stages in this roadmap are implemented in the current public plotting API.

2. `adtl.corr_dotplot()` is the sole correlation-scatter entry point and includes marginal panels, fit visibility, identity lines, scales, limits, padding, and axis-specific reference lines.

3. `adtl.adata_histograms()` and `adtl.datapoints()` include the additive controls specified below, including KDE underfill and per-metric legend formatting, without changing their existing defaults or returns.

4. `adtl.ranked_waterfall()`, `adtl.category_composition()`, `adtl.residual_diagnostic()`, `adtl.longitudinal_trajectories()`, `adtl.kaplan_meier_plot()`, and `adtl.continuous_effect_plot()` are exported through both the package root and plotting namespace.

5. The implemented baseline is deterministic, documented, and covered by focused tests. No development correlation alias or separate concordance function is part of the public API.

6. Stage 5 completes the remaining public APIs required by downstream analytical pipelines: KDE underfill, custom legend metric formatting, precomputed Kaplan-Meier rendering, and precomputed continuous-effect rendering without inferential fitting.

## 1. Purpose

Expand the public `adata_science_tools` plotting API so common analytical figures can be produced without handwritten Matplotlib layers or post-call axis manipulation.

This roadmap prefers extending an existing plotting function whenever its current purpose and return contract are compatible with the requested behavior. A new public function is proposed only when reusing an existing function would blur its meaning or make its signature unmanageably complex.

The current correlation function is `adtl.corr_dotplot()` (singular). It now includes the optional marginal-histogram functionality that previously lived in a separate development entry point. This roadmap treats `corr_dotplot()` as the only supported correlation-scatter API.

## 2. Existing-function-first decisions

| Plot requirement | Existing function considered | Decision |
|---|---|---|
| Scatter with correlation, optional marginals, optional fit, identity line, scales, limits, and reference lines | `adtl.corr_dotplot()` | Preserve the newly promoted marginal API and extend the existing function |
| Single or grouped histogram/KDE | `adtl.adata_histograms()` | Extend existing function |
| Categorical points with boxes/violins, markers, summaries, and annotations | `adtl.datapoints()` | Extend existing function |
| Two-condition paired points | `adtl.paired_datapoints()` | Retain without changing its two-condition semantics |
| Multi-timepoint subject trajectories | `adtl.paired_datapoints()` | Add a new function because multi-timepoint gap and display-value behavior do not fit the paired API cleanly |
| Ranked vertical waterfall bars | Existing column and row plot helpers | Add a new focused function; the existing helpers have different orientation and semantics |
| Ordered stacked category composition | Existing column plot helpers | Add a new focused function; no existing API returns the required composition table |
| Supplied-residual diagnostic | `adtl.corr_dotplot()` | Add a new focused function so residual plots do not compute or imply correlation/regression inference |
| KDE curve underfill in histogram panels | `adtl.adata_histograms()` | Extend the existing histogram function with additive fill controls |
| Per-metric formatting in categorical summary legends | `adtl.datapoints()` | Extend the existing metric legend contract with a validated format mapping |
| Precomputed Kaplan-Meier curve, censor, confidence-band, and risk-table display | Existing tabular plot helpers | Add a focused renderer that performs no survival fitting |
| Precomputed continuous-effect curve with confidence band and observed-data layer | Existing correlation and tabular plot helpers | Add a focused renderer that performs no model fitting or prediction |

### 2.1 Latest baseline incorporated

1. `adtl.corr_dotplot()` already accepts x and y marginal-histogram controls, including bins, fill, KDE, all-observation overlays, and panel ratios.

2. With both marginal panels disabled, the second return value remains a single `plt.Axes` for backward compatibility.

3. With either marginal panel enabled, the second return value is `{"main": ..., "x_marginal": ..., "y_marginal": ...}`, and any disabled panel is `None`.

4. Marginals use the same filtered observations as the scatter and follow `subset_key` grouping and `subset_palette` ordering.

5. No separate development function, marginal alias, or wrapper is part of this roadmap.

## 3. Design principles

1. Preserve current defaults and tuple shapes for existing functions.

2. Add new behavior through keyword-only arguments.

3. Keep sorting, jitter, palettes, and legends deterministic.

4. Keep category definitions, transformations, group eligibility, reference values, and display substitutions caller-controlled.

5. Do not add a dependency.

6. Avoid full-matrix densification for AnnData input.

7. Return prepared plot data from new tabular functions so every rendered observation is auditable.

8. Keep inferential model fitting outside generic plots, except for the existing documented correlation behavior.

## 4. Stage 1: simple tabular plots

Stage 1 updates `adtl.corr_dotplot()` and adds three plot types that do not have a semantically compatible existing function.

### 4.1 Update existing `adtl.corr_dotplot()`

Use `corr_dotplot` for both standard correlation scatters and descriptive identity-line scatters. The fit and statistics are still calculated and returned for backward compatibility, but their artists can be hidden.

#### Proposed full updated signature

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

#### Baseline versus remaining additions

The marginal parameters from `show_x_marginal_hist` through `y_marginal_hist_width_ratio` are already present in the latest baseline and are retained unchanged. The remaining proposed parameters are `show_fit`, the four identity-line controls, the scale and limit controls, the two padding controls, and the axis-specific reference-line controls.

#### Remaining proposed argument behavior

1. The existing marginal arguments and their defaults remain unchanged from the latest baseline.

2. `show_fit=False` suppresses fit-line artists but does not change the returned fit and correlation values.

3. `show_identity_line=True` adds a y=x reference line without changing correlation calculations.

4. `identity_limits="shared_axes"` draws the identity line over the shared visible numeric interval; `"data"` uses the combined finite data range.

5. `xscale`, `yscale`, `xlims`, and `ylims` are applied inside the function, and enabled marginal axes remain synchronized with the main axis.

6. Explicit limits override `xlim_padding_fraction` and `ylim_padding_fraction`; zero-span ranges receive a documented finite fallback padding.

7. Nonpositive rendered values on a log axis raise a clear error.

8. Axis-specific reference lines take precedence over the coarse legacy `axes_lines` behavior when explicitly supplied.

9. The conditional axes return contract described in Section 2.1 remains unchanged.

10. Existing calls retain their current fit, statistics, marginal, scale, and zero-line behavior.

### 4.2 Add new `adtl.ranked_waterfall()`

No existing public plotting function provides stable ranked vertical bars, categorical direction colors, labeled reference lines, and a returned ranked table.

#### Proposed full signature

```python
def ranked_waterfall(
    df: pd.DataFrame,
    *,
    value: str,
    label: str,
    color_by: str | None = None,
    color_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    ascending: bool = True,
    tie_breaker: str | None = None,
    allow_duplicate_labels: bool = False,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    bar_width: float = 0.8,
    bar_alpha: float = 1.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    tick_rotation: float = 90,
    tick_fontsize: float | None = 7,
    legend_title: str | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

The returned table contains the plotted row order, resolved zero-based rank, and resolved bar color.

### 4.3 Add new `adtl.category_composition()`

No existing public helper cleanly covers an ordered stacked composition plot while returning its exact count or normalized table.

#### Proposed full signature

```python
def category_composition(
    df: pd.DataFrame,
    *,
    x: str,
    category: str,
    x_order: Sequence[Any] | None = None,
    category_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    normalize: Literal[False, "fraction", "percent"] = False,
    include_unobserved_x: bool = True,
    include_unobserved_categories: bool = True,
    missing_category: Literal["drop", "error", "label"] = "drop",
    missing_label: str = "Missing",
    annotate: bool = False,
    annotation_format: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend_title: str | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] = (7, 5),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

The returned table contains the exact plotted count, fraction, or percentage values in configured order.

### 4.4 Add new `adtl.residual_diagnostic()`

This remains separate from `corr_dotplot` because a residual diagnostic must not imply that a correlation or regression model was fitted by the plotting function.

#### Proposed full signature

```python
def residual_diagnostic(
    df: pd.DataFrame,
    *,
    x: str,
    residual: str,
    x_transform: Literal["none", "log", "log2", "log10"] = "none",
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    point_color: Any = "#4477AA",
    point_size: float = 48,
    point_alpha: float = 0.8,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 4),
    dropna: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

The function plots supplied residuals only and returns both original and transformed x values.

## 5. Stage 2: update existing `adtl.adata_histograms()`

The existing histogram function already supports AnnData/DataFrame input, grouped overlays, variable grouping, filtering, shared limits, zero/mean lines, and KDE. Extend it instead of adding a parallel histogram API.

### Proposed full updated signature

```python
def adata_histograms(
    adata: anndata.AnnData | None = None,
    *,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate", "all"] = "aggregate",
    collapse_func: Literal[
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "count",
        "select_max_ref_value",
    ] = "mean",
    ref_values_obsm_key: str = "ref_values",
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    subset_min_count: int | None = None,
    subset_small_group_policy: Literal["exclude", "error", "keep"] = "exclude",
    subset_legend_metrics: Sequence[
        Literal["count", "mean", "median"]
    ] | None = None,
    subset_label_format: str | None = None,
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    show_all_obs_hist: bool = True,
    all_obs_color: Any = "0.7",
    all_obs_alpha: float = 0.20,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    xlims: Sequence[float] | None = None,
    add_zero_line: bool = True,
    add_mean_line: bool = True,
    add_mean_to_legend: bool = True,
    highlight_negative_mean_legend: bool = True,
    zero_line_style: Mapping[str, Any] | None = None,
    mean_line_style: Mapping[str, Any] | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    bins: int | str | Sequence[float] = "auto",
    binwidth: float | None = None,
    binrange: tuple[float, float] | None = None,
    stat: Literal[
        "count",
        "frequency",
        "probability",
        "percent",
        "density",
    ] = "density",
    multiple: Literal["layer", "dodge", "stack", "fill"] | None = None,
    element: Literal["bars", "step", "poly"] | None = None,
    fill: bool | None = True,
    kde: bool = True,
    kde_fill: bool = False,
    kde_fill_alpha: float = 0.20,
    kde_bw_method: str | float | None = None,
    kde_grid_points: int | None = None,
    kde_clip: tuple[float, float] | None = None,
    common_bins: bool = True,
    common_norm: bool = False,
    discrete: bool | None = None,
    cumulative: bool = False,
    alpha: float | None = None,
    color: Any | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    subplot_title_var_col: str | None = None,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend_loc: str | int | None = None,
    legend_bbox_to_anchor: tuple[float, ...] | None = None,
    legend: bool = True,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
```

### New argument behavior

1. `subset_min_count` is calculated from finite values after existing filtering and missing/zero handling.

2. Group exclusion affects drawing only and never mutates caller data.

3. Explicit palette mappings keep category colors stable after exclusions.

4. `subset_label_format` may reference only `group`, `count`, `mean`, and `median`.

5. KDE settings are identical across groups; a group with fewer than two distinct values skips only its KDE layer.

6. Existing zero/mean switches remain backward compatible and de-duplicate equivalent configured reference lines.

7. `kde_fill=True` fills only KDE curve artists from zero to the rendered KDE height, using the corresponding KDE line color and `kde_fill_alpha`.

8. KDE underfill uses the same normalization as the rendered KDE. In particular, count-scaled histograms retain count-scaled KDE curves and underfill rather than silently switching to density.

9. KDE underfill is skipped when the corresponding KDE is skipped, never creates a legend entry, and never fills zero, mean, or other reference lines.

10. `kde_fill_alpha` must be finite and within `[0, 1]`. Existing calls remain unfilled because `kde_fill` defaults to `False`.

## 6. Stage 3: update existing `adtl.datapoints()`

The current function already provides DataFrame/AnnData input, observation and variable filtering, categorical x groups, boxplots, violins, deterministic jitter, faceting, and summary legends. Extend it rather than adding a new categorical point function.

### Proposed full updated signature

```python
def datapoints(
    input_data: anndata.AnnData | pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate", "all"] = "aggregate",
    collapse_func: Literal[
        "mean", "median", "sum", "min", "max", "count"
    ] = "mean",
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    summary_filter_obs_by_isin_lists: Mapping[
        str, Sequence[Any]
    ] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    subplot_by_obs_key: str | None = None,
    subplot_by_var_key: str | None = None,
    subplot_by_var_missing_label: str = "Missing",
    subplot_order: Sequence[Any] | None = None,
    x_order: Sequence[Any] | None = None,
    x_order_include_unobserved: bool = False,
    x_by_obs_key: str | None = None,
    x_by_obs_missing_label: str = "Missing",
    x_by_obs_multi_var_mode: Literal[
        "panel_by_variable", "pool_variables"
    ] = "panel_by_variable",
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    color: Any | None = None,
    marker_by_obs_key: str | None = None,
    marker_order: Sequence[Any] | None = None,
    marker_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    jitter_amount: float = 0.2,
    random_seed: int | None = 0,
    point_size: float = 60,
    point_alpha: float = 0.85,
    boxplot: bool = True,
    boxplot_width: float = 0.55,
    boxplot_showfliers: bool = False,
    violinplot: bool = False,
    violin_width: float = 0.8,
    violin_alpha: float = 0.25,
    legend_metrics: Sequence[
        Literal["mean", "median", "count", "std", "sem"]
    ] | None = ("mean",),
    legend_metric_formats: Mapping[
        Literal["mean", "median", "count", "std", "sem"],
        str,
    ] | None = None,
    show_all_data_metrics: bool = True,
    highlight_negative_mean_legend: bool = True,
    group_annotations: Sequence[Mapping[str, Any]] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharey: bool = False,
    yscale: str = "linear",
    ylims: Sequence[float] | None = None,
    add_zero_line: bool = False,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend_loc: str | int | None = None,
    legend_bbox_to_anchor: tuple[float, ...] | None = None,
    legend_scope: Literal["axis", "figure"] = "axis",
    append_marker_handles_to_legend: bool = True,
    append_reference_handles_to_legend: bool = True,
    legend: bool = True,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
    savefig: bool = False,
    file_name: str = "datapoints.png",
    logger: logging.Logger | None = None,
    log_level: int | str | None = None,
    allow_unused_params: bool = False,
    **params: Any,
) -> tuple[plt.Figure, dict[str, plt.Axes], pd.DataFrame]:
```

### New argument behavior

1. `summary_filter_obs_by_isin_lists` affects boxes, violins, metrics, and annotations without hiding point rows.

2. The returned table adds `summary_included` and resolved marker-style fields.

3. `marker_by_obs_key` is independent of the existing subset/color channel.

4. Marker mappings support marker symbol, filled/open mode, label, optional fixed face/edge colors, size, and alpha.

5. `group_annotations` support count, mean, median, standard deviation, and standard error at the metric position or axes top/bottom.

6. Log scale validates every rendered point and summary value.

7. Existing legend entries remain first; marker handles and reference-line handles are appended in deterministic order.

8. `legend_metric_formats` optionally overrides the text for individual metrics while preserving the configured `legend_metrics` and group order.

9. Each format string may reference `{metric}` and `{value}`. Count values are passed as integers; mean, median, standard deviation, and standard error are passed as floats.

10. Metrics without a custom format retain the existing default formatting. Unsupported metric names, non-string formats, or invalid placeholders raise `ValueError` before drawing.

11. Formatting changes legend text only. It does not change summary populations, calculated values, point rows, boxes, violins, annotations, returned plot data, or the existing return tuple.

## 7. Stage 4: add new `adtl.longitudinal_trajectories()`

`adtl.paired_datapoints()` remains the correct API for exactly two paired conditions. Extending it to arbitrary ordered categories, adjacent-gap rules, separate display values, and three independent visual channels would weaken its existing contract. A new focused function is therefore warranted.

### Proposed full signature

```python
def longitudinal_trajectories(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    subject: str,
    x_order: Sequence[Any],
    display_y: str | None = None,
    line_eligible: str | None = None,
    connect: Literal["adjacent", "all", "none"] = "adjacent",
    line_color_by: str | None = None,
    point_color_by: str | None = None,
    color_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    marker_by: str | None = None,
    marker_order: Sequence[Any] | None = None,
    marker_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    line_color: Any = "0.75",
    line_width: float = 0.8,
    line_alpha: float = 1.0,
    point_size: float = 48,
    point_alpha: float = 0.9,
    x_jitter: float = 0.0,
    random_seed: int | None = 0,
    yscale: str = "linear",
    ylims: Sequence[float] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.5, 5.5),
    color_legend_title: str | None = None,
    marker_legend_title: str | None = None,
    color_legend_kwargs: Mapping[str, Any] | None = None,
    marker_legend_kwargs: Mapping[str, Any] | None = None,
    dropna_display: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

### Required behavior

1. `y` is the exact value eligible for a line; `display_y` is the plotted point value.

2. `line_eligible` optionally names an additional boolean eligibility column.

3. Adjacent mode connects only consecutive configured x positions.

4. Duplicate subject/x rows raise a clear error.

5. Line color, point color, and marker style are independent.

6. The returned table includes position, exact value, display value, line eligibility, resolved styles, and deterministic segment identifiers.

## 8. Preserve the promoted marginal contract

Marginal correlation plots are now part of `adtl.corr_dotplot()` and must be implemented, documented, and tested there. Section 4.1 contains the only proposed correlation-scatter signature.

Required compatibility behavior:

1. Enabling only the x marginal returns an axes dictionary with a populated `main` and `x_marginal`, and `y_marginal=None`.

2. Enabling only the y marginal returns an axes dictionary with a populated `main` and `y_marginal`, and `x_marginal=None`.

3. Enabling both marginals returns all three axes.

4. Disabling both marginals retains the legacy single-axes return.

5. Marginal bins, fill, KDE, all-observation overlays, grouping, palette order, filtering, and panel ratios retain their current behavior unless an explicit new keyword changes them.

6. New scales, limits, and padding controls keep the main and enabled marginal axes synchronized.

7. Returned fit, correlation, and p-value remain numerically compatible with the current implementation.

## 8A. Stage 5: remaining public analytical renderers

Stage 5 completes four downstream requirements without introducing model fitting or dataset-specific behavior. The two existing-function additions are specified in Sections 5 and 6. The following two functions are new focused renderers.

### 8A.1 Add new `adtl.kaplan_meier_plot()`

The function accepts caller-precomputed survival steps, confidence limits, censor coordinates, and numbers at risk. It must not calculate a Kaplan-Meier estimator, confidence interval, censor status, risk set, cutoff, group definition, or log-rank statistic.

#### Proposed full signature

```python
def kaplan_meier_plot(
    curve_df: pd.DataFrame,
    risk_table_df: pd.DataFrame,
    *,
    censor_df: pd.DataFrame | None = None,
    time: str = "time",
    survival: str = "survival",
    ci_lower: str = "ci_lower",
    ci_upper: str = "ci_upper",
    group: str = "group",
    risk_time: str = "time",
    risk_count: str = "n_at_risk",
    group_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    ci_alpha: float = 0.20,
    censor_marker: str = "+",
    censor_size: float = 42,
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    title: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[Any, str] | None = None,
    figsize: tuple[float, float] = (8, 6.5),
    show: bool = True,
) -> tuple[
    plt.Figure,
    dict[str, plt.Axes],
    pd.DataFrame,
    pd.DataFrame,
]:
```

#### Required behavior

1. Draw one post-step curve and confidence band per resolved group, plus supplied censor markers at their supplied survival coordinates.

2. Draw a vertically aligned numbers-at-risk panel sharing the time axis. Every displayed risk time must resolve once for every displayed group.

3. Validate finite numeric plotted values, probabilities and confidence bounds within `[0, 1]`, `ci_lower <= survival <= ci_upper`, nonnegative risk counts, matching curve/risk groups, and censor groups contained in the curve groups.

4. Resolve group and palette order deterministically. Explicit mappings retain stable colors when a group is absent or input rows are reordered.

5. Preserve time-zero censor markers and time-zero risk counts exactly as supplied.

6. Return prepared curve and risk tables in rendered order. Do not mutate any caller-owned table.

7. Export the function through both `adtl.kaplan_meier_plot` and `adtl.pl.kaplan_meier_plot`.

### 8A.2 Add new `adtl.continuous_effect_plot()`

The function accepts a caller-precomputed continuous-effect curve, confidence limits, and optional observed-data layer. It must not fit a regression or survival model, calculate a prediction, derive a reference value, calculate confidence limits, jitter observations, or classify observed outcomes.

#### Proposed full signature

```python
def continuous_effect_plot(
    curve_df: pd.DataFrame,
    *,
    x: str,
    estimate: str,
    ci_lower: str,
    ci_upper: str,
    observed_df: pd.DataFrame | None = None,
    observed_x: str | None = None,
    observed_y: str | None = None,
    observed_category: str | None = None,
    observed_order: Sequence[Any] | None = None,
    observed_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    line_color: Any = "#4477AA",
    ci_alpha: float = 0.20,
    xscale: str = "log",
    ylims: Sequence[float] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    annotation: str | None = None,
    annotation_xy: tuple[float, float] = (0.03, 0.97),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6.5, 5),
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes,
    pd.DataFrame,
    pd.DataFrame,
]:
```

#### Required behavior

1. Draw the supplied estimate curve and confidence band in stable x order without recalculating any value.

2. Draw an optional observed layer either as one neutral group or by caller-supplied category order and styles.

3. Support caller-owned reference lines, labels, limits, annotation text, and an existing axis for deterministic multi-panel composition.

4. Validate complete finite curve coordinates, `ci_lower <= estimate <= ci_upper`, positive x values for logarithmic display, complete finite observed coordinates, increasing y limits, and supported observed style keys.

5. Return prepared curve and observed tables in rendered order without mutating caller input.

6. When `ax` is supplied, draw only on that axis and do not show, close, or lay out the owning figure. When `ax` is omitted, create and manage the figure according to `figsize` and `show`.

7. Export the function through both `adtl.continuous_effect_plot` and `adtl.pl.continuous_effect_plot`.

## 9. Shared reference-line contract

Every new `x_reference_lines` or `y_reference_lines` argument accepts an ordered sequence of mappings.

```python
ReferenceLine = Mapping[
    Literal[
        "value",
        "label",
        "color",
        "linestyle",
        "linewidth",
        "alpha",
        "zorder",
    ],
    Any,
]
```

Required behavior:

1. `value` is required and numeric.

2. Unsupported keys raise `ValueError`.

3. Configured order is preserved in legends.

4. Equivalent legacy zero/mean lines and explicit reference lines are not duplicated.

## 10. Documentation requirements

1. Publish every complete signature shown in this plan.

2. Mark each argument as existing or newly added in release notes.

3. Include small synthetic examples without dataset-specific labels or values.

4. Explain display values versus summary values and line-eligible values.

5. Explain category ordering, palette mapping, missing-value handling, scales, limits, references, and return contracts.

6. Document the conditional single-axes versus axes-dictionary return from `corr_dotplot` when marginals are disabled or enabled.

## 11. Testing strategy

1. Verify root and plotting-namespace exports.

2. Lock in legacy behavior before testing new keyword branches.

3. Test returned data, artist positions, line coordinates, colors, marker fills, scales, limits, and legend order.

4. Test deterministic sorting and jitter.

5. Test missing columns, duplicate keys, invalid limits, unsupported styles, and invalid transform domains.

6. Test sparse or degenerate groups only where they are reachable under the public contract.

7. Prefer structural assertions over broad pixel comparisons.

8. Use a small number of tolerance-based image tests for combined marker/color/line encodings.

9. Run focused tests for each function before the full package suite.

10. For KDE underfill, assert collection count, line-matched color, configured alpha, exact curve coordinates, count-versus-density scaling, no legend entry, and no fill for skipped KDEs or reference lines.

11. For legend formatting, assert mixed default/custom formats, deterministic group order, integer count formatting, float metric formatting, invalid placeholders, and unchanged returned data.

12. For Kaplan-Meier rendering, assert step coordinates, confidence polygons, censor coordinates including time zero, group/color order, risk-table values and alignment, and input immutability.

13. For continuous-effect rendering, assert line/band coordinates, observed style/order, reference lines, external-axis behavior, log-domain validation, annotations, returned tables, and input immutability.

## 12. Delivery order

1. Stage 1: extend `corr_dotplot`; add waterfall, composition, and residual functions.

2. Stage 2: extend `adata_histograms`.

3. Stage 3: extend `datapoints`.

4. Stage 4: add `longitudinal_trajectories` while retaining `paired_datapoints` unchanged.

5. Stage 5: add histogram KDE underfill, datapoint legend metric formats, `kaplan_meier_plot`, and `continuous_effect_plot`.

Each stage includes implementation, focused tests, documentation, compatibility review, and a full plotting-suite run before the next stage begins.

## 13. Non-goals

1. No interactive plotting framework.

2. No new theme system.

3. No unrelated legacy plotting refactor.

4. No dataset-specific categories, values, thresholds, or interpretation rules.

5. No model fitting inside the residual or composition functions.

6. No further public API removals as part of the remaining roadmap.

## 14. Definition of done

The roadmap is complete when the updated and new functions, including stage 5, match their documented full signatures, preserve legacy defaults, are deterministic and test-backed, and can produce the described general plot families without handwritten post-processing.

This public version intentionally contains no motivating project details, local paths, report metadata, group labels, data values, counts, or dataset-specific acceptance results.
