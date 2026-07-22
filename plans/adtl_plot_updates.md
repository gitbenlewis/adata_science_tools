# Public plotting feature roadmap for `adata_science_tools`

Status: proposed public feature plan

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

## 12. Delivery order

1. Stage 1: extend `corr_dotplot`; add waterfall, composition, and residual functions.

2. Stage 2: extend `adata_histograms`.

3. Stage 3: extend `datapoints`.

4. Stage 4: add `longitudinal_trajectories` while retaining `paired_datapoints` unchanged.

Each stage includes implementation, focused tests, documentation, compatibility review, and a full plotting-suite run before the next stage begins.

## 13. Non-goals

1. No interactive plotting framework.

2. No new theme system.

3. No unrelated legacy plotting refactor.

4. No dataset-specific categories, values, thresholds, or interpretation rules.

5. No model fitting inside the residual or composition functions.

6. No further public API removals as part of the remaining roadmap.

## 14. Definition of done

The roadmap is complete when the updated and new functions match their documented full signatures, preserve legacy defaults, are deterministic and test-backed, and can produce the described general plot families without handwritten post-processing.

This public version intentionally contains no motivating project details, local paths, report metadata, group labels, data values, counts, or dataset-specific acceptance results.

## 15. Implementation plan

### 15.1 Scope and invariants

1. Treat the complete signatures in Sections 4–7 as the target public API, including argument order, keyword-only placement, defaults, type annotations, and tuple shapes.

2. Preserve existing default behavior unless this roadmap explicitly defines a change. In particular, keep the current correlation calculations, filtering order, marginal rendering, legacy zero/mean-line switches, deterministic jitter defaults, and `paired_datapoints()` semantics.

3. Do not change a statistical model, threshold, biological assumption, or interpretation rule. The new functions are descriptive plotting utilities; `residual_diagnostic()` must use caller-supplied residuals and must not fit a model.

4. Add no external dependency. Use the package's existing pandas, NumPy, Matplotlib, seaborn, scipy, statsmodels, and AnnData stack.

5. For AnnData inputs, select required observations and variables before conversion and densify only the selected sparse block needed for the plot.

6. Implement one stage at a time. Each stage includes code, focused tests, exports, documentation, compatibility review, and verification before the next stage starts.

### 15.2 File ownership and proposed structure

| File | Planned responsibility |
|---|---|
| `_plotting/_utils.py` | Hold only behavior-identical helpers reused by at least two public functions, such as reference-line validation/drawing, scale-domain validation, or style normalization. |
| `_plotting/_corr_dotplots.py` | Extend `corr_dotplot()` without changing its calculation path or conditional axes return. Keep `corr_dotplot_dev()` as a deprecated wrapper rather than a second implementation. |
| `_plotting/_tabular_plots.py` | New module for `ranked_waterfall()`, `category_composition()`, and `residual_diagnostic()`. |
| `_plotting/_histograms.py` | Extend `adata_histograms()` with group eligibility, legend metrics, KDE controls, style mappings, and reference lines. |
| `_plotting/_datapoints.py` | Extend `datapoints()` with summary-only filtering, marker encodings, annotations, scale handling, and reference lines. Leave `paired_datapoints()` unchanged. |
| `_plotting/_longitudinal.py` | New module for `longitudinal_trajectories()`. |
| `_plotting/__init__.py` | Import the new modules so functions are exposed through both `adtl` and `adtl.pl`. |
| `tests/test_corr_dotplots.py` | Lock legacy behavior and test the new correlation branches. |
| `tests/test_tabular_plots.py` | New returned-data and artist-structure tests for the Stage 1 tabular functions. |
| `tests/test_histograms.py` | Cover Stage 2 additions while retaining the current histogram regression matrix. |
| `tests/test_datapoints.py` | Cover Stage 3 additions while retaining current input, grouping, faceting, overlay, and jitter tests. |
| `tests/test_longitudinal.py` | New ordering, segment, style, validation, and determinism tests for Stage 4. |
| `docs/_corr_dotplots.md`, `docs/_histograms.md`, `docs/_datapoints.md` | Publish updated signatures, new-argument notes, synthetic examples, and detailed contracts. |
| `docs/_tabular_plots.md`, `docs/_longitudinal.md` | New API pages with complete signatures, synthetic examples, return schemas, and failure behavior. |
| `docs/_plotting_updates.md` | New release-notes page classifying every argument as existing or new. |
| `docs/README.md` | Add the new pages to the plotting documentation index. |

The root `__init__.py` already wildcard-re-exports `_plotting`, so it should not need an edit. Prove root export behavior with tests instead of changing it preemptively.

### 15.3 Stage 0: baseline and shared contracts

1. Before code changes, run `python tests/test_corr_dotplots.py`, `python tests/test_histograms.py`, `python tests/test_datapoints.py`, and `python tests/test_paired_datapoints.py`; record results and warnings in Section 16.

2. Add legacy-contract tests before implementing new branches. Lock current numerical correlation results, default artists, marginal axes return shapes, histogram zero/mean behavior, datapoint return columns, deterministic jitter, and paired behavior.

3. Define one internal reference-line contract. Validate without mutating caller mappings, require a finite numeric `value`, reject unsupported keys, preserve configured order, and return handles in that order for legends.

4. Add scale or style helpers only when a second identical call site exists. Keep function-specific data preparation in the owning module and avoid a new plotting framework.

5. Keep validation deterministic and side-effect free. Do not mutate caller DataFrames, categorical definitions, or supplied palette/style mappings.

### 15.4 Stage 1: correlation and simple tabular plots

#### 15.4.1 Extend `corr_dotplot()`

1. Add the remaining parameters in the exact Section 4.1 order: `show_fit`, identity controls, scales, explicit limits, padding fractions, and axis-specific reference lines.

2. Leave data assembly, filtering, fit calculation, correlation calculation, p-value calculation, and returned numerical values on the current code path. `show_fit=False` suppresses fit artists only.

3. Separate fit computation from rendering narrowly enough that `show_fit`, `show_fit_legend`, `show_stats_text`, and `show_all_obs_fit` remain independent and testable.

4. Validate identity mode, scales, limits, and padding before drawing. Reject reversed or non-finite limits, negative padding, and nonpositive values that would actually be rendered on a log axis.

5. Resolve automatic limits from finite rendered data. Explicit limits override padding. Use deterministic finite padding for zero-span ranges and transformed-space padding for log axes.

6. For `identity_limits=shared_axes`, draw y=x over the intersection of the final visible x/y ranges. For `data`, use the combined finite data range and allow axes clipping.

7. When an axis-specific reference sequence is `None`, retain that axis's legacy `axes_lines` origin behavior. An explicit sequence, including an empty one, replaces legacy behavior for that axis.

8. Synchronize enabled marginal scales and limits with the main axes without changing marginal bins, KDE, grouping, palette order, filtering, or returned observations.

9. Preserve one `Axes` when both marginals are disabled and the three-key axes dictionary when either is enabled. Keep `corr_dotplot_dev()` as a deprecated forwarding wrapper because the roadmap forbids further public removals.

10. Test numerical compatibility separately from artist visibility, then structurally assert identity coordinates, scales, limits, padding, reference order, log failures, and all four marginal layouts.

#### 15.4.2 Add `ranked_waterfall()`

1. Validate required columns, finite numeric values, duplicate-label policy, tie-breaker presence, palette coverage, reference lines, and positive bar dimensions before creating a figure.

2. Sort stably by value, optionally by `tie_breaker`, and finally by original input order; then assign zero-based ranks.

3. Resolve colors from the full configured category order before drawing. Draw bars at returned rank positions and align ticks with returned row order.

4. Return a new ranked DataFrame containing the plotted fields plus stable `rank` and `resolved_color` columns without mutating the input.

5. Test both sort directions, ties, duplicate-label behavior, mapping and sequence palettes, reference order, returned columns, bar positions, legends, and `show=False`.

#### 15.4.3 Add `category_composition()`

1. Validate required columns, explicit orders, normalization, missing policy, annotation format, and palette coverage. Prefer explicit order, then categorical dtype order, then deterministic first-observed order.

2. Apply missing-category handling before aggregation. Reindex counts to resolved x/category orders so unobserved levels appear only when requested.

3. Compute counts first and derive fractions or percentages vectorially. Represent zero-total x groups with zeros rather than infinities or implicit drops.

4. Draw and legend stacks in category order. Generate annotations from the exact returned values.

5. Return the exact plotted count, fraction, or percentage table with named index and columns.

6. Test observed/unobserved levels, explicit orders, missing policies, zero-total groups, all normalization modes, palettes, annotations, legends, returned values, and `show=False`.

#### 15.4.4 Add `residual_diagnostic()`

1. Validate required columns, construct numeric working arrays for only x and residual values, apply `dropna` once, and never mutate the source DataFrame.

2. Compute only the requested x transformation. Reject invalid log domains and do not fit, center, smooth, correlate, or otherwise transform residuals.

3. Draw supplied residuals and ordered y references. Return original x, transformed x, and residual values for every rendered point.

4. Test every transform, invalid domains, missing-data behavior, references, returned values, artist coordinates, `show=False`, and the absence of any inferred fit line.

### 15.5 Stage 2: extend `adata_histograms()`

1. Add Stage 2 parameters in the exact signature order while preserving current defaults. Expand `subset_palette` to mappings without changing existing sequence or named-palette behavior.

2. Preserve current data selection, variable grouping, filtering, aggregation, and sparse slicing. Calculate subgroup finite counts only after existing missing/zero handling.

3. Resolve subgroup order and complete palette mappings before small-group policy is applied so exclusions cannot shift surviving colors.

4. Implement explicit exclude, error, and keep branches for `subset_small_group_policy`; validate `subset_min_count` as a nonnegative integer.

5. Compute requested legend metrics from exact drawn values. Restrict format fields to `group`, `count`, `mean`, and `median`, validating formats before plotting.

6. Apply identical KDE settings across groups. A group with fewer than two distinct finite values loses only its KDE layer, not its histogram.

7. Apply zero/mean styles and ordered references through the shared line contract. De-duplicate only lines with exactly equal numeric values after validation.

8. Test eligibility timing, all policies, mapping stability, labels/metrics, KDE forwarding and degenerate groups, styles, reference de-duplication, and legacy defaults.

9. Update `docs/_histograms.md` with the full signature, existing/new table, synthetic examples, eligibility timing, KDE behavior, references, palettes, and unchanged return contract.

### 15.6 Stage 3: extend `datapoints()`

1. Add Stage 3 parameters in exact order while keeping current input aliases, filtering, grouping, faceting, overlay defaults, saving behavior, and return tuple.

2. Build current point-level data first, then add `summary_included` using `summary_filter_obs_by_isin_lists`. Existing filters continue to control point visibility.

3. Use only `summary_included` rows for boxes, violins, metrics, and annotations. A summary-empty group retains points but receives no data-derived summary artist.

4. Resolve marker categories independently from subset colors. Validate marker-style fields and return stable columns for category, symbol, fill, colors, size, and alpha.

5. Preserve deterministic jitter and returned row order; marker changes must not alter jitter or subset colors.

6. Calculate annotations from the same summary rows used by overlays and metrics. Support count, mean, median, standard deviation, and standard error at metric or axes-relative positions.

7. Validate scale and limits after filtering but before drawing. Reject every nonpositive point, overlay input, metric, annotation, limit, or reference that would render on a log axis.

8. Assemble legends in deterministic blocks: existing point/subset entries, marker handles, then reference handles. Preserve axis/figure scope and de-duplicate labels.

9. Test point-versus-summary inclusion, returned fields, marker mapping/fill, jitter, annotations, log validation, limits, references, legend order, panels, and unchanged defaults.

10. Update `docs/_datapoints.md` with the signature, existing/new table, point-versus-summary examples, marker/annotation schemas, scales, references, legends, and return additions.

### 15.7 Stage 4: add `longitudinal_trajectories()`

1. Add `_plotting/_longitudinal.py` rather than extending `paired_datapoints()`. Validate required columns, unique nonempty `x_order`, duplicate subject/x pairs, connect mode, scale/limits, styles, and references.

2. Build a prepared table with resolved x position, exact y, display y, eligibility, jittered x, colors, marker fields, and line color without mutating input.

3. Use `y` for line eligibility and `display_y` for points; when absent, `display_y` defaults to `y`. An optional eligibility column is an additional boolean condition.

4. In adjacent mode connect only consecutive configured positions. In all mode connect consecutive available eligible points across gaps. In none mode create no segments.

5. Assign stable segment identifiers before drawing and return each row's deterministic segment membership so every line is auditable.

6. Resolve line color, point color, and marker style independently. Generate jitter after stable ordering and reuse the same endpoints for points and incident lines.

7. Apply `dropna_display` only to point eligibility. Missing exact y independently prevents line eligibility. Validate all rendered values, endpoints, limits, and references on log scale.

8. Draw segments, then points, then references. Build separate ordered color and marker legends.

9. Test duplicates, ordering, exact/display values, eligibility, all connect modes, gaps, independent styles, jitter/segment determinism, missing display, log behavior, schemas, legends, and `show=False`.

10. Add `docs/_longitudinal.md` with the signature, synthetic example, value semantics, connection rules, styles, missing handling, return schema, and separation from paired plots.

### 15.8 Documentation, exports, and release notes

1. Import both new modules from `_plotting/__init__.py` and test every new function at `adtl.<name>` and `adtl.pl.<name>`.

2. Publish every full signature in its API page. Use `docs/_plotting_updates.md` to classify arguments as existing or new and link to detailed behavior.

3. Use small generic synthetic examples. For new tabular functions, show returned plot data alongside the call.

4. Document category order, palette resolution, missing handling, scales, limits, references, and return shapes. State the conditional correlation axes contract prominently.

5. Update `docs/README.md`. Change the root README only if these public pages would otherwise be undiscoverable.

### 15.9 Verification gates

1. Record focused baseline and post-change commands, results, elapsed times, warnings, and deviations in Section 16.

2. Minimum focused gates are the correlation, tabular, histogram, datapoint, paired-datapoint, and longitudinal test modules as they become applicable.

3. After every stage run `python -m unittest discover -s tests`; report pre-existing warnings separately and do not claim broad compatibility if this command is skipped or fails.

4. Compile changed Python modules/tests, inspect public signatures with `inspect.signature`, verify both namespaces, assert returned-table column order, and run `git diff --check`.

5. Prefer structural assertions for patches, scatter offsets, line coordinates, scales, limits, colors, markers, annotations, axes dictionaries, and legend order. Use only a few tolerance-based image tests where structure is insufficient.

6. Before closing a stage, audit for input mutation, accidental API/default changes, full-matrix densification, nondeterminism, unrelated formatting, new dependencies, and statistical changes.

### 15.10 Stage-level diff boundaries

1. Stage 1: correlation module/tests/docs; new tabular module/tests/docs; immediately reused utilities; plotting exports; release-notes page; documentation index.

2. Stage 2: histogram module/tests/docs; already-established shared helpers if required; histogram release-note entries.

3. Stage 3: datapoint module/tests/docs; already-established shared helpers if required; datapoint release-note entries.

4. Stage 4: new longitudinal module/tests/docs; plotting export; documentation index and release-note entries.

5. Do not mix cleanup of column plots, general plots, deprecated helpers, examples, or unrelated documentation into these stages.

### 15.11 Completion criteria

1. Updated and new functions match this roadmap; legacy calls that omit new keywords retain numerical results, default artists, and tuple shapes.

2. Every new function is exposed from both namespaces and returns documented prepared data without mutating caller input.

3. Sorting, palettes, jitter, markers, segments, annotations, and legends are deterministic and test-backed.

4. References, scales, limits, missing values, invalid domains, duplicate keys, sparse selections, degenerate groups, and unobserved categories are documented and tested.

5. Focused and repository suites, compilation, signature/export checks, and diff checks pass; skipped commands and remaining risks are recorded.

6. Documentation contains full signatures, existing/new argument classification, generic examples, returns, and distinctions among visible, summary, exact, display, and line-eligible values.

## 16. Implementation scratchpad

This is a live engineering record. Append implementation evidence and move resolved decisions into the locked section rather than silently rewriting history.

### 16.1 Baseline observations recorded during planning

1. `corr_dotplot()` already has promoted x/y marginals and the conditional axes return. Remaining work is fit visibility, identity, scales, limits, padding, and references.

2. `corr_dotplot_dev()` remains a deprecated wrapper. `corr_dotplot()` is canonical, while the no-removals non-goal requires retaining the wrapper.

3. `adata_histograms()` already supports advertised inputs, variable grouping/collapse, observation groups, filtering, palette sequences, zero/mean lines, seaborn controls, and its two-item return.

4. `datapoints()` already supports input dispatch, variable grouping, faceting, deterministic jitter, box/violin overlays, metrics, limits, legend scopes, and returned data.

5. `paired_datapoints()` has dedicated regression coverage and is not the implementation base for multi-timepoint trajectories.

6. `_plotting/_utils.py` is small while larger modules contain local logic. Introduce helpers only for identical new contracts, not as a broad refactor.

7. Wildcard exports should expose imported new modules through both public namespaces, but export tests must prove this.

8. Tests use `unittest`, a noninteractive Matplotlib backend, and structural assertions, matching this plan.

9. No current release-notes file covers this family, so the plan adds `docs/_plotting_updates.md`.

### 16.2 Locked decisions

1. Retain `corr_dotplot_dev()` only as a deprecated forwarding wrapper.

2. Put the Stage 1 DataFrame-only functions in `_plotting/_tabular_plots.py` and trajectories in `_plotting/_longitudinal.py`.

3. Use original input order as the final stable row tie breaker. For categories prefer explicit order, then categorical dtype order, then first-observed order.

4. Resolve palettes from complete configured orders before drawing-only exclusions.

5. Treat reference values as equivalent only when exactly equal after numeric validation and float conversion, avoiding silent merging of nearby scientific thresholds.

6. Apply padding in transformed space for log axes and data space for linear axes, with deterministic finite handling for zero spans.

7. Keep returned data in plot order with explicit resolved-style fields and no opaque artist objects.

8. Prefer structural tests; reserve image tests for combined encodings that artist/data inspection cannot verify.

9. Missing waterfall values or labels raise because no drop policy authorizes silent removal.

10. A composition `missing_label` that collides with a real category raises rather than merging meanings.

11. An explicit empty reference sequence draws no configured lines; only `None` invokes an applicable legacy fallback.

12. `subset_min_count=0` makes every representable resolved subgroup eligible.

13. The all-observation histogram overlay is excluded from subgroup labels and metrics, preserving its legacy label.

14. Marker styles accept `marker`, `filled`, `label`, `facecolor`, `edgecolor`, `size`, and `alpha`. Defaults cycle deterministic symbols, use filled markers, inherit point color, size, and alpha, and render open markers without a face.

15. Unobserved reserved x categories retain their position and omit data-derived annotations.

16. Longitudinal segment endpoints with conflicting configured line colors raise.

17. Longitudinal `segment_ids` are deterministic tuples, with an empty tuple for points in no segment.

18. Longitudinal input x values absent from `x_order` raise rather than being silently excluded.

### 16.3 Decisions resolved during stage review

All ten pre-implementation questions were resolved as approved and are recorded in Section 16.2, items 9 through 18.

### 16.4 Risk register

1. Large signatures can introduce accidental parameter/default changes. Mitigation: explicit signature assertions.

2. Correlation rendering changes can alter numerical inputs. Mitigation: retain the calculation path and compare returned baseline values.

3. Shared preparation can densify full AnnData matrices. Mitigation: selected-block extraction and sparse regression tests.

4. Matplotlib/seaborn can reorder combined legends. Mitigation: explicitly assemble and test ordered handle blocks.

5. Log validation can reject too much or too little. Mitigation: validate prepared values that will actually render.

6. Missing/unobserved categories and exclusions can shift colors/order. Mitigation: resolve orders and mappings before drawing and return exact plot data.

7. Annotations can diverge from summary artists. Mitigation: use one summary inclusion mask and metric path.

8. Trajectory jitter, gaps, and styles can disconnect lines from points. Mitigation: resolve auditable segments and endpoint coordinates before drawing.

9. Legacy modules invite unrelated cleanup. Mitigation: enforce Section 15.10 diff boundaries.

### 16.5 Implementation log template

For each stage append: date/status; approved scope and resolved decisions; baseline commands/results/warnings; files and behavior changed; focused tests; broad tests and static checks; deviations; remaining risks and next gate.

### 16.6 Current status

1. Planning and all four implementation stages completed on 2026-07-22.

2. The roadmap is implemented additively; existing defaults, scientific calculations, and public return-tuple shapes remain unchanged unless a new keyword is used.

3. The broad compatibility gate passes all 293 discovered tests.

4. The implementation is ready for final diff review and a user-directed commit.

### 16.7 Implementation log

1. Scope and decisions: implementation was authorized with `approve_2_through_9`; the ten stage decisions were resolved as recorded in Section 16.2.

2. Baselines: `python tests/test_corr_dotplots.py` passed 25 tests, `python tests/test_histograms.py` passed 43, `python tests/test_datapoints.py` passed 33, and `python tests/test_paired_datapoints.py` passed 28. Per-module elapsed times were not captured.

3. Stage 1: added shared reference normalization/drawing helpers; extended correlation plots with fit visibility, identity, scale, limit, padding, and reference controls; added ranked waterfall, category composition, and residual diagnostic APIs; used selected-column AnnData extraction to avoid whole-matrix densification.

4. Stage 2: extended histograms with finite subgroup eligibility, small-group policies, subgroup metrics, mapping palettes, KDE controls, styles, and ordered references while retaining the all-observation overlay contract.

5. Stage 3: extended datapoints with summary-only inclusion, independent marker styles, annotations, scale/limit validation, ordered references, deterministic legend blocks, and auditable returned fields; `paired_datapoints()` was left unchanged.

6. Stage 4: added longitudinal trajectories with exact/display values, adjacent/all/none connection modes, deterministic segment tuples, independent color/marker channels, gap rules, log validation, and auditable prepared output.

7. Focused post-change results: new correlation 16/16, histogram 12/12, datapoints 15/15, tabular 8/8, longitudinal 14/14, and cross-roadmap edge coverage 7/7. Legacy correlation, histogram, datapoints, and paired-datapoints modules also remained green.

8. Broad gate: `/usr/bin/time -p python -m unittest discover -s tests` passed 293 tests in 18.843 test seconds and 20.65 wall seconds.

9. Static and security checks: changed Python modules and tests compile; exact public signatures and exports pass focused assertions; `git diff --check` passes; intended APIs resolve identically from both public namespaces; helper imports remain private; changed and new files contain no detected credentials, email addresses, private paths, internal hosts, or sensitive organizational information.

10. Warnings and deviations: only pre-existing seaborn long-palette warnings and small-sample SciPy/statsmodels warnings were observed. No statistical models, thresholds, biological assumptions, external dependencies, or paired-plot semantics were changed. No roadmap deviations remain.

11. Post-review hardening added finite-only histogram rendering, collision-safe mixed-type subgroup legends, strict mean-legend switches, numeric subgroup formatting, generic correlation matrix shape validation, early scale checks, collision-safe datapoint scratch fields, pre-draw result-field collision errors, annotation error normalization, and private helper imports.

12. Correlation scales now include base-2 `log2` and `log1p`/`expm1`. Nonlinear fit and identity artists densely sample their raw-coordinate relations; drawable subgroup fits validate before figure creation; automatic limits include active fit, reference, and applicable origin coordinates; marginals stay synchronized; and returned statistics remain on the untransformed values.
