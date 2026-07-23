# `_datapoints`

Unpaired variable-level datapoint plotting for `AnnData` objects and wide
`pandas.DataFrame` inputs.

## `datapoints`

`datapoints(...)` draws selected variables or variable groups as categorical
x-axis entries. It is intended for config-driven plotting runs that need the
same obs/var filtering and grouping conventions as `adata_histograms()` without
requiring paired reference/target observations.

### Full signature

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
    collapse_func: Literal["mean", "median", "sum", "min", "max", "count"] = "mean",
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
        "panel_by_variable",
        "pool_variables",
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
    legend_metrics: Sequence[Literal["mean", "median", "count", "std", "sem"]] | None = ("mean",),
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

### Existing and new arguments

| Status | Arguments |
|---|---|
| Existing | input selection, variable grouping/collapse, observation and variable filters, subset colors, panels, x categories, deterministic jitter, box/violin overlays, legend metrics, figure sizing, saving, and missing/zero handling |
| New | `summary_filter_obs_by_isin_lists`, mapping-form `subset_palette`, `marker_by_obs_key`, `marker_order`, `marker_styles`, `legend_metric_formats`, `group_annotations`, `yscale`, `y_reference_lines`, `append_marker_handles_to_legend`, and `append_reference_handles_to_legend` |

### Synthetic example

```python
plot_input = pd.DataFrame(
    {
        "feature_a": [1.0, 12.0, 2.0, 14.0],
        "group": ["first", "first", "second", "second"],
        "point_class": ["filled", "open", "filled", "open"],
        "summary_set": ["include", "exclude", "include", "exclude"],
    },
    index=["row_1", "row_2", "row_3", "row_4"],
)

fig, axes, plot_df = adtl.datapoints(
    df=plot_input,
    var_names=["feature_a"],
    x_by_obs_key="group",
    summary_filter_obs_by_isin_lists={"summary_set": ["include"]},
    marker_by_obs_key="point_class",
    marker_order=["filled", "open"],
    marker_styles={
        "filled": {"marker": "o", "label": "Filled"},
        "open": {
            "marker": "s",
            "filled": False,
            "label": "Open",
            "edgecolor": "black",
        },
    },
    group_annotations=[
        {
            "metric": "mean",
            "position": "metric",
            "format": "{value:.2f}",
        }
    ],
    legend_metrics=("count", "mean"),
    legend_metric_formats={
        "count": "n={value:d}",
        "mean": "average={value:.2f}",
    },
    y_reference_lines=[
        {"value": 5.0, "label": "Guide", "linestyle": "--"}
    ],
    show=False,
)

plot_df[[
    "obs_name",
    "value",
    "summary_included",
    "marker_category",
    "resolved_marker",
    "resolved_marker_filled",
]]
```

### Important behavior

1. INPUTS: Provide exactly one of `input_data`, `adata`, or `df`. For wide
   `DataFrame` input, provide `var_names` or `var_df` so metadata columns are
   not inferred as feature columns. Config-driven callers can pass
   `**{"input": data}` as an alias for `input_data`.

2. FILTERS: `filter_obs_by_isin_lists` and `filter_vars_by_isin_lists` use
   AND semantics with the `{"column": ["allowed", ...]}` shape used by nearby
   plotting helpers.

3. MATRIX SOURCE: `AnnData` input supports `.X`, `layers[layer]`, and `raw.X`.
   The function selects requested variable columns before converting sparse
   slices to dense arrays.

4. X-AXIS: By default, one axis named `"all"` contains the selected variable
   names as x-axis categories. With `var_groupby_key`, `collapse_mode="aggregate"`
   uses variable-group names as x categories, while `collapse_mode="stack"` uses
   source variable names. Set `x_by_obs_key="column"` to use observation
   metadata groups as x-axis categories instead. Missing `x_by_obs_key` values
   are routed to `x_by_obs_missing_label`, which defaults to `"Missing"`.
   `x_order` orders the displayed x-axis labels; for config-driven calls, raw
   typed values such as `[2, 1]` and string labels such as `["2", "1"]` both
   match displayed labels. Set `x_order_include_unobserved=True` to keep every
   requested `x_order` label as a tick in each panel even when no post-filter
   datapoints exist for that label; this reserves axis positions only and does
   not add placeholder rows to the returned table. When `subset_obs_key` and
   `x_by_obs_key` are the same obs column, legend and color order follow
   `x_order` unless `subset_order` is supplied; the same rule applies inside
   `subplot_by_obs_key` panels.

5. OBS-GROUP X-AXIS: With `x_by_obs_key` and multiple selected variables or
   groups, `x_by_obs_multi_var_mode="panel_by_variable"` is the default and
   creates one panel per selected variable/group. Use
   `x_by_obs_multi_var_mode="pool_variables"` to pool all selected variables or
   groups within each obs-group x category.

6. PANELS: `subplot_by_obs_key` splits observations into panels by obs metadata.
   `subplot_by_var_key` splits selected x categories into panels by var metadata.
   The two subplot modes are mutually exclusive in v1. Missing values in
   `subplot_by_obs_key` raise instead of silently dropping observations. Missing
   values in `subplot_by_var_key` are routed to `subplot_by_var_missing_label`,
   which defaults to `"Missing"`.

7. POINTS VERSUS SUMMARIES: Existing observation filters determine which point
   rows are visible. `summary_filter_obs_by_isin_lists` is applied afterward and
   sets the returned `summary_included` boolean without removing point rows.
   Boxes, violins, legend metrics, and annotations use only included rows. A
   summary-empty group keeps its points but receives no box, violin, or
   data-derived annotation.

8. OVERLAYS: Box plots are enabled by default. Violin plots are opt-in with
   `violinplot=True`; when both overlays are enabled, violins draw behind a
   lightweight outline box and the strip points.

9. MARKERS: `marker_by_obs_key` is independent of `subset_obs_key`. Categories
   follow `marker_order`, then deterministic observed order. Each
   `marker_styles` entry may use `marker` (the Matplotlib symbol), `filled`,
   `label`, `facecolor`, `edgecolor`, `size`, and `alpha`; unsupported keys
   raise. Defaults cycle through deterministic symbols and otherwise inherit
   the point color, `point_size`, and `point_alpha`. Open markers render with no
   face while retaining their resolved edge color.

10. ANNOTATIONS: Each `group_annotations` mapping requires `metric` (`count`,
    `mean`, `median`, `std`, or `sem`). `position` is `metric`, `axes_top`, or
    `axes_bottom`; unobserved and summary-empty x categories are skipped. Optional
    `label`, `format`, and `text_kwargs` fields control text. `format` may use
    `metric`, `label`, `value`, `count`, and `x_label`. Unsupported named or
    positional fields raise `ValueError` before drawing.

11. AXIS SCALE AND LIMITS: `yscale` is validated against Matplotlib scales.
    The callable-only `function` and `functionlog` scales are rejected because
    this API does not accept their required transform functions.
    `ylims=[low, high]` must be finite and increasing and is applied after
    drawing. For `yscale="log"`, every visible point and rendered summary
    metric, limit, and reference must be positive; `add_zero_line=True` raises.

12. REFERENCES: `add_zero_line=True` retains the legacy red dotted y=0 line.
    `y_reference_lines` is an ordered sequence of mappings with required finite
    numeric `value` and optional `label`, `color`, `linestyle`, `linewidth`,
    `alpha`, and `zorder`. Unsupported keys raise. An explicit reference at
    exactly zero is not duplicated when the legacy zero line is enabled.

13. LEGEND METRICS: `legend_metrics` can include `mean`, `median`, `count`,
    `std`, and `sem`. When `legend=True`, labels include all-data metrics plus
    per-`subset_obs_key` group metrics. Metrics are computed after value
    filtering and from `summary_included` rows only. Panel-level metrics pool
    x categories when a panel contains more than one. `legend_metric_formats`
    optionally replaces the text for individual selected metrics while
    retaining `legend_metrics` order and group order. Each mapping value may use
    only the exact `{metric}` and `{value}` fields. Count values are supplied as
    integers; mean, median, standard deviation, and standard error values are
    supplied as floats. Metrics without an override retain the existing
    `count=N` or `<metric>=<three-significant-digit value>` text. Unsupported
    metric keys, non-string formats, invalid fields, and incompatible format
    specifications raise `ValueError` before drawing. Formatting changes legend
    text only; it does not change summary calculations, artists, annotations,
    returned rows, or the return tuple.

14. LEGEND ORDER: Existing subset and metric entries remain first. Marker
    handles and then labeled reference handles are appended in configured
    order, with labels de-duplicated. The same block order is used for
    `legend_scope="axis"` and `legend_scope="figure"`. The two `append_*`
    switches suppress only their corresponding new legend blocks. When mixed
    category types have the same default string label, only those collisions
    use `repr` labels (for example, `1` and `'1'`). Explicit marker labels are
    unchanged.

15. PALETTES: `subset_palette` accepts a complete mapping as well as existing
    sequence and named-palette forms. Mapping colors remain attached to their
    category regardless of panel composition; missing mapping keys raise.

16. DETERMINISM: Summary masks and marker styling do not alter returned row
    order or seeded jitter. No caller DataFrame or style mapping is mutated.

17. COLUMN COLLISIONS: Caller metadata named `_point_color` or `_jittered_x`
    is preserved because rendering uses collision-proof scratch columns. A
    selected role column that conflicts with a documented returned field raises
    `ValueError` before drawing. Exact matches remain valid only for
    `subset_obs_key="subset_value"`, string-valued
    `subplot_by_obs_key="panel"`, `subplot_by_var_key="panel"`,
    `x_by_obs_key="x_label"`, and `marker_by_obs_key="marker_category"`. These
    preserve the role's value and documented meaning; non-string observation
    panel values conflict with the normalized returned panel labels and raise.

18. PUBLISHED DOCS: The GitHub Pages docs hub regenerates this page from the
    repository source during its deploy workflow.

### Return value

`datapoints(...)` returns:

```python
fig, axes, plot_df
```

`axes` maps panel names to Matplotlib axes. `plot_df` is the deterministic
long-form plotting table and retains the existing `panel`, `variable`,
`source_variable`, `obs_name`, `x_label`, `x_order`, `value`, and
`subset_value` fields, plus requested subset or subplot metadata.

Stage 3 adds `summary_included`, `marker_category`, `resolved_marker`,
`resolved_marker_filled`, `resolved_marker_label`,
`resolved_marker_facecolor`, `rendered_marker_facecolor`,
`resolved_marker_edgecolor`, `resolved_marker_size`, and
`resolved_marker_alpha`. `resolved_marker_facecolor` records an optional fixed
style color; `rendered_marker_facecolor` records the actual inherited or open
face used for the point. These are plain auditable values, not artist objects.
