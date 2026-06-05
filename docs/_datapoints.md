# `_datapoints`

Unpaired variable-level datapoint plotting for `AnnData` objects and wide
`pandas.DataFrame` inputs.

## `datapoints`

`datapoints(...)` draws selected variables or variable groups as categorical
x-axis entries. It is intended for config-driven plotting runs that need the
same obs/var filtering and grouping conventions as `adata_histograms()` without
requiring paired reference/target observations.

```python
fig, axes, plot_df = adtl.datapoints(
    adata=adata,
    var_names=["IL6", "TNF", "CXCL10"],
    subset_obs_key="condition",
    subplot_by_var_key="assay",
    legend_metrics=("mean", "median", "count"),
    violinplot=True,
    show=False,
)
```

```python
fig, axes, plot_df = adtl.datapoints(
    adata=adata,
    var_names=["IL6"],
    x_by_obs_key="condition",
    add_zero_line=True,
    ylims=[-3, 3],
    show=False,
)
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

7. OVERLAYS: Box plots are enabled by default. Violin plots are opt-in with
   `violinplot=True`; when both overlays are enabled, violins draw behind a
   lightweight outline box and the strip points.

8. AXIS REFERENCES: `add_zero_line=True` draws a red dotted horizontal
   reference line at `y=0`. `ylims=[low, high]` applies explicit y-axis limits
   after reference lines are drawn.

9. LEGEND METRICS: `legend_metrics` can include `mean`, `median`, `count`,
   `std`, and `sem`. When `legend=True`, labels include all-data metrics plus
   per-`subset_obs_key` group metrics. Metrics are computed after `nas2zeros`,
   `dropna`, and `dropzeros`.

10. METRIC SCOPE: Legend metrics are panel-level summaries. If one panel contains
   multiple x-axis variables, each metric pools across those x categories.

11. PUBLISHED DOCS: The GitHub Pages docs hub regenerates this page from the
   repository source during its deploy workflow.

### Return value

`datapoints(...)` returns:

```python
fig, axes, plot_df
```

`axes` maps panel names to Matplotlib axes. `plot_df` is the deterministic
long-form plotting table and includes `panel`, `variable`, `source_variable`,
`obs_name`, `x_label`, `x_order`, `value`, and `subset_value`, plus the original
subset or subplot metadata columns when requested.
