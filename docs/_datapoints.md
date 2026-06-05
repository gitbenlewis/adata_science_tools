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
   source variable names.

5. PANELS: `subplot_by_obs_key` splits observations into panels by obs metadata.
   `subplot_by_var_key` splits selected x categories into panels by var metadata.
   The two subplot modes are mutually exclusive in v1. Missing values in
   `subplot_by_obs_key` raise instead of silently dropping observations.

6. OVERLAYS: Box plots are enabled by default. Violin plots are opt-in with
   `violinplot=True`; when both overlays are enabled, violins draw behind a
   lightweight outline box and the strip points.

7. LEGEND METRICS: `legend_metrics` can include `mean`, `median`, `count`,
   `std`, and `sem`. When `legend=True`, labels include all-data metrics plus
   per-`subset_obs_key` group metrics. Metrics are computed after `nas2zeros`,
   `dropna`, and `dropzeros`.

8. METRIC SCOPE: Legend metrics are panel-level summaries. If one panel contains
   multiple x-axis variables, each metric pools across those x categories.

### Return value

`datapoints(...)` returns:

```python
fig, axes, plot_df
```

`axes` maps panel names to Matplotlib axes. `plot_df` is the deterministic
long-form plotting table and includes `panel`, `variable`, `source_variable`,
`obs_name`, `x_label`, `x_order`, `value`, and `subset_value`, plus the original
subset or subplot metadata columns when requested.
