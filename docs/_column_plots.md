# `_column_plots`

Horizontal bar and log2 fold-change dotplot figure builders from `_plotting/_column_plots.py`.

These functions are the package's main helpers for per-feature summary columns and composite bar-plus-dotplot layouts.

## Public entry points

- `barh_column`
- `l2fc_dotplot_single`
- `l2fc_dotplot_column`
- `barh_l2fc_dotplot_column`
- `barh_dotplot_dotplot_column`
- `barh_dotplot_dotplot_dotplot_column`
- `barh_4X_dotplot_column`

## Shared data model

Most functions support one of these input paths:

- `adata`, optionally with `layer`
- explicit `x_df`, `obs_df`, and `var_df`

Shared expectations:

- `feature_list` is required for every feature-oriented plot.
- Bar panels read expression values from `adata.X`, `adata.layers[layer]`, or `x_df`.
- Dotplot panels read per-feature statistics from `adata.var` or `var_df`.
- Grouping for bar plots comes from `comparison_col` in `adata.obs` or `obs_df`.

## `barh_column`

Use `barh_column(...)` for a single column of horizontal grouped bar plots, with optional stripplot overlays.

```python
fig, axes = adtl.barh_column(
    adata=adata,
    layer="pgml",
    feature_list=["IL6", "TNF", "CXCL10"],
    comparison_col="Treatment",
    include_stripplot=True,
    savefig=True,
    file_name="results/barh_column.png",
)
```

Important behavior:

- `feature_list` is required.
- `use_adata_raw=True` switches to `adata.raw.to_adata()`.
- Sparse matrices are densified before building the plotting `DataFrame`.
- `comparison_order` fixes category order when provided.
- `feature_label_vars_col` is used for display labels when available; otherwise the feature index is used.
- The function returns `(fig, axes)`.

## `l2fc_dotplot_single`

Use `l2fc_dotplot_single(...)` for one dotplot axis summarizing log2 fold change and p-value significance for a list of features.

```python
fig, ax = adtl.l2fc_dotplot_single(
    adata=adata,
    feature_list=["IL6", "TNF", "CXCL10"],
    dotplot_pval_vars_col_label="pvalue",
    dotplot_l2fc_vars_col_label="log2FoldChange",
)
```

Important behavior:

- Requires `adata` or `var_df`.
- Requires both the p-value column and the log2 fold-change column in `var_df`.
- Points below the threshold are greyed out.
- A red ring marks the `pvalue_cutoff_ring` threshold in `-log10(p)` space.
- The return value is `(fig, ax)`.

## `l2fc_dotplot_column`

Use `l2fc_dotplot_column(...)` for a vertically stacked column of one-feature-per-row dotplots.

```python
fig, axes = adtl.l2fc_dotplot_column(
    adata=adata,
    feature_list=feature_list,
    dotplot_pval_vars_col_label="paired_pvalue",
    dotplot_l2fc_vars_col_label="log2FoldChange",
    dotplot_sharex=True,
)
```

Important behavior:

- Accepts `adata` or `var_df`.
- Returns `(fig, ax)` for a single feature and `(fig, axes)` for multiple features.
- Optional annotation text uses the current feature's log2 fold change and p-value.

## `barh_l2fc_dotplot_column`

This is the main composite plotting helper used in the example workflow.

```python
fig, subfigs = adtl.barh_l2fc_dotplot_column(
    adata=adata,
    layer="pgml",
    feature_list=feature_list,
    comparison_col="Treatment",
    dotplot_pval_vars_col_label="FDR",
    dotplot_l2fc_vars_col_label="log2FoldChange",
    savefig=True,
    file_name="results/barh_l2fc_dotplot.png",
)
```

### Layout

- left subfigure: grouped horizontal bars with optional stripplot overlay
- right subfigure: one-feature-per-row log2 fold-change dotplots

### Important behavior

- Returns `(fig, subfigs)`.
- Supports either `adata` or explicit `x_df` plus `obs_df` plus `var_df`.
- `hue_palette_color_list` overrides the category colors for the bar panel.
- Dotplots derive marker color and size from `-log10(p)` and draw a red ring at the cutoff.
- Legends for the bar and dot panels are controlled separately.

This is the function called by `example_PMID_33969320/scripts/make_diff_datapoint_plots.py`.

## Advanced composite variants

The remaining functions extend the same pattern by adding more dotplot columns:

- `barh_dotplot_dotplot_column`: bar column plus two dotplot columns
- `barh_dotplot_dotplot_dotplot_column`: bar column plus three dotplot columns
- `barh_4X_dotplot_column`: bar column plus four dotplot columns

Each added dotplot column gets its own parameter family:

- `dotplot2_*`
- `dotplot3_*`
- `dotplot4_*` for `barh_4X_dotplot_column`

Important behavior:

- These functions still return `(fig, subfigs)`.
- Every added dotplot panel requires its own `*_pval_vars_col_label` and `*_l2fc_vars_col_label`.
- `hue_palette_color_list` must provide at least one color per `comparison_col` category when used.
- These layouts are best treated as configuration-heavy report builders rather than small convenience wrappers.

## Common caveats

- Many functions call `plt.show()` internally.
- `savefig=True` writes the figure with `plt.savefig(...)`.
- Missing required features raise early `KeyError` or `ValueError`.
- There do not appear to be dedicated regression tests for this module; this page is based on current code and the example plotting scripts.
