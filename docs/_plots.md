# `_plots`

General plotting utilities from `_plotting/_plots.py`.

This module contains volcano-plot, QQ-plot, and paired datapoint plotting helpers:

- `volcano_plot_generic`
- `qqplot`
- `paired_datapoints`
- `timeseries_paired_datapoints`

The newer configurable Pre/Post or ref/target paired plotting API is documented in
[`_paired_datapoints.md`](_paired_datapoints.md). Older replacements live in
[`_plots_depreciated.md`](_plots_depreciated.md).

## `volcano_plot_generic`

Use `volcano_plot_generic(...)` to render a volcano plot from a results `DataFrame`.

```python
import adata_science_tools as adtl

ax = adtl.volcano_plot_generic(
    adata.var,
    l2fc_col="log2FoldChange",
    pvalue_col="pvalue",
    comparison_label="COVID over NOT",
    label_top_features=True,
    savefig=True,
    file_name="results/volcano.png",
)
```

### Important behavior

- Input is a plain `DataFrame`; the function does not currently accept `AnnData` directly.
- Missing p-values are filled with `1`, and the plot uses a derived `-log10(pvalue)` column.
- If `hue_column` is not provided, the plot colors points by an internal `Significance` category with levels `Not Significant`, `alpha=0.2`, `alpha=0.1`, and `alpha=0.05`.
- Significance thresholds combine `pvalue_col` with `abs(l2fc_col) >= log2FoldChange_threshold`.
- `pvalue_threshold` adds a horizontal reference line using the original p-value scale.
- `label_top_features=True` labels extreme or significant rows using `feature_label_col`, truncated by `label_features_char_limit`.
- `savefig=True` writes the figure with `plt.savefig(...)`.
- The return value is the Matplotlib or Seaborn axes object.

### Notes

- The function prints the copied `DataFrame` shape and the save path when saving.
- Axis limits default to high quantiles of the current data when `xlimit` or `ylimit` are not supplied.
- The implementation was updated to use `pvalue_col`; the older `padj`-based variant is legacy.

## `qqplot`

Use `qqplot(...)` to compare observed versus expected `-log10(p)` values.

```python
out = adtl.qqplot(
    adata,
    pvalue_column="model_FDR",
    source="var",
    title="QQ plot: model_FDR",
    savefig=True,
    filename="results/model_fdr_qqplot.png",
)
```

### Supported input modes

- array-like raw p-values
- `pandas.DataFrame` plus `pvalue_column`
- `AnnData` plus `pvalue_column`, read from `adata.var` or `adata.obs`

### Important behavior

- `source="auto"` on `AnnData` checks `adata.var` first, then `adata.obs`.
- Non-finite values and values outside `[0, 1]` are dropped before plotting.
- P-values are clipped away from zero to avoid `-log10(0)`.
- `plotting_position` supports `"Blom"` and `"Weibull"`.
- `annotate_lambda=True` attempts to compute genomic inflation `lambda_gc`.
- The function can either create its own axes or draw into a supplied `ax`.
- If `return_points=True`, the returned dict also includes `expected` and `observed`.

### Return value

`qqplot(...)` returns a dict with:

- `fig`
- `ax`
- `source`
- `n`
- optional `lambda_gc`
- optional `expected` and `observed`

## `timeseries_paired_datapoints`

Use `timeseries_paired_datapoints(...)` for per-feature paired datapoint plots across ordered time or condition labels from `adata.obs`.

```python
adtl.timeseries_paired_datapoints(
    adata,
    feature_name="IL6",
    x_col="TimePoint",
    Hue="Treatment_unique",
    subplotby="DiseaseGroup",
    layer="norm",
    subject_col="Subject_ID",
    connect_lines=True,
    savefig=True,
    file_name="results/IL6_timeseries.png",
)
```

### Important behavior

- This function is `AnnData`-only.
- `feature_name` must exist in `adata.var_names`.
- `layer` must exist in `adata.layers`.
- `x_col`, `Hue`, and optional `subplotby` are read from `adata.obs`.
- If `feature_name_label_col` exists in `adata.var`, that value is used for the title label.
- Up to 16 p-value columns from `adata.var` can be appended as footer text with paired `pvalue_label*` and `pvalue_col_in_var*` arguments.
- If `connect_lines=True`, repeated observations are connected by `subject_col`.
- The function calls `plt.show()` and then closes the figure with `plt.close(fig)`.

### Return behavior

The current implementation does not return a figure object. Treat it as a show-and-optionally-save helper.

## Coverage note

This page documents current code in `_plotting/_plots.py` and repo example usage such as `example_PMID_33969320/scripts/make_volcano_plots.py`. Dedicated regression coverage exists for `paired_datapoints()` in `tests/test_paired_datapoints.py`; the other functions on this page are based on current code and repo example usage.
