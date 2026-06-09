# `_plots`

General plotting utilities from `_plotting/_plots.py`.

This module contains volcano-plot, QQ-plot, and paired time-series datapoint plotting helpers:

- `volcano_plot_generic`
- `qqplot`
- `timeseries_paired_datapoints`

The configurable unpaired and Pre/Post/ref-target datapoint plotting APIs live in
`_plotting/_datapoints.py` and are documented in [`_datapoints.md`](_datapoints.md)
and [`_paired_datapoints.md`](_paired_datapoints.md). Older replacements live in
[`_plots_depreciated.md`](_plots_depreciated.md).

## `volcano_plot_generic`

Use `volcano_plot_generic(...)` to render a volcano plot from a results `DataFrame`.

### Full signature

```python
def volcano_plot_generic(
        _df,
        l2fc_col: str | None = 'log2FoldChange',
        set_xlabel: str | None = 'log2fc model',
        xlimit: str | None = None,
        pvalue_col: str | None = 'pvalue',
        set_ylabel: str | None = '-log10(pvalue)',
        ylimit: str | None = None,
        title_text: str | None = 'volcano_plot',
        comparison_label: str | None = ' Comparison',
        hue_column: str | None = None,
        hue_palette_color_list: list | None = None,
        log2FoldChange_threshold: float | None = .1,
        pvalue_threshold: float | None = None,
        figsize: tuple | None = (15, 10),
        legend_bbox_to_anchor: tuple | None = (1.15, 1),
        title_fontsize: int | None = None,
        axis_label_and_tick_fontsize: int | None = None,
        legend_fontsize: int | None = None,
        label_top_features: bool | None = False,
        only_label_hue_dots: bool | None = True,
        label_top_features_fontsize: int | None = None,
        label_features_char_limit: int | None = 40,
        feature_label_col: str | None = 'gene_names',
        n_top_features: int | None = 50,
        dot_size_shrink_factor: int | None = 300,
        savefig: bool | None = False,
        file_name: str | None = 'volcano_plot.png',
                     ):

```

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

### Full signature

```python
def qqplot(
    data,
    pvalue_column: str | None = None,
    *,
    source: str = "auto",      # "auto" | "var" | "obs" (for AnnData) | "df"
    title: str | None = None,
    pvalue_column_plot_label: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple = (5, 5),
    show: bool = True,
    return_points: bool = False,
    annotate_lambda: bool = True,
    savefig: bool = False,
    filename: str = "qqplot_pvalues.png",
    plotting_position: str = "Blom"  # "Blom" or "Weibull"
):
```

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

### Full signature

```python
def timeseries_paired_datapoints(
    adata,
    feature_name,
    x_col='TimePoint',
    feature_name_label_col=None,
    layer='norm',
    Hue='Treatment_unique',
    subplotby=None,
    analyte_label='analyte_Level',
    savefig=False,
    file_name='test',
    pvalue_label1='paired-ttest',
    pvalue_col_in_var1=None,
    pvalue_label2=None,
    pvalue_col_in_var2=None,
    pvalue_label3=None,
    pvalue_col_in_var3=None,
    pvalue_label4=None,
    pvalue_col_in_var4=None,
    pvalue_label5=None,
    pvalue_col_in_var5=None,
    pvalue_label6=None,
    pvalue_col_in_var6=None,
    pvalue_label7=None,
    pvalue_col_in_var7=None,
    pvalue_label8=None,
    pvalue_col_in_var8=None,
    pvalue_label9=None,
    pvalue_col_in_var9=None,
    pvalue_label10=None,
    pvalue_col_in_var10=None,
    pvalue_label11=None,
    pvalue_col_in_var11=None,
    pvalue_label12=None,
    pvalue_col_in_var12=None,
    pvalue_label13=None,
    pvalue_col_in_var13=None,
    pvalue_label14=None,
    pvalue_col_in_var14=None,
    pvalue_label15=None,
    pvalue_col_in_var15=None,
    pvalue_label16=None,
    pvalue_col_in_var16=None,

    subject_col='Subject_ID',
    connect_lines=True,
    jitter_amount=0.2,
    legend=False,
    figsize=(10, 6),
    color_list=["#88CCEE", "#AA4499", "#117733", "#44AA99", "#332288", "#999933", "#DDCC77", "#661100", "#CC6677", "#882255"],
    jump_n_colors=0,
):
```

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

This page documents current code in `_plotting/_plots.py` and repo example usage such as `example_PMID_33969320/scripts/make_volcano_plots.py`. Dedicated datapoint regression coverage lives in `tests/test_datapoints.py` and `tests/test_paired_datapoints.py`; the functions on this page are based on current code and repo example usage.
