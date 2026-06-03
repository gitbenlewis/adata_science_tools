# `_histograms`

Histogram plotting helpers from `_plotting/_histograms.py`.

## Main entry point

1. `adata_histograms`

## `adata_histograms`

`adata_histograms(...)` draws one histogram per selected variable from either an `AnnData` object or a wide `pandas.DataFrame`.
It can also draw one histogram per variable metadata group, such as one panel
per gene when multiple variant columns map to the same gene.
By default, histograms are filled density plots with KDE overlays. Subgroup
histograms use `palette=palettes.tol_colors` unless `subset_palette` is supplied,
and keep the same subgroup-to-color mapping across every panel.
Panels include a red dotted zero reference line by default, plus dashed mean
lines whose values are added to legends by default.

## Full signature

```python
def adata_histograms(
    adata: anndata.AnnData | None = None,
    *,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate"] = "aggregate",
    collapse_func: Literal["mean", "median", "sum", "min", "max", "count"] = "mean",
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Sequence[Any] | str | None = None,
    show_all_obs_hist: bool = False,
    all_obs_color: Any = "0.7",
    all_obs_alpha: float = 0.20,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    xlims: Sequence[float] | None = None,
    add_zero_line: bool = True,
    add_mean_line: bool = True,
    add_mean_to_legend: bool = True,
    bins: int | str | Sequence[float] = "auto",
    binwidth: float | None = None,
    binrange: tuple[float, float] | None = None,
    stat: Literal["count", "frequency", "probability", "percent", "density"] = "density",
    multiple: Literal["layer", "dodge", "stack", "fill"] | None = None,
    element: Literal["bars", "step", "poly"] | None = None,
    fill: bool | None = True,
    kde: bool = True,
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
    legend: bool = True,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
```

## Basic example

```python
import adata_science_tools as adtl

fig, axes = adtl.adata_histograms(
    adata=adata,
    layer="log1p",
    filter_vars_by_isin_lists={"feature_type": ["protein"]},
    filter_obs_by_isin_lists={"Treatment": ["drug"]},
    subset_obs_key="Batch",
    show_all_obs_hist=True,
    sharex=True,
    xlims=[-2, 2],
    bins=30,
    show=False,
)
```

## Supported input modes

1. `adata=...` uses `.X`, `adata.layers[layer]`, or `adata.raw.X` when `use_raw=True`.

2. `df=...` expects rows to be observations and selected feature columns to contain the values to plot. Provide `var_names` or `var_df.index` so metadata columns are not guessed as features.

3. `var_df=...` is feature metadata for DataFrame input. It is used for `filter_vars_by_isin_lists` and optional subplot labels, not as the numeric values being plotted.

## Variable-grouped histograms

1. `var_groupby_key="column"` groups raw variable columns by a variable metadata column after `filter_vars_by_isin_lists` is applied.

2. In grouped mode, `var_names=[...]` selects group names, not raw variable names, and returned axes are keyed by group name strings.

3. `collapse_mode="stack"` pools all non-missing observation-by-variant values in each group into one variant-level distribution. Observations with more measured variants contribute more values.

4. `collapse_mode="aggregate"` applies `collapse_func` across variants within each observation, producing one observation-level value per group before plotting.

5. Grouped mode defaults to `collapse_mode="aggregate"` with `collapse_func="mean"` so each observation contributes at most one value per group.

6. Missing values are handled after stacking or aggregation. `sum` preserves all-missing observations as missing, while `count` returns `0` for observations with no non-missing variants.

7. DataFrame input with `var_groupby_key` requires `var_df` because group metadata must come from variable metadata.

```python
adtl.adata_histograms(
    adata=adata,
    var_groupby_key="Gene",
    var_names=["GENE_A"],
    collapse_mode="stack",
    subset_obs_key="Treatment",
    sharex=True,
    xlims=[-2, 2],
)
```

```python
adtl.adata_histograms(
    adata=adata,
    var_groupby_key="Gene",
    var_names=["GENE_A"],
    collapse_mode="aggregate",
    collapse_func="mean",
    subset_obs_key="Treatment",
    sharex=True,
    xlims=[-2, 2],
)
```

## Filtering and Subsets

1. `filter_obs_by_isin_lists={"column": ["allowed"]}` filters observations with AND semantics.

2. `filter_vars_by_isin_lists={"column": ["allowed"]}` filters variables with AND semantics.

3. `subset_obs_key="column"` draws overlapping histograms by observation metadata group after observation filtering.

4. `show_all_obs_hist=True` adds a neutral non-subsetted all-observation density histogram behind the subset histograms for each variable.

5. `palette` controls subgroup colors by default; `subset_palette` overrides those colors when provided.

6. `add_zero_line=True` draws a red dotted vertical reference line at `x=0`.

7. `add_mean_line=True` draws dashed vertical mean lines for the overall histogram or each subgroup histogram.

8. `add_mean_to_legend=True` adds the mean value to the legend when a mean line is drawn and legends are enabled.

9. Missing `subset_obs_key` values are ignored for grouped histogram layers; variables with no plottable subgroup rows get an annotated empty panel instead of stopping the full figure.

## Important behavior

1. AnnData extraction is column-focused for the selected variables and does not densify the full matrix.

2. The default `stat="density"` and `kde=True` normalize distributions for shape comparison; use `stat="count"` and/or `kde=False` for raw count histograms.

3. KDE is skipped for panels or grouped layers with fewer than two distinct plottable values so sparse subsets still draw histograms.

4. `sharex=True` applies common x-axis limits across selected variable panels; `xlims=[lower, upper]` sets explicit x-axis limits for every panel.

5. `show=False` closes the figure before returning, matching the package's other test-backed plotting APIs.

6. The return value is `(fig, axes)` where `axes` is a dict keyed by selected variable name.
