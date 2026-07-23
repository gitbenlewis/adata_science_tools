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
    stat: Literal["count", "frequency", "probability", "percent", "density"] = "density",
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

## Existing and new arguments

| Argument | Status | Purpose |
|---|---|---|
| `subset_min_count` | New | Minimum finite plotted values required per subgroup and panel |
| `subset_small_group_policy` | New | Exclude, reject, or retain subgroups below the minimum |
| `subset_legend_metrics` | New | Add subgroup count, mean, and/or median to labels |
| `subset_label_format` | New | Format subgroup labels from the supported metric fields |
| Mapping form of `subset_palette` | New | Keep explicit category colors stable when groups are excluded |
| `zero_line_style` | New | Override the existing zero-line style |
| `mean_line_style` | New | Override the existing mean-line style |
| `x_reference_lines` | New | Draw ordered, optionally labeled vertical references |
| `kde_fill` | New | Fill the area beneath each rendered KDE curve |
| `kde_fill_alpha` | New | Set KDE underfill opacity from 0 through 1 |
| `kde_bw_method` | New | Forward a shared KDE bandwidth method |
| `kde_grid_points` | New | Forward a shared KDE grid size |
| `kde_clip` | New | Forward shared KDE support bounds |
| All remaining arguments in the signature | Existing | Preserve the prior input, filtering, grouping, plotting, legend, and return behavior |

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

## Legend positioning

Use `legend_loc` and `legend_bbox_to_anchor` to move per-panel legends.

```python
adtl.adata_histograms(
    adata=adata,
    var_names=["GENE_A"],
    subset_obs_key="Treatment",
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.02, 1),
    show=False,
)
```

## Supported input modes

1. `adata=...` uses `.X`, `adata.layers[layer]`, or `adata.raw.X` when `use_raw=True`.

2. `df=...` expects rows to be observations and selected feature columns to contain the values to plot. Provide `var_names` or `var_df.index` so metadata columns are not guessed as features.

3. `var_df=...` is feature metadata for DataFrame input. It is used for `filter_vars_by_isin_lists` and optional subplot labels, not as the numeric values being plotted.

## All-variable stacked histogram

1. With `var_groupby_key=None`, `collapse_mode="all"` draws one combined panel keyed as `"all"` by stacking all selected raw variable values.

2. `var_names=[...]` still selects raw variable names in this mode, and `filter_vars_by_isin_lists` is applied before stacking.

3. Observations with more selected measured variables contribute more values to the histogram; `collapse_func` is ignored because no within-observation aggregation is performed.

```python
adtl.adata_histograms(
    adata=adata,
    var_names=["GENE_A_variant_1", "GENE_A_variant_2"],
    collapse_mode="all",
    subset_obs_key="Treatment",
    sharex=True,
    xlims=[-2, 2],
)
```

## Variable-grouped histograms

1. `var_groupby_key="column"` groups raw variable columns by a variable metadata column after `filter_vars_by_isin_lists` is applied.

2. In grouped mode, `var_names=[...]` selects group names, not raw variable names, and returned axes are keyed by group name strings.

3. `collapse_mode="stack"` pools all non-missing observation-by-variant values in each group into one variant-level distribution. Observations with more measured variants contribute more values.

4. `collapse_mode="aggregate"` applies `collapse_func` across variants within each observation, producing one observation-level value per group before plotting.

5. Grouped mode defaults to `collapse_mode="aggregate"` with `collapse_func="mean"` so each observation contributes at most one value per group.

6. Missing values are handled after stacking or aggregation. `sum` preserves all-missing observations as missing, while `count` returns `0` for observations with no non-missing variants.

7. `collapse_func="select_max_ref_value"` selects one variant per observation and group using the largest non-missing reference value from `adata.obsm[ref_values_obsm_key]`, then plots that selected variant's value from `.X`, `layer`, or `.raw.X`.

8. Reference values may be a DataFrame with observation index and variant columns, or an array-like value with rows aligned to `adata.obs_names` and columns aligned to the active plotting variable axis.

9. Tied maximum reference values are resolved by the first variant in filtered variable order and logged as warnings. Observations with all reference values missing produce missing collapsed values before `nas2zeros`, `dropna`, and `dropzeros` are applied.

10. DataFrame input with `var_groupby_key` requires `var_df` because group metadata must come from variable metadata. `select_max_ref_value` is only supported for AnnData input.

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
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    opperation_flavor="relative_change_l2fc",
    save_source_values_obsm=True,
)

adtl.adata_histograms(
    adata=post_minus_pre,
    var_groupby_key="Gene",
    var_names=["GENE_A"],
    collapse_mode="aggregate",
    collapse_func="select_max_ref_value",
    ref_values_obsm_key="pre_values",
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

4. In subset mode, `show_all_obs_hist=True` by default adds a neutral non-subsetted all-observation histogram behind the subset histograms for each variable; set `show_all_obs_hist=False` to opt out.

5. `palette` controls subgroup colors by default; `subset_palette` overrides those colors when provided.

6. `add_zero_line=True` draws a red dotted vertical reference line at `x=0`.

7. `add_mean_line=True` draws dashed vertical mean lines for the overall histogram, the all-data subset overlay, or each subgroup histogram.

8. `add_mean_to_legend=True` adds mean values to the legend when mean lines are drawn and legends are enabled. In subset mode, the legend starts with `All data (mean=...)`, computed from all filtered plotted values in the panel after value filtering and before missing `subset_obs_key` labels are ignored for subgroup layers.

9. `highlight_negative_mean_legend=True` renders calculated negative mean legend entries in red bold text by default. This applies to non-subset labels such as `Mean = -2`, all-data labels such as `All data (mean=-1)`, and subgroup labels such as `case (mean=-3)`. Set `highlight_negative_mean_legend=False` to keep Matplotlib's default legend text styling.

10. Missing `subset_obs_key` values are ignored for grouped histogram layers.
    A panel with no plottable subgroup rows is annotated only when no all-data
    overlay was drawn. If eligibility filtering excludes every subgroup, that
    annotation reads `No eligible <subset_obs_key> groups`; otherwise it reads
    `No non-missing <subset_obs_key> groups`.

## Subgroup eligibility, colors, and labels

1. `subset_min_count` is evaluated separately for each panel. Counts include only finite values after `nas2zeros`, `dropna`, and `dropzeros` have been applied.

2. `subset_small_group_policy="exclude"` removes groups below the minimum from drawing only. `"error"` raises with the panel and finite group counts, while `"keep"` draws the resolved groups regardless of the minimum.

3. `subset_min_count=0` keeps every representable resolved group eligible, including an observed resolved group with no finite values in a particular panel.

4. Eligibility does not mutate the caller input. Group order and the complete color mapping are resolved before per-panel exclusion, so a surviving group does not change color between panels. A mapping supplied through `subset_palette` must cover every resolved subgroup.

5. `subset_legend_metrics` accepts `count`, `mean`, and `median` in the requested order. Metrics are calculated from the exact finite values eligible for that subgroup layer.

6. `subset_label_format` may reference only `{group}`, `{count}`, `{mean}`, and `{median}` and supports normal Python format specifications, for example `"{group}: n={count}, median={median:.2f}"`.

7. The neutral all-observation overlay is not a subgroup and does not use subgroup metrics or `subset_label_format`. Its existing `All data (mean=...)` label remains unchanged.

```python
import pandas as pd
import adata_science_tools as adtl

measurements = pd.DataFrame(
    {
        "response": [0.2, 0.5, 0.7, 1.1, 1.3],
        "cohort": ["A", "A", "B", "C", "C"],
    }
)

fig, axes = adtl.adata_histograms(
    df=measurements,
    var_names=["response"],
    subset_obs_key="cohort",
    subset_order=["A", "B", "C"],
    subset_min_count=2,
    subset_small_group_policy="exclude",
    subset_palette={"A": "#4477AA", "B": "#EE6677", "C": "#228833"},
    subset_label_format="{group}: n={count}, median={median:.2f}",
    bins=5,
    show=False,
)
```

In this example, group `B` is excluded from the histogram because its finite count is one. Groups `A` and `C` retain their explicitly mapped colors.

## KDE controls

1. `kde_bw_method`, `kde_grid_points`, and `kde_clip` are forwarded as the same KDE configuration for all rendered distributions. They correspond to seaborn KDE `bw_method`, `gridsize`, and `clip` settings.

2. A subgroup with fewer than two distinct finite values still draws its histogram layer and skips only its KDE curve. Other eligible groups retain KDE when their values support it.

3. `kde_fill=True` fills from zero to the exact rendered KDE height. Each fill inherits its KDE line color and uses `kde_fill_alpha`, which must be finite and within `[0, 1]`.

4. Underfill retains the rendered KDE normalization. For example, `stat="count"` produces count-scaled KDE curves and fills rather than density-scaled fills.

5. A skipped KDE has no underfill. KDE fills use no legend entry and do not apply to histogram outlines, zero lines, mean lines, or configured reference lines. The default `kde_fill=False` preserves the existing unfilled appearance.

```python
adtl.adata_histograms(
    df=measurements,
    var_names=["response"],
    subset_obs_key="cohort",
    kde=True,
    kde_fill=True,
    kde_fill_alpha=0.25,
    kde_bw_method=0.5,
    kde_grid_points=128,
    kde_clip=(0.0, 2.0),
    show=False,
)
```

## Line styles and references

1. `zero_line_style` and `mean_line_style` accept `color`, `linestyle`, `linewidth`, `alpha`, and `zorder`. Supplied values override the existing zero- and mean-line defaults without changing their switches.

2. `x_reference_lines` is an ordered sequence of mappings. Each mapping requires a finite numeric `value` and may include `label`, `color`, `linestyle`, `linewidth`, `alpha`, and `zorder`. Unsupported keys raise `ValueError`.

3. Labeled references are appended to the legend in configured order. A reference exactly equal after numeric conversion to an enabled zero line, overall mean line, all-observation mean line, subgroup mean line, or earlier reference is not drawn again. Nearby but unequal scientific thresholds remain distinct.

```python
adtl.adata_histograms(
    df=measurements,
    var_names=["response"],
    zero_line_style={"color": "0.4", "linestyle": "--"},
    mean_line_style={"color": "#4477AA", "linewidth": 2},
    x_reference_lines=[
        {"value": 0.75, "label": "Review threshold", "color": "#CC6677"},
        {"value": 1.25, "label": "Upper threshold", "color": "#AA3377"},
    ],
    show=False,
)
```

## Important behavior

1. AnnData extraction is column-focused for the selected variables and does not densify the full matrix.

2. The default `stat="density"` and `kde=True` normalize distributions for shape comparison; use `stat="count"` and/or `kde=False` for raw count histograms.

3. KDE is skipped for panels or grouped layers with fewer than two distinct plottable values so sparse subsets still draw histograms.

4. `sharex=True` applies common x-axis limits across selected variable panels; `xlims=[lower, upper]` sets explicit x-axis limits for every panel.

5. `show=False` closes the figure before returning, matching the package's other test-backed plotting APIs.

6. The unchanged return value is `(fig, axes)` where `axes` is a dict keyed by selected variable name, selected group name, or `"all"` for `collapse_mode="all"`.
