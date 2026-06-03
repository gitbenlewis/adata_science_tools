# `_histograms`

Histogram plotting helpers from `_plotting/_histograms.py`.

## Main entry point

1. `adata_histograms`

## `adata_histograms`

`adata_histograms(...)` draws one histogram per selected variable from either an `AnnData` object or a wide `pandas.DataFrame`.
By default, histograms are filled density plots with KDE overlays. Subgroup
histograms use `palettes.tol_colors` and keep the same subgroup-to-color mapping
across every panel.

```python
import adata_science_tools as adtl

fig, axes = adtl.adata_histograms(
    adata=adata,
    layer="log1p",
    filter_vars_by_isin_lists={"feature_type": ["protein"]},
    filter_obs_by_isin_lists={"Treatment": ["drug"]},
    subset_obs_key="Batch",
    show_all_obs_hist=True,
    bins=30,
    show=False,
)
```

## Supported input modes

1. `adata=...` uses `.X`, `adata.layers[layer]`, or `adata.raw.X` when `use_raw=True`.

2. `df=...` expects rows to be observations and selected feature columns to contain the values to plot. Provide `var_names` or `var_df.index` so metadata columns are not guessed as features.

3. `var_df=...` is feature metadata for DataFrame input. It is used for `filter_vars_by_isin_lists` and optional subplot labels, not as the numeric values being plotted.

## Filtering and Subsets

1. `filter_obs_by_isin_lists={"column": ["allowed"]}` filters observations with AND semantics.

2. `filter_vars_by_isin_lists={"column": ["allowed"]}` filters variables with AND semantics.

3. `subset_obs_key="column"` draws overlapping histograms by observation metadata group after observation filtering.

4. `show_all_obs_hist=True` adds a neutral non-subsetted all-observation density histogram behind the subset histograms for each variable.

5. Missing `subset_obs_key` values are ignored for grouped histogram layers; variables with no plottable subgroup rows get an annotated empty panel instead of stopping the full figure.

## Important behavior

1. AnnData extraction is column-focused for the selected variables and does not densify the full matrix.

2. The default `stat="density"` and `kde=True` normalize distributions for shape comparison; use `stat="count"` and/or `kde=False` for raw count histograms.

3. `show=False` closes the figure before returning, matching the package's other test-backed plotting APIs.

4. The return value is `(fig, axes)` where `axes` is a dict keyed by selected variable name.
