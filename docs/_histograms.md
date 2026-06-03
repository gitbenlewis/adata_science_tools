# `_histograms`

Histogram plotting helpers from `_plotting/_histograms.py`.

## Main entry point

1. `adata_histograms`

## `adata_histograms`

`adata_histograms(...)` draws one histogram per selected variable from either an `AnnData` object or a wide `pandas.DataFrame`.

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

4. `show_all_obs_hist=True` adds a neutral all-observation histogram behind subset histograms.

5. `subset_obs_key` requires at least one non-missing value after observation filtering.

## Important behavior

1. AnnData extraction is column-focused for the selected variables and does not densify the full matrix.

2. `show=False` closes the figure before returning, matching the package's other test-backed plotting APIs.

3. The return value is `(fig, axes)` where `axes` is a dict keyed by selected variable name.
