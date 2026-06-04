# `_preprocessing`

Preprocessing helpers for transforming and filtering `AnnData` objects.

The current row-operation APIs live in
[`_preprocessing/_adata_row_operations.py`](../_preprocessing/_adata_row_operations.py)
and are re-exported from `adata_science_tools._preprocessing`.

## Main row-operation entry points

- `CFG_filter_adata_by_obs`
- `compute_paired_mean_adata`
- `compute_paired_difference_adata`
- `ref_vs_target_adata`

## `ref_vs_target_adata`

`ref_vs_target_adata(...)` builds a new `AnnData` object with one observation per
matched target/reference pair. It is intended for paired Pre/Post-style
transforms such as `Post - Pre`, percent change, fold change, and log2 fold
change.

### Full signature

The runtime Python signature keeps optional configuration in `**params` for
backward-compatible config-driven use:

```python
def ref_vs_target_adata(
    adata: ad.AnnData,
    groupby_key: str = "Pre_or_Post_obs_col",
    groupby_key_target_value: str = "Post",
    groupby_key_ref_value: str = "Pre",
    opperation_flavor: str = "subtraction",
    obs_dfs: str = "merge",
    ref_obs_suffix: str = ".src_pre",
    target_obs_suffix: str = ".src_post",
    keep_var_df: bool = True,
    **params,
) -> ad.AnnData | tuple[ad.AnnData, pd.DataFrame]:
```

The full supported call surface, with `**params` expanded, is:

```python
result = adtl.ref_vs_target_adata(
    adata,
    groupby_key="Pre_or_Post_obs_col",
    groupby_key_target_value="Post",
    groupby_key_ref_value="Pre",
    opperation_flavor="subtraction",
    obs_dfs="merge",
    ref_obs_suffix=".src_pre",
    target_obs_suffix=".src_post",
    keep_var_df=True,
    pair_by_key="SubjectID",  # required
    layer=None,
    layers_to_compute=None,
    base_layer=None,
    epsilon=1e-9,
    target_min_value=None,
    target_max_value=None,
    ref_min_value=None,
    ref_max_value=None,
    bounds_fill_missing=False,
    bounds_fill_missing_paired_only=False,
    merge_shared_obs_cols=False,
    return_df=False,
    allow_unused_params=False,
    logger=None,
    log_level="INFO",
    save_source_values_obsm=False,
    target_values_obsm_key="post_values",
    ref_values_obsm_key="pre_values",
)
```

`operation_flavor` is also accepted as a corrected alias for the typo-compatible
`opperation_flavor`.

```python
import adata_science_tools as adtl

post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    groupby_key="Pre_or_Post_obs_col",
    groupby_key_target_value="Post",
    groupby_key_ref_value="Pre",
    pair_by_key="SubjectID",
)
```

### Pairing rules

- `pair_by_key` is required through `**params`.
- The function selects target rows with
  `adata.obs[groupby_key] == groupby_key_target_value`.
- It selects reference rows with
  `adata.obs[groupby_key] == groupby_key_ref_value`.
- Pair IDs are stringified for matching and become the returned observation
  index.
- Missing pair IDs in either selected group raise `ValueError`.
- Duplicate pair IDs within either selected group raise `ValueError`.
- Target-only and reference-only pair IDs are dropped, logged, and stored in
  `result.uns["ref_vs_target_adata"]`.
- If there are no overlapping pair IDs, the function raises `ValueError`.

### Operations

The default operation is subtraction, computed as target minus reference:

```text
target - reference
```

Supported operation names are:

- `subtraction`
- `relative_change_pct`
- `relative_change_fc`
- `relative_change_l2fc`

The corrected parameter alias `operation_flavor` is accepted through `**params`,
but the public signature keeps the existing typo-compatible
`opperation_flavor`.

Relative operations use `epsilon` from `**params`, defaulting to `1e-9`:

```text
relative_change_pct  = ((target - reference) / (reference + epsilon)) * 100
relative_change_fc   = (target + epsilon) / (reference + epsilon)
relative_change_l2fc = log2((target + epsilon) / (reference + epsilon))
```

### Data sources

By default, the function computes from `adata.X`. To compute a layer, pass
`layer`:

```python
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    layer="RFU",
)
```

To compute more than one source, pass `layers_to_compute`. Use `None` in that
list for `adata.X`. The returned `.X` is selected by `base_layer`, defaulting to
the first requested source.

```python
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    layers_to_compute=[None, "RFU"],
    base_layer="RFU",
)
```

When a layer source is requested, the computed values for that source are also
stored in `result.layers[source]`.

### Bounds and LOD-style clamping

Optional bounds clamp valid paired values before the selected operation is
computed:

- `target_min_value`
- `target_max_value`
- `ref_min_value`
- `ref_max_value`
- `bounds_fill_missing`
- `bounds_fill_missing_paired_only`

For example, `target_min_value=0.5` treats any selected target value below `0.5`
as `0.5`. Bounds are clamping controls, not filters.

By default, bounds do not impute missing values. Set
`bounds_fill_missing=True` to fill every missing value on each bounded side
before clipping and computation. The fill value uses side-specific precedence:
the side's min value when present, otherwise the side's max value.

Set `bounds_fill_missing_paired_only=True` to fill missing values only when the
opposite side of the same pair and variable is present. If both missing-fill
flags are `True`, paired-only fill behavior is used. Numeric clipping of
present values is unchanged.

For one variable with `ref_min_value=2`, `target_min_value=1`, and
`bounds_fill_missing_paired_only=True`:

| Raw reference | Raw target | Bounded reference | Bounded target | Reason |
|---:|---:|---:|---:|---|
| `10` | `NaN` | `10` | `1` | Missing target is filled because reference is present. |
| `NaN` | `NaN` | `NaN` | `NaN` | Both sides are missing, so neither side is filled. |
| `NaN` | `20` | `2` | `20` | Missing reference is filled because target is present. |
| `0.5` | `0.25` | `2` | `1` | Present values are still clipped to side-specific bounds. |

### Returned metadata

The returned object stores:

- one observation per matched pair;
- `ref_obs_name`, `target_obs_name`, `pair_order`, source group labels, and the
  operation name in `.obs`;
- operation and source metadata in `.uns["ref_vs_target_adata"]`;
- dropped unmatched pair IDs in both `.uns["ref_vs_target_adata"]` and flat
  convenience keys;
- copied `adata.var` plus operation metadata when `keep_var_df=True`;
- generated operation-only `.var` metadata when `keep_var_df=False`.

With `obs_dfs="merge"`, both source `.obs` tables are included with suffixes
from `ref_obs_suffix` and `target_obs_suffix`. If
`merge_shared_obs_cols=True`, columns whose retained pair values are identical
in both sources are collapsed to one column. `obs_dfs="keep_ref"` and
`obs_dfs="keep_target"` keep one source table plus provenance columns.

### Optional source-value `obsm`

Set `save_source_values_obsm=True` to store the paired, ordered source values
used for the returned `.X` before the final operation result is returned:

```python
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    save_source_values_obsm=True,
    target_values_obsm_key="post_values",
    ref_values_obsm_key="pre_values",
)
```

The stored `obsm` values are `pandas.DataFrame` objects aligned to returned
observations and variables:

- `result.obsm["post_values"]`
- `result.obsm["pre_values"]`

When bounds are requested, these source-value tables reflect the bounded values
used for computation.

`adtl.save_dataset(result, "path/to/result.h5ad")` exports these tables by
default as `.obsm.<key>.csv` files, for example
`result.obsm.pre_values.csv` and `result.obsm.post_values.csv` when the
default keys are used.

### DataFrame return

Set `return_df=True` to return both the result object and a feature matrix
DataFrame for the base source:

```python
result_adata, result_df = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    return_df=True,
)
```
