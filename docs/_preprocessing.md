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

For example, `target_min_value=0.5` treats any selected target value below `0.5`
as `0.5`. Bounds are clamping controls, not filters.

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
