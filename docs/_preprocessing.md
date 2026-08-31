# `_preprocessing`

Preprocessing helpers for filtering observations and deriving new observation
or variable matrices from `AnnData` objects. The implementations live in
[`_adata_row_operations.py`](../_preprocessing/_adata_row_operations.py) and
[`_adata_column_operations.py`](../_preprocessing/_adata_column_operations.py).
They are re-exported from both `adata_science_tools._preprocessing` and the
top-level package.

## Public entry points

Row operations:

- `CFG_filter_adata_by_obs`
- `compute_paired_mean_adata`
- `compute_paired_difference_adata`
- `ref_vs_target_adata`

Column operations:

- `compute_var_ratios_sums_diffs_adata`
- `compute_var_ratios_sums_diffs_adata_multiple_layers`

## `CFG_filter_adata_by_obs`

`CFG_filter_adata_by_obs(...)` filters observations from explicit arguments or
from the equivalent keys in a YAML-derived dataset dictionary.

### Full signature

```python
def CFG_filter_adata_by_obs(
    adata: ad.AnnData,
    dataset_cfg: dict | None = None,
    filter_obs_boolean_column: str | None = None,
    filter_obs_column_key: str | None = None,
    filter_obs_column_values_list: Sequence | None = None,
    copy: bool = True,
    logger: logging.Logger | None = None,
    **kwargs,
) -> ad.AnnData:
```

```python
filtered = adtl.CFG_filter_adata_by_obs(
    adata,
    filter_obs_boolean_column="include_sample",
    filter_obs_column_key="cohort",
    filter_obs_column_values_list=["discovery", "validation"],
)
```

When `dataset_cfg` contains a filter key, its value takes precedence over the
matching explicit argument. Boolean-column and value-list filters are applied
in sequence with AND semantics. Value matching first attempts numeric
comparison and falls back to string comparison. Missing requested columns raise
`KeyError`.

If no complete filter is configured, `copy=True` returns an unchanged copy and
`copy=False` returns the original object. With active filtering,
`copy=False` returns an `AnnData` view rather than forcing a copy.

## `compute_paired_mean_adata` and `compute_paired_difference_adata`

These older helpers sort two selected observation groups by a pairing column,
then return their elementwise mean or signed difference.

### Full signatures

```python
def compute_paired_mean_adata(
    adata,
    layer="RFU",
    pair_by_key="AnimalID_Tattoo",
    groupby_key="Treatment_unique",
    datapoint_1="drug78hr",
    datapoint_2="drug30hr",
    debug_mode=True,
    layers_to_compute=None,
    base_layer=None,
):

def compute_paired_difference_adata(
    adata,
    layer="RFU",
    pair_by_key="AnimalID_Tattoo",
    groupby_key="Treatment_unique",
    datapoint_1="drug78hr",
    datapoint_2="drug30hr",
    debug_mode=True,
    layers_to_compute=None,
    base_layer=None,
):
```

```python
paired_mean, paired_mean_df = adtl.compute_paired_mean_adata(
    adata,
    layer=None,
    pair_by_key="subject_id",
    groupby_key="visit",
    datapoint_1="post",
    datapoint_2="pre",
    debug_mode=False,
)

paired_difference, paired_difference_df = adtl.compute_paired_difference_adata(
    adata,
    layer=None,
    pair_by_key="subject_id",
    groupby_key="visit",
    datapoint_1="post",
    datapoint_2="pre",
    debug_mode=False,
)
```

The mean is `(datapoint_1 + datapoint_2) / 2`; the difference is
`datapoint_1 - datapoint_2`. Each function always returns an `AnnData` and the
base result as a `DataFrame`. The returned variable metadata is copied from the
input.

`layer=None` uses `adata.X`. To compute more than one source, pass a non-empty
`layers_to_compute` list containing layer names or `None` for `adata.X`.
`base_layer` selects which result becomes `.X` and must be present in that list;
named results are also stored in matching output layers.

These helpers align rows only by sorting the selected groups on categorical
codes. They do not validate duplicate, missing, or unmatched pair IDs, so use
them only when each selected group contains exactly one row per pair. Prefer
`ref_vs_target_adata(...)` below when explicit one-to-one validation and
unmatched-pair provenance are required. The runtime default is
`debug_mode=True`; set it to `False` outside an interactive notebook.

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
    opperation_flavor: str | Sequence[str] = "subtraction",
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
    opperation_flavor="subtraction",  # or a list of operation strings
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
    select_max_ref_value_var_groupby_key=None,
    select_max_ref_value_filter_vars_by_isin_lists=None,
    select_max_ref_value_source_obsm_key="selected_source_variable",
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

`opperation_flavor` can also be a non-empty sequence of operation names. In
that mode, each requested operation is computed and stored as a layer in the
returned object. The returned `.X` uses the first requested operation for the
selected base source, so `["subtraction", "relative_change_pct"]` keeps
`subtraction` in `.X`.

```python
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    opperation_flavor=["subtraction", "relative_change_pct", "relative_change_l2fc"],
)
```

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

When multiple operations are requested for one source, operation names become
layer keys:

- `result.layers["subtraction"]`
- `result.layers["relative_change_pct"]`

When multiple operations and multiple sources are requested together, layers are
named as `source__operation`. The `.X` source is labeled `X` in layer keys:

```python
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    opperation_flavor=["subtraction", "relative_change_l2fc"],
    layers_to_compute=[None, "RFU"],
    base_layer="RFU",
)
```

This creates layers such as `X__subtraction`, `X__relative_change_l2fc`,
`RFU__subtraction`, and `RFU__relative_change_l2fc`. The returned `.X` is
`RFU__subtraction` because `base_layer="RFU"` and `subtraction` is the first
requested operation.

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
present values is unchanged. A missing side is filled only when that side has a
min or max bound; if no bound is provided for that side, the missing value stays
missing.

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
- multi-operation metadata such as `operation_flavors`, `operation_layer_keys`,
  `operation_layer_key_by_source_operation`, and `base_operation_layer`;
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
used for the returned `.X` base source before the final operation result is
returned:

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

### Optional max-reference variable selection

Set `select_max_ref_value_var_groupby_key` to collapse variables by a column in
`adata.var` after target/reference rows are paired and bounds are applied, but
before the requested operation is computed. For each matched pair and variable
group, the function selects the first variable with the maximum bounded
reference value and uses that same selected variable for every requested source
and operation.

```python
grouped_post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="SubjectID",
    save_source_values_obsm=True,
    select_max_ref_value_var_groupby_key="feature_group",
    select_max_ref_value_filter_vars_by_isin_lists={"feature_type": ["primary"]},
    select_max_ref_value_source_obsm_key="selected_source_variable",
)
```

The returned variables are the stringified group labels. Saved source-value
`obsm` tables are collapsed to those same labels, and
`result.obsm["selected_source_variable"]` records which original variable was
selected for each returned observation and group. Empty selected-source entries
mean that all reference values were missing for that observation and group; the
collapsed numeric values remain missing.

Selection metadata is stored in `result.uns["ref_vs_target_adata"]`, including
whether selection was enabled, the grouping key, the normalized filter config,
the selected-source `obsm` key, the number of output groups, and the number of
pair/group selections with tied maximum reference values.

### Logging

`logger` and `log_level` control `INFO`-level argument summaries, progress
messages, and the final `.X` source selection. The argument summary reports
`AnnData` shape, matrix type, layer keys, and column names without logging
matrix values or full `obs`/`var` tables. The final `.X` log records the base
source, base operation, operation-layer key when applicable, shape, and dtype.

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

## `compute_var_ratios_sums_diffs_adata`

`compute_var_ratios_sums_diffs_adata(...)` reads a CSV specification and adds
derived variables along the feature axis.

### Full signature

```python
def compute_var_ratios_sums_diffs_adata(
    adata: ad.AnnData,
    derived_variables_csv_file: str = "derived_variables_csv_file.csv",
    numerator_var_names_col: str = "numerator_var_names",
    denominator_var_names_col: str = "denominator_var_names",
    new_var_names_col: str = "new_var_names",
    var_meta_data_cols_list: list[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    transform: str = "linear",
    return_new_adata_only: bool = False,
    logger: logging.Logger | None = None,
    log_level="INFO",
):
```

The CSV requires the numerator-expression and new-name columns. The denominator
column is optional and is treated as empty when absent. Expressions may combine
feature names with `+` and `-` signs:

| `numerator_var_names` | `denominator_var_names` | `new_var_names` | Result |
| --- | --- | --- | --- |
| `feature_a+feature_b` | empty | `feature_sum` | `feature_a + feature_b` |
| `feature_a-feature_b` | empty | `feature_difference` | `feature_a - feature_b` |
| `feature_a` | `feature_b` | `feature_ratio` | `feature_a / feature_b` |

```python
with_derived_variables = adtl.compute_var_ratios_sums_diffs_adata(
    adata,
    derived_variables_csv_file="config/derived_variables.csv",
    layer="RFU",
    var_meta_data_cols_list=["display_name", "feature_class"],
)
```

`transform="linear"` performs the configured arithmetic directly.
`transform="ln"` assumes the selected values are natural-log transformed: it
returns to linear space for sums and differences, then logs the derived result;
ratios are returned as the difference between logged numerator and denominator.
Non-positive derived values become missing when a logarithm is required.

By default, the derived variables are concatenated to the input along the
feature axis. Set `return_new_adata_only=True` to return only the derived
variables. Rows with missing source features, blank or duplicate new names, or
new names that already exist are skipped with a warning and printed status
message. `use_raw=True` selects `adata.raw.X` and overrides `layer`.

## `compute_var_ratios_sums_diffs_adata_multiple_layers`

The multi-layer wrapper applies the same CSV definition to several input
matrices.

### Full signature

```python
def compute_var_ratios_sums_diffs_adata_multiple_layers(
    adata: ad.AnnData,
    layers_to_compute: list[str] | None = None,
    layers_transforms: list[str] | None = None,
    base_layer: str | None = None,
    **kwargs,
) -> ad.AnnData:
```

```python
with_derived_variables = adtl.compute_var_ratios_sums_diffs_adata_multiple_layers(
    adata,
    layers_to_compute=[None, "RFU"],
    layers_transforms=["linear", "linear"],
    base_layer="RFU",
    derived_variables_csv_file="config/derived_variables.csv",
)
```

Use `None` in `layers_to_compute` for `adata.X`. When omitted, the wrapper uses
only `adata.X`; omitted transforms default to `"linear"` for every source.
`layers_transforms` must match the number of selected sources, and `base_layer`
must be one of them. All sources must produce the same derived variable names.
The base source becomes returned `.X`, while named layer results are stored in
the corresponding output layers.
