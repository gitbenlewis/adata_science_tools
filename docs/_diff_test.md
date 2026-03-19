# `_diff_test`

Differential testing utilities for `AnnData` objects and CSV-backed expression tables.

This module exposes one public API:

- `diff_test`

`diff_test(...)` computes feature-wise summary statistics plus one or more hypothesis tests across:

- independent groups,
- paired groups, or
- nested paired comparisons such as `(target - target_control)` vs `(ref - ref_control)`.

It returns a feature-indexed `pandas.DataFrame` and can optionally save results to CSV, write a log file, and store a copy in `adata.uns`.

## Supported test families

The `tests` argument accepts these values:

- `ttest_ind`
- `mannwhitneyu`
- `ttest_rel`
- `WilcoxonSigned`
- `ttest_rel_nested`
- `WilcoxonSigned_nested`

In practice:

- `ttest_ind` and `mannwhitneyu` are for independent groups.
- `ttest_rel` and `WilcoxonSigned` are for paired groups.
- `ttest_rel_nested` and `WilcoxonSigned_nested` are for paired difference-of-differences workflows.

## Input modes

`diff_test(...)` supports two mutually exclusive input modes.

### `AnnData` mode

Use `adata` plus either:

- `adata.X`,
- `adata.layers[layer]`, or
- `adata.raw.X` / `adata.raw.layers[layer]` when `use_raw=True`.

Independent example:

```python
import adata_science_tools as adtl

results = adtl.diff_test(
    adata,
    groupby_key="Treatment",
    groupby_key_target_values=["drug"],
    groupby_key_ref_values=["vehicle"],
    tests=["ttest_ind", "mannwhitneyu"],
    add_values2results=True,
    sortby="Age",
    save_log=False,
)
```

### `x_df` / `obs_df` / `var_df` mode

You can also provide a data matrix plus matching observation and variable tables:

```python
results = adtl.diff_test(
    None,
    x_df="x.csv",
    obs_df="obs.csv",
    var_df="var.csv",
    groupby_key="Treatment",
    groupby_key_target_values=["drug"],
    groupby_key_ref_values=["vehicle"],
    tests=["ttest_ind"],
    add_adata_var_column_key_list=["annotation"],
    save_table=True,
    save_path="results/diff_test.csv",
    save_log=True,
)
```

Important behavior:

- If any of `x_df`, `obs_df`, or `var_df` are provided, then all three are required.
- CSV paths are read with `pandas.read_csv(index_col=0)`.
- `obs_df.index` must match `x_df.index`.
- `var_df.index` must match `x_df.columns`.
- When CSV or DataFrame inputs are used, `adata` is ignored and the function constructs a temporary `AnnData`.
- In CSV-backed mode, `layer` is reset to `None` and `use_raw` is disabled.

## Required keys and pairing rules

`groupby_key` is required for all current workflows.

`pair_by_key` is required when any paired or nested paired test is requested:

- `ttest_rel`
- `WilcoxonSigned`
- `ttest_rel_nested`
- `WilcoxonSigned_nested`

Paired alignment is based on the overlapping IDs in `adata.obs[pair_by_key]`. If the target and reference groups have no overlapping IDs, the function raises:

```text
No overlapping pairs on '<pair_by_key>'
```

If `groupby_key_ref_values` is `None`, the implementation uses all other categories in `groupby_key` that are not present in `groupby_key_target_values`.

## Examples by workflow

### Independent comparison

```python
results = adtl.diff_test(
    adata,
    groupby_key="Treatment",
    groupby_key_target_values=["drug"],
    groupby_key_ref_values=["vehicle"],
    tests=["ttest_ind", "mannwhitneyu"],
    add_values2results=True,
    sortby="Age",
    save_log=False,
)
```

This produces:

- group means and CVs,
- fold change and log2 fold change,
- `ttest_ind_*` columns,
- `mannwhitneyu_*` columns,
- independent-group normality columns,
- optional exported value and order columns such as `drug_values`, `vehicle_values`, `drug_Age_order`, and `vehicle_Age_order`.

### Paired comparison

```python
results = adtl.diff_test(
    adata,
    groupby_key="Treatment",
    groupby_key_target_values=["drug"],
    groupby_key_ref_values=["vehicle"],
    tests=["ttest_rel", "WilcoxonSigned"],
    pair_by_key="SubjectID",
    add_values2results=True,
    save_log=False,
)
```

This adds paired outputs such as:

- `SubjectID_order`,
- `drug_minus_vehicle_values`,
- `paired_PCTchange_target_ref`,
- `ttest_rel_mean_paired_fc_target_ref`,
- `WilcoxonSigned_mean_paired_l2fc_target_ref`,
- normality columns for paired differences.

### Nested paired comparison

```python
results = adtl.diff_test(
    adata,
    groupby_key="Treatment",
    groupby_key_target_values=["drug"],
    groupby_key_ref_values=["vehicle"],
    nested_groupby_key_target_values=[("drug", "predoseDrug")],
    nested_groupby_key_ref_values=[("vehicle", "predoseVeh")],
    tests=["ttest_rel_nested", "WilcoxonSigned_nested"],
    pair_by_key="AnimalID",
    add_values2results=True,
    add_adata_var_column_key_list=["feature_class"],
    save_log=False,
)
```

This adds nested outputs such as:

- `drug_minus_predoseDrug_values`,
- `vehicle_minus_predoseVeh_values`,
- `drug_diffcontrol_minus_vehicle_diffcontrol_values`,
- `ttest_rel_nested_*`,
- `WilcoxonSigned_nested_*`,
- normality columns for nested paired differences,
- merged feature annotations from `adata.var`.

Current implementation note:

- Nested comparison lists are accepted as lists of tuples, but the function currently uses the first tuple from `nested_groupby_key_target_values` and the first tuple from `nested_groupby_key_ref_values`.

## Output table shape

The returned `DataFrame` is indexed by feature name.

Before any statistics are computed, the function removes variables with zero expression across all observations in the chosen matrix. As a result, fully zero features do not appear in the result index.

Common column families include:

- `mean:{group}_{value}`
- `CVpct:{group}_{value}`
- `PCTchange_target_ref`
- `fc_target_ref`
- `l2fc_target_ref`
- `<test>_stat...`
- `<test>_pvals...`
- `<test>_pvals_FDR...`
- `<test>_constVAR...`
- `shapiro_pvals: ...`
- `ks_pvals: ...`

When `add_values2results=True`, the function also stores list-valued per-feature exports, for example:

- `<group>_values`
- `<target>_minus_<ref>_values`
- `<pair_by_key>_order`

For independent workflows:

- if `sortby` names a column in `adata.obs`, the order columns are named with that key, such as `drug_Age_order`;
- otherwise the order falls back to observation names.

## Save and storage behavior

### CSV output

If `save_table=True` and `save_path` is set, the results table is written to CSV:

```python
results = adtl.diff_test(
    adata,
    groupby_key="Treatment",
    groupby_key_target_values=["drug"],
    groupby_key_ref_values=["vehicle"],
    tests=["ttest_ind"],
    save_table=True,
    save_path="results/diff_test.csv",
    save_log=False,
)
```

### Log file

If `save_log=True`, the function writes a log file at:

- `<save_path>.log`

For example, `save_path="results/diff_test.csv"` produces `results/diff_test.csv.log`.

`save_log=True` requires `save_path` to be set.

### `adata.uns` storage

If `save_result_to_adata_uns_as_dict=True`, the function stores results at:

- `adata.uns["diff_test_results"][f"{groupby_key}_{targets}_over_{refs}"]`

Example key:

- `Treatment_drug_over_vehicle`

Important behavior:

- list-like columns ending in `_values` or `_order` are converted to JSON strings before storage in `adata.uns`;
- this keeps the payload writable through `h5ad` serialization;
- the returned `DataFrame` remains unchanged and still contains Python list values in memory.

## Validation and edge cases

The current implementation and regression tests lock in the following behavior.

### Alternate-input validation

- Partial alternate inputs are rejected. You cannot provide only `x_df` plus one of `obs_df` or `var_df`.
- `obs_df.index` must match `x_df.index`.
- `var_df.index` must match `x_df.columns`.

### Pairing validation

- Paired and nested paired tests require `pair_by_key`.
- If paired groups do not share any IDs on `pair_by_key`, the function raises a no-overlap error.
- If target/control and ref/control nested pair orders do not match, the function raises a mismatched-pairing error.

### Non-finite values

- `np.nan`, `np.inf`, and `-np.inf` are sanitized to `NaN` in the statistic paths before computing tests and summary metrics.
- This allows non-finite features to stay in the table if enough finite observations remain for the requested calculations.

### Constant-variance features

- Features that are constant within both compared groups, or constant in paired differences, remain in the output.
- For those features, the relevant `<test>_constVAR...` column is `True`.
- The corresponding test statistic and p-value columns are set to `NaN`.

## Practical notes

- `save_log` defaults to `True` in the function signature, so examples that do not want file output should pass `save_log=False`.
- `use_raw=True` switches the data source to `adata.raw` when available.
- `add_adata_var_column_key_list` merges selected `adata.var` columns onto the result table after the statistical outputs are built.
- `diff_test(...)` is a statistics-and-export utility. For model-based regression summaries, see [`_model_fit.md`](_model_fit.md).
