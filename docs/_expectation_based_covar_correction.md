# `_expectation_based_covar_correction`

Expectation-based covariate correction utilities for `AnnData` objects.

This module fits per-feature OLS models using columns from `adata.obs`, stores the fitted coefficients in a feature-indexed table, and then uses that table to:

- predict expected values for new observations,
- regress out covariate-driven components, or
- compute residual and ratio-based transforms relative to the fitted expectation.

The public entry points are:

- `calculate_expectations`
- `reconstruct_expectation_model_spec`
- `convert_ols_summary_to_expectation_df`
- `save_expectation_model_files`
- `predict_expectation`
- `regress_out`
- `excess_expectation`
- `regression_expectation_correction_adata`

## Core model

For each feature in `adata.var_names`, the module fits an OLS model against one or more predictors in `adata.obs`.

Conceptually, for observation `j` and feature `i`:

```text
expected[i, j] = intercept[i] + sum_k beta[i, k] * predictor_value[j, k]
```

Categorical predictors are expanded through the Patsy design matrix, so coefficient columns follow Patsy term names such as `Gender[T.Male]`.

## `calculate_expectations`

Use `calculate_expectations(...)` to fit the model and return the expectation coefficient table.

```python
import adata_science_tools as adtl

expectation_df = adtl.calculate_expectations(
    adata,
    predictors=["NHS_Case", "Age", "Gender"],
    layer="pgml",
    model_name="case_age_gender",
)
```

### Important behavior

- `fit_method="ols"` is currently the only supported fit mode.
- `predictors` is the primary argument; `covariates` is accepted as an alias.
- Rows can be filtered before fitting with either explicit filter arguments or `dataset_cfg`.
- The returned `DataFrame` stores a prediction spec in `expectation_df.attrs["model_spec"]`.
- If `save_result_to_adata_uns_as_dict=True`, the result is stored in `adata.uns["expectation_model"][model_name]`.

### Filtering before fit

You can fit on a subset of observations while leaving the input `adata` unchanged:

```python
expectation_df = adtl.calculate_expectations(
    adata,
    predictors=["Age"],
    layer="pgml",
    model_name="controls_age",
    filter_obs_boolean_column="use_for_expectation",
)
```

The same pattern can be driven through `dataset_cfg`:

```python
expectation_df = adtl.calculate_expectations(
    adata,
    dataset_cfg={
        "predictors": ["Age"],
        "layer": "pgml",
        "model_name": "controls_age",
        "filter_obs_boolean_column": "use_for_expectation",
    },
)
```

## Expectation table schema

`expectation_df` is indexed by `adata.var_names`.

Coefficient columns match fitted design terms:

- `intercept`
- one column per numeric predictor, for example `Age`
- one column per categorical contrast, for example `Gender[T.Male]`

Optional metadata columns are added when `include_metadata=True`:

- `fit_formula`
- `fit_nobs`
- `fit_r2`
- `fit_ok`
- `fit_warning`
- `fit_method`

This is the current implementation schema. The table does not use a `beta_<covariate>` naming convention.

## `model_spec`

The prediction contract is stored in `expectation_df.attrs["model_spec"]`. It contains the information needed to rebuild the design matrix later, including:

- `predictors`
- `formula_rhs`
- `design_terms`
- `coefficient_columns`
- `categorical_levels`
- `model_name`
- `layer`
- `use_raw`

This matters because a plain CSV reload loses `DataFrame.attrs`. If you reload the table from disk, you must provide the `model_spec` again unless a sibling YAML file is available.

## Saving and reloading artifacts

Use `save_expectation_model_files(...)` to write both artifacts together:

```python
csv_path, model_spec_path = adtl.save_expectation_model_files(
    expectation_df,
    "results/expectation_table.csv",
)
```

This writes:

- `results/expectation_table.csv`
- `results/expectation_table.model_spec.yaml`

Then either of these round-trips is valid:

```python
predicted = adtl.predict_expectation(adata, "results/expectation_table.csv")
```

```python
predicted = adtl.predict_expectation(
    adata,
    loaded_df,
    model_spec="results/expectation_table.model_spec.yaml",
)
```

If you only reload the CSV into a `DataFrame` and do not supply a `model_spec`, prediction will fail.

## `predict_expectation`

`predict_expectation(...)` returns a dense `numpy.ndarray` with shape `(adata.n_obs, adata.n_vars)`.

```python
predicted = adtl.predict_expectation(adata, expectation_df)
```

### Important behavior

- Features are aligned to `adata.var_names`.
- All required coefficient columns must be present and non-null.
- All predictor columns must exist in `adata.obs` unless `baseline` is used.
- Missing predictor values are rejected.
- Unseen categorical levels are rejected.

### Baseline prediction

`baseline` replaces the predictor values for every observation before building the design matrix:

```python
baseline_expected = adtl.predict_expectation(
    adata,
    expectation_df,
    include_intercept=False,
    baseline={"NHS_Case": 0.0, "Age": 40.0, "Gender": "Female"},
)
```

This is mainly used by `regress_out(..., flavor="obs_minus_exp_covar_baseline")`.

## `regress_out`

`regress_out(...)` writes a corrected matrix into `adata.layers[output_layer]` and returns either a copy or the original object, depending on `inplace`.

Supported flavors:

- `obs_minus_exp_covar`
- `obs_minus_exp_covar_baseline`

Examples:

```python
corrected = adtl.regress_out(
    adata,
    expectation_df,
    flavor="obs_minus_exp_covar",
    input_layer="pgml",
    output_layer="obs_minus_exp_covar",
    inplace=False,
)
```

```python
corrected = adtl.regress_out(
    adata,
    expectation_df,
    flavor="obs_minus_exp_covar_baseline",
    input_layer="pgml",
    output_layer="obs_minus_exp_covar_baseline",
    baseline={"NHS_Case": 0.0, "Age": 40.0, "Gender": "Female"},
    inplace=False,
)
```

### Flavor definitions

- `obs_minus_exp_covar`: subtracts the covariate-only component predicted with `include_intercept=False`.
- `obs_minus_exp_covar_baseline`: subtracts the difference between the observed covariate component and a baseline covariate component.

## `excess_expectation`

`excess_expectation(...)` computes residual or ratio-based transforms and stores the result in `adata.layers[output_layer]`.

Supported flavors:

- `obs_minus_exp_val`
- `obs_over_exp`
- `log_obs_over_exp`
- `log2_obs_over_exp`

Example:

```python
ratio = adtl.excess_expectation(
    adata,
    expectation_df,
    flavor="obs_over_exp",
    input_layer="pgml",
    output_layer="obs_over_exp",
    inplace=False,
)
```

When the input layer is itself log-transformed but the desired ratio is still on
the original scale, set `ratio_input_transform`:

```python
ratio = adtl.excess_expectation(
    adata,
    expectation_df,
    flavor="obs_over_exp",
    input_layer="ln_pgml",
    output_layer="ln_pgml_obs_over_exp",
    ratio_input_transform="ln",
    inplace=False,
)
```

Supported `ratio_input_transform` values:

- `none`: use the input layer values directly
- `ln`: convert observed and expected values with `exp(...)` before forming the ratio
- `log1p`: convert observed and expected values with `expm1(...)` before forming the ratio

### Numeric rules

- `obs_minus_exp_val` subtracts the full expected value, including the intercept.
- Ratio-based flavors require strictly positive expected values unless `eps` is provided.
- Log flavors also require strictly positive observed values unless `eps` is provided.
- When `eps` is set, it is added to both numerator and denominator before ratio or log computation.
- `ratio_input_transform` is applied before the ratio and any optional `log` or `log2` output.

## `convert_ols_summary_to_expectation_df`

If you already have an OLS summary table from `fit_smf_ols_models_and_summarize_adata(...)`, you can convert it into an expectation table:

```python
ols_summary_df = adtl.fit_smf_ols_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["NHS_Case", "Age", "Gender"],
    model_name="ols_roundtrip",
    include_fdr=False,
)

expectation_df = adtl.convert_ols_summary_to_expectation_df(
    ols_summary_df,
    predictors=["NHS_Case", "Age", "Gender"],
    model_name="ols_roundtrip",
    layer="pgml",
    reference_adata=adata,
)
```

This is useful when model fitting and expectation export happen in separate steps.

## `reconstruct_expectation_model_spec`

Use `reconstruct_expectation_model_spec(...)` when you have a loaded expectation table but no saved YAML sidecar:

```python
model_spec = adtl.reconstruct_expectation_model_spec(
    loaded_df,
    predictors=["NHS_Case", "Age", "Gender"],
    model_name="case_age_gender",
    layer="pgml",
    reference_adata=adata,
)
```

`reference_adata` or `reference_obs_df` is especially important when categorical predictor levels must be reconstructed.

## Wrapper workflow

`regression_expectation_correction_adata(...)` is the config-oriented wrapper for fit-plus-correct workflows.

Minimal dict-driven example:

```python
dataset_cfg = {
    "run_out_dir": "results",
    "filename": "cfg_corrected.h5ad",
    "calculate_expectations_params": {
        "predictors": ["Age"],
        "layer": "pgml",
        "model_name": "wrapper_cfg",
        "filter_obs_boolean_column": "use_for_expectation",
        "save_path": "results/cfg_expectation.csv",
    },
    "predict_expectation_params": {
        "baseline": {"Age": 45.0},
    },
    "regress_out_params": {
        "flavor": "obs_minus_exp_covar_baseline",
        "input_layer": "pgml",
        "output_layer": "cfg_corrected",
    },
}

corrected_adata = adtl.regression_expectation_correction_adata(
    adata,
    dataset_cfg=dataset_cfg,
)
```

### Wrapper behavior

- At least one of `regress_out_params` or `excess_expectation_params` must be active.
- If `expectation_df` is not supplied, the wrapper runs `calculate_expectations(...)`.
- The wrapper can save expectation artifacts and the corrected `h5ad`.
- `predict_expectation_params["baseline"]` is copied into `regress_out_params["baseline"]` when needed.
- When both `regress_out_params` and `excess_expectation_params` are active, the wrapper applies `regress_out(...)` first and then `excess_expectation(...)` on the working AnnData copy.
- `excess_expectation_params` supports the same `ratio_input_transform` values as the direct helper.

## Common validation errors

Expect these failure modes when inputs do not match the fitted model:

- predictor columns missing from `adata.obs`
- baseline values missing for one or more predictors
- missing or malformed `model_spec`
- missing sibling `.model_spec.yaml` for CSV-backed prediction
- unseen categorical levels at prediction time
- expectation tables missing required feature rows or coefficient columns
- `NaN` values in required coefficient columns
- non-positive expectation values for ratio-based transforms without `eps`
- non-positive observed values for log transforms without `eps`

## Practical workflow summary

1. Fit `expectation_df` with `calculate_expectations(...)`.
2. Save the CSV plus YAML sidecar with `save_expectation_model_files(...)` if the model will be reused.
3. Run `predict_expectation(...)` when you need the expected matrix itself.
4. Run `regress_out(...)` for covariate correction layers.
5. Run `excess_expectation(...)` for residual, ratio, or log-ratio layers.
