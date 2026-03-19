# `_model_fit`

Model-fitting utilities for feature-wise OLS and MixedLM summaries on `AnnData` objects.

This module provides two levels of API:

- low-level `*_wide(...)` functions that fit models from a combined observation-plus-feature `DataFrame`;
- higher-level `*_adata(...)` wrappers that build that table from `AnnData`, optionally filter observations first, and handle saving or `adata.uns` storage.

The current primary APIs are:

- `fit_smf_ols_models_and_summarize_wide`
- `fit_smf_ols_models_and_summarize_adata`
- `fit_smf_mixedlm_models_and_summarize_wide`
- `fit_smf_mixedlm_models_and_summarize_adata`

The `old_fit_smf_ols_models_and_summarize_adata` and `old_fit_smf_mixedlm_models_and_summarize_adata` helpers are legacy compatibility wrappers. They do not expose the newer filter or model-spec sidecar features and should not be treated as the preferred interface.

## OLS workflow

Use `fit_smf_ols_models_and_summarize_adata(...)` to fit one OLS model per feature.

```python
import adata_science_tools as adtl

ols_results = adtl.fit_smf_ols_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["NHS_Case", "Age", "Gender"],
    model_name="ols_unit",
    save_table=True,
    save_model_spec_yaml=True,
    save_path="results/ols_results.csv",
    save_result_to_adata_uns_as_dict=True,
    include_fdr=False,
)
```

Important behavior:

- `predictors` is normalized through a YAML-friendly list validator.
- Numeric-like predictor columns are coerced to numeric dtype before fitting so continuous covariates do not become dummy-coded categories.
- Per-feature OLS p-value columns use the `P>|t|` naming pattern.
- If `include_fdr=True`, per-term FDR columns are added with the `_FDR` suffix.

## MixedLM workflow

Use `fit_smf_mixedlm_models_and_summarize_adata(...)` to fit one mixed-effects model per feature.

```python
mixedlm_results = adtl.fit_smf_mixedlm_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["NHS_Case", "Age", "Gender"],
    group="Batch",
    model_name="mixedlm_unit",
    reml=False,
    save_table=True,
    save_model_spec_yaml=True,
    save_path="results/mixedlm_results.csv",
    save_result_to_adata_uns_as_dict=True,
    include_fdr=False,
)
```

Important behavior:

- `group` is required.
- MixedLM also expects a non-empty `predictors` list.
- Per-term MixedLM p-value columns use the `P>|z|` naming pattern.
- The summary includes grouping-specific fields such as `n_groups`, `Method`, random-effect variances, and, when available, per-group random effects.

## Filtering before fit

Both `*_adata(...)` wrappers can create a filtered working `AnnData` through:

- `dataset_cfg`
- `filter_obs_boolean_column`
- `filter_obs_column_key`
- `filter_obs_column_values_list`

Example:

```python
filtered_results = adtl.fit_smf_ols_models_and_summarize_adata(
    adata,
    dataset_cfg={
        "filter_obs_boolean_column": "use_for_expectation",
    },
    layer="pgml",
    predictors=["Age"],
    model_name="ols_filtered",
    return_filtered_adata=True,
)
```

Wrapper behavior:

- if any filter inputs are provided, the wrapper creates `work_adata` with `CFG_filter_adata_by_obs(...)`;
- the fit is run against `work_adata`, not the original `adata`;
- when `return_filtered_adata=True` and a filtered copy was created, the wrapper returns `(results, work_adata)` instead of only `results`.

## Low-level `*_wide(...)` APIs

The wide functions operate on a single `obs_X_df` table that contains:

- one column per feature, and
- one column per predictor, plus `group` for MixedLM.

Use them when the combined table already exists or when you want to bypass `AnnData` handling:

- `fit_smf_ols_models_and_summarize_wide(obs_X_df, feature_columns, predictors, ...)`
- `fit_smf_mixedlm_models_and_summarize_wide(obs_X_df, feature_columns, predictors, group=..., ...)`

The `*_adata(...)` wrappers build this table internally with `make_df_obs_adataX(...)`, so most users should start with the `AnnData` APIs unless they already have the combined matrix.

## Result table schema

Both OLS and MixedLM result tables are indexed by feature name and include `var_names` as the first column.

Column names are prefixed with `model_name`, so changing `model_name` changes the entire result schema namespace.

Representative OLS column patterns include:

- `<model_name>_Formula`
- `<model_name>_Converged`
- `<model_name>_Warnings`
- `<model_name>_nobs`
- `<model_name>_R-squared`
- `<model_name>_Coef_<term>`
- `<model_name>_StdErr_<term>`
- `<model_name>_tStat_<term>`
- `<model_name>_P>|t|_<term>`
- `<model_name>_P>|t|_<term>_FDR`

Representative MixedLM column patterns include:

- `<model_name>_Formula`
- `<model_name>_Converged`
- `<model_name>_Warnings`
- `<model_name>_nobs`
- `<model_name>_n_groups`
- `<model_name>_Method`
- `<model_name>_Var_RE_<term>`
- `<model_name>_Var_Residual`
- `<model_name>_Coef_<term>`
- `<model_name>_StdErr_<term>`
- `<model_name>_tStat_<term>`
- `<model_name>_P>|z|_<term>`
- `<model_name>_P>|z|_<term>_FDR`

The exact set of columns depends on:

- the fitted design terms,
- the chosen `model_name`,
- whether warnings or random effects are present,
- whether `include_fdr=True`.

## Sidecar YAML model specs

The current `*_adata(...)` wrappers can write a sibling `.model_spec.yaml` file when:

- `save_table=True`
- `save_model_spec_yaml=True`
- `save_path` is provided

If `save_model_spec_yaml=True` is requested without `save_table=True` and `save_path`, the wrappers raise a `ValueError`.

Example:

```python
ols_results = adtl.fit_smf_ols_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["NHS_Case", "Age", "Gender"],
    model_name="ols_unit",
    save_table=True,
    save_model_spec_yaml=True,
    save_path="results/ols_results.csv",
)
```

This writes:

- `results/ols_results.csv`
- `results/ols_results.model_spec.yaml`

The YAML captures the fitting contract for downstream consumers, including:

- `fit_method`
- `model_name`
- `predictors`
- `layer`
- `use_raw`
- `formula_rhs`
- `coefficient_terms`
- `coefficient_columns`

For MixedLM sidecars, the YAML also includes:

- `group`
- `reml`

This is the metadata used by the expectation-correction workflow documented in [`_expectation_based_covar_correction.md`](_expectation_based_covar_correction.md), which can consume OLS summary outputs and derived coefficient tables.

## `adata.uns` storage

When `save_result_to_adata_uns_as_dict=True`, the wrappers store results under model-type-specific namespaces.

OLS storage:

- `work_adata.uns["ols_model_results"][f"OLS_model_results_{model_name}"]`
- `work_adata.uns["ols_model_specs"][f"OLS_model_results_{model_name}"]` when `save_model_spec_yaml=True`

MixedLM storage:

- `work_adata.uns["mixedlm_model_results"][f"mixedlm_model_results_{model_name}"]`
- `work_adata.uns["mixedlm_model_specs"][f"mixedlm_model_results_{model_name}"]` when `save_model_spec_yaml=True`

If filtering created a new `work_adata` and `save_results_to_original_adata_uns=True`, the same results and model specs are also written into the original `adata.uns`.

## Behavior and constraints

The current implementation locks in these behaviors.

### Predictor handling

- Wrapper inputs such as `predictors` and `add_adata_var_column_key_list` are normalized with `_ensure_list(...)`.
- YAML-style lists are expected.
- Passing a single string instead of a list raises a `TypeError`.
- Numeric-like predictors are coerced to numeric dtype before fitting.

### OLS-specific notes

- OLS treats missing and infinite values as `NaN` and fits only complete-case rows for each feature.
- If a feature has no complete-case rows after that cleanup, the OLS wide function does not raise.
- Instead, it records a summary row with `Converged=False` and a warning message explaining why the fit was skipped.

### MixedLM-specific notes

- MixedLM raises early when required fit conditions are not met.
- Missing feature, predictor, or group columns raise a `ValueError`.
- Zero complete-case rows for a feature raise a `ValueError`.
- Fewer than two non-empty groups after filtering also raises a `ValueError`.
- MixedLM tries to collect random effects, but if covariance inversion fails it records that failure in the warnings field and continues building the summary row.

## Feature annotation merge

Both `*_adata(...)` wrappers can merge selected `adata.var` columns into the result table through `add_adata_var_column_key_list`.

Example:

```python
ols_results = adtl.fit_smf_ols_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["Age"],
    model_name="ols_with_var",
    add_adata_var_column_key_list=["gene_name", "feature_class"],
)
```

Missing `adata.var` keys are skipped with a warning or print message, depending on the code path.

## Practical guidance

- Use the `*_adata(...)` wrappers for most workflows.
- Use the `*_wide(...)` functions when you already have an `obs_X_df` table.
- Use `save_model_spec_yaml=True` only when you also want a saved CSV artifact.
- Use OLS outputs as the upstream summary source for expectation-model workflows described in [`_expectation_based_covar_correction.md`](_expectation_based_covar_correction.md).
