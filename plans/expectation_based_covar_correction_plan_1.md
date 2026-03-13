Save the text below as **`expectation_based_covar_correction_plan_1.md`**.

---

# expectation_based_covar_correction_plan_1.md

## Purpose — short

A practical, repo-aligned plan to add an **expectation-based covariate correction** feature to `adata_science_tools` that:

* reuses existing helpers (`adtl.fit_smf_ols_models_and_summarize_adata`, `adtl.CFG_filter_adata_by_obs`) and config-driven scripts,
* fully supports being driven from a `config.yaml` (no tuples or Python-only types required), and
* allows *subsetting the data prior to building the expectation model* (for example: build an expectation model using only Controls and then apply it to full cohort).

This document adapts the original feature plan to the existing repo layout and config style. The original plan content is included as the conceptual spec and is extended here to be implementation-ready for `adata_science_tools`. 

---

## Files inspected and assumptions

* Expectation-feature spec (original plan). 
* Model fit implementation containing `fit_smf_ols_models_and_summarize_adata` + wide-fitting helpers (source of current model-fit behavior and result format). 
* Existing `CFG_filter_adata_by_obs()` implementation (filter-by-obs helper) located at `_preprocessing/_adata_row_operations.py`. This will be re-used / hardened as needed. 
* Example project config showing how your pipelines currently provide YAML-driven model-fit runs (config). Use this as canonical config layout. 

Assume function signatures may already have been updated for filter-args per earlier iterations (if not, the plan shows exact changes to make).

---

## High-level design

1. **Expectation model definition**

   * Feature-wise linear expectation:
     `expected_ij = intercept_i + sum_k beta_i,k * covar_j,k`
   * Expectations stored as a DataFrame indexed by `adata.var_names`, columns `intercept` and `beta_<covar>` plus optional metadata columns.

2. **Two main user-facing operations**

   * `regress_out(...)` — covariate correction flavors (`obs_minus_exp_covar`, `obs_minus_exp_covar_baseline`).
   * `excess_expectation(...)` — residuals and ratio-based flavors (`obs_minus_exp_val`, `obs_over_exp`, `log_obs_over_exp`, `log2_obs_over_exp`).

3. **Config-driven flow**

   * Use project `config.yaml` sections (like the  example) to define runs. Each run can:

     * provide `adata_path`, `layer`, `feature_columns`, `predictors`, `model_name`,
     * provide `filter_obs_*` keys to subset data prior to fitting (or supply a `dataset_cfg` object),
     * request results to be saved to `adata.uns` or to disk.

4. **No tuples in config**

   * YAML config uses only JSON-native types: strings, lists, numbers, booleans, dicts, null.
   * Where the code internally prefers tuples (e.g., suffixes), functions will accept a YAML list and normalize to tuple internally.

---

## API surfaces and signatures (repo-compatible)

Below are recommended public APIs to implement or finalize in `adata_science_tools`. They are intentionally small and map to the original plan while using existing repo helpers.

### calculate_expectations(...)

Fits feature-wise models and returns an expectations DataFrame.

```
def calculate_expectations(
    adata,
    covariates: list[str],
    layer: str = None,
    use_raw: bool = False,
    fit_method: str = "ols",
    var_names: list[str] = None,
    predictors_transform: dict = None,   # optional: per-covariate transform config
    save_path: str = None,               # optional CSV path
    include_metadata: bool = True        # add fit metadata columns
) -> pd.DataFrame:
    """
    Returns expectation_df indexed by adata.var_names.
    Columns: 'intercept', 'beta_<covar>' for each covariate plus optional fit metadata.
    """
```

**Implementation notes:**

* For `fit_method == "ols"` use the existing `adtl.fit_smf_ols_models_and_summarize_adata()` or its `*_wide()` helper:

  * Construct an `obs_X_df` from `make_df_obs_adataX(adata, layer=..., use_raw=...)`.
  * Call `fit_smf_ols_models_and_summarize_wide()` (already in repo) to fit per-feature models and extract intercept and betas.
* Ensure returned DataFrame columns follow the `intercept` and `beta_<covar>` naming convention.
* Add optional `fit_*` metadata columns (`fit_method`, `model_formula`, `fit_n`, `fit_r2`, `fit_ok`, `fit_message`).

---

### predict_expectation(...)

Reconstruct expected matrix for every observation.

```
def predict_expectation(
    adata,
    expectation_df,
    covariates: list[str] = None,
    include_intercept: bool = True,
    baseline: dict = None
) -> np.ndarray:
    """
    Returns expected matrix shaped (n_obs, n_vars). If baseline provided and
    include_intercept True, it is applied only for covariate-baseline variants.
    """
```

**Implementation notes:**

* Validate `expectation_df.index` matches `adata.var_names` and that `beta_<cov>` columns exist for all requested covariates.
* Compute `expected_ij = intercept_i + sum_k beta_i,k * covar_j,k` in a numerically stable vectorized manner (matrix multiplication: obs x covariates  × covariates x features).

---

### regress_out(...)

```
def regress_out(
    adata,
    expectation_df,
    covariates: list[str] = None,
    baseline: dict = None,
    flavor: str = "obs_minus_exp_covar",
    input_layer: str = None,
    output_layer: str = None,
    inplace: bool = False
) -> anndata.AnnData:
    """
    Apply covariate-correction flavors and store output in adata.layers[output_layer].
    Flavors: obs_minus_exp_covar, obs_minus_exp_covar_baseline
    """
```

**Notes:**

* Compute covariate-only expected matrix `C` and baseline `C_baseline` when needed.
* `obs_minus_exp_covar = Y - C`
* `obs_minus_exp_covar_baseline = Y - (C - C_baseline)`

---

### excess_expectation(...)

```
def excess_expectation(
    adata,
    expectation_df,
    covariates: list[str] = None,
    flavor: str = "obs_minus_exp_val",
    input_layer: str = None,
    output_layer: str = None,
    inplace: bool = False,
    eps: float = None
) -> anndata.AnnData:
    """
    Flavors: obs_minus_exp_val, obs_over_exp, log_obs_over_exp, log2_obs_over_exp
    """
```

**Numeric safeguards:**

* `obs_over_exp = Y / E` — require `E` strictly positive or allow `eps` to adjust denom; raise if `E <= 0` and `eps` is None.
* For logs require `Y > 0` and `E > 0` (or use `eps`).

---

## Config schema and examples

Use the  YAML style as canonical. The feature expects runs to be described under `model_fit_params` or a new `expectation_params` section. Key constraints: use lists, not tuples.

Example minimal `config.yaml` fragment for an expectation run:

```
expectation_params:
  repo_results_dir: 
  default_params:
    layer: pgml
    covariates: ['Age', 'Gender']
    save_path: null

  expectation_runs:
    Control_Age_Only:
      run: true
      adata_path: 
      layer: pgml
      covariates: ['Age']
      predictors: ['Age']
      model_name: Control_Age
      filter_obs_boolean_column: null
      filter_obs_column_key: NHS_group
      filter_obs_column_values_list: ['Control']
      save_path: 
      save_result_to_adata_uns_as_dict: true
      add_adata_var_column_key_list: ['gene_name']
```

Notes:

* `filter_obs_*` keys allow subsetting the adata *before* fitting expectations. The existing `CFG_filter_adata_by_obs()` is designed to accept exactly those keys; the plan uses it directly. 
* `predictors` is a YAML list. If a single predictor is needed use a one-element list: `['Age']`.
* No tuples. If any function previously took a tuple default, update the function signature to accept a list or None and coerce internally to a tuple when needed.

---

## Integration with existing repo helpers

1. **Subsetting**
   Use `adtl.CFG_filter_adata_by_obs()` to subset adata before building the expectation model, exactly as your example `con_adata = CFG_filter_adata_by_obs(adata, **run_values)`. Ensure the helper accepts either:

   * a `dataset_cfg` dict (run block from YAML) or
   * explicit args: `filter_obs_boolean_column`, `filter_obs_column_key`, `filter_obs_column_values_list`.
     The repo already contains this helper; harden it to do numeric-aware matching and to return either a copy or a view based on a `copy` flag. 

2. **Model fitting**
   For "fit" logic, call `adtl.fit_smf_ols_models_and_summarize_adata()` with the filtered `work_adata` (or allow the `fit` function to accept `dataset_cfg`/filter args itself and create the filtered `work_adata` internally). The repo contains this function and its wide-fitting counterpart; re-use it to extract betas and intercepts for all features. The existing function returns rich model summaries, which can be repurposed to construct the expectation DataFrame. 

3. **No code duplication**
   Reuse the repo's summary table code to populate `intercept` and `beta_<covar>` columns. Where the existing function produces many columns, select what you need and save extra metadata columns as recommended.

---

## Internal normalization and YAML compatibility (no tuples)

* Any public function receiving `merge_suffixes` or similar should accept a `list` and internally call:

  def _normalize_merge_suffixes(merge_suffixes):
  if merge_suffixes is None:
  return ("", "_feature_results")
  if isinstance(merge_suffixes, (list, tuple)) and len(merge_suffixes) == 2:
  return (str(merge_suffixes[0]), str(merge_suffixes[1]))
  raise ValueError("merge_suffixes must be a list/tuple of two strings or None")

* Add `_ensure_list()` helper to validate YAML list arguments (e.g., `predictors`, `covariates`), raising clear errors when a single comma-separated string is passed.

---

## Subsetting workflow (explicit sequence)

1. Load `adata = anndata.read_h5ad(run['adata_path'])`.
2. Create `work_adata`:

   * either `work_adata = adtl.CFG_filter_adata_by_obs(adata, **run)` or
   * call `adtl.fit_*` with filter args and let it create `work_adata` internally. Both patterns should produce identical `expectation_df`.
3. Fit expectation model using `calculate_expectations()` (which internally calls `fit_smf_ols_models_and_summarize_adata()`).
4. Save `expectation_df` to CSV and/or `work_adata.uns["expectation_model"]` per config.
5. Apply transforms: `regress_out(...)`, `excess_expectation(...)` to `full_adata` or any adata you want to transform.

The repo already contains scripts that iterate YAML-defined runs and call `adtl.fit_smf_ols_models_and_summarize_adata()`; align the new expectation-run loop to that pattern (see `make_model_fit_tables.py`). 

---

## Storage and naming conventions

* **Layers:** use canonical layer names from original plan:

  * `obs_minus_exp_covar`
  * `obs_minus_exp_covar_baseline`
  * `obs_minus_exp_val`
  * `obs_over_exp`
  * `log_obs_over_exp`
  * `log2_obs_over_exp`

* **Model metadata** in `adata.uns["expectation_model"]`:

  {
  "covariates": ["Age","Gender"],
  "fit_method": "ols",
  "baseline": {"Age": 50, "Gender": 0},
  "var_index": "adata.var_names",
  "expectation_df_path": "/path/Control_Age_expectations.csv"
  }

* **Coefficients** mirrored optionally into `adata.var` or `adata.varm["expectation_betas"]` for quick per-feature access.

---

## Validation and numerical safeguards

* Validate shapes and alignment: `expectation_df.index.equals(adata.var_names)` or provide alignment step (reindex expectation_df to adata.var_names with informative errors on missing features).
* Validate covariates presence: every `covar` must be a column in `adata.obs`.
* For ratio/log transforms:

  * require `expected > 0` and `observed > 0` unless caller supplies `eps` (floating small positive).
  * explicitly document the `eps` behavior: add to denominator if `eps` specified; if `eps` is None, raise an error on nonpositive denom/observed values.
* For `obs_minus_exp_covar_baseline`: require `baseline` keys match covariates; supply a default baseline computed from `adata.obs` if `baseline == "data_mean"`.

---

## Unit tests (suggested)

1. **CFG filter tests**

   * boolean filter, column+values string match, column+values numeric match (string fallback), missing column -> KeyError.

2. **calculate_expectations equivalence**

   * prefilter + calculate_expectations == calculate_expectations called with filter args (results equal within tolerance). Use a small synthetic `AnnData`.

3. **regress_out correctness**

   * compute covariate-only expectation matrix `C` and assert `obs_minus_exp_covar` layer equals `Y - C`.

4. **excess expectation numeric safeguards**

   * ensure ratio/log functions error or use `eps`, test boundary cases `E==0`, `Y==0`, negative values.

5. **config parsing**

   * test YAML with lists and test error on single string for `predictors` (verify clear error message).

---

## Migration notes and backwards compatibility

* Existing code that used tuples in function call sites will continue to work if you accept tuples as well; but document YAML syntax as lists.
* `merge_feature_results_to_var()` and any other function with tuple defaults should be updated to accept lists and normalize to tuples internally.
* Where `fit_smf_ols_models_and_summarize_adata()` or mixedlm variants accept `dataset_cfg` or filter args, update example scripts to show both patterns: prefilter then fit, and fit-with-filter-args convenience call.

---

## Implementation step-by-step (developer checklist)

1. **Hardening filter helper**

   * Update `_preprocessing/_adata_row_operations.py:CFG_filter_adata_by_obs` to accept `dataset_cfg`, `copy` flag, numeric-aware matching, `logger`, and clear KeyError messages. (File exists — refine rather than create duplicate.) 

2. **Expectation fit glue**

   * Add `calculate_expectations()` to `anndata_tools` core (module `anndata_tools/_expectation.py`).
   * Internally call `adtl.fit_smf_ols_models_and_summarize_adata()` (or `fit_smf_ols_models_and_summarize_wide()`), and build standardized `expectation_df`.

3. **Predict + transform**

   * Implement `predict_expectation()`, `regress_out()`, `excess_expectation()` in `_expectation.py` with vectorized implementations.

4. **YAML-driven runner**

   * Add `scripts/make_expectation_tables.py` mirroring `make_model_fit_tables.py` where runs are defined in `config.yaml` under `expectation_params.expectation_runs`. Reuse logging/runner scaffolding used in the repo. See existing example scripts for style. 

5. **Normalization helpers**

   * Add `_ensure_list()` and `_normalize_merge_suffixes()` helpers in `_utils.py` or `_cfg_helpers.py`.

6. **Unit tests**

   * Add tests under `tests/test_expectation.py` covering the key cases above.


7. **Docs & examples**

   * Update README with a minimal example that uses config yaml sytle `config.yaml` to run the expectation fit and apply `obs_minus_exp_covar` to a full dataset.



---

## Example call patterns (user-friendly)

**A. Prefilter then fit (explicit)**

```
adata = anndata.read_h5ad(run_values['adata_path'])
con_adata = adtl.CFG_filter_adata_by_obs(adata, **run_values).copy()
expectation_df = adtl.calculate_expectations(
    con_adata,
    covariates=run_values['predictors'],
    layer=run_values['layer']
)
# store expectation metadata and apply transform to full adata
adata.uns['expectation_model'] = {...}
adtl.regress_out(adata, expectation_df, covariates=run_values['predictors'], flavor='obs_minus_exp_covar', output_layer='obs_minus_exp_covar')
```

**B. Fit-with-filter-args (convenience — implemented inside the fit-call)**

```
expectation_df = adtl.calculate_expectations(
    adata,
    covariates=['Age'],
    layer='pgml',
    dataset_cfg={'filter_obs_column_key': 'NHS_group', 'filter_obs_column_values_list': ['Control']}
)
```

Both patterns must produce identical `expectation_df` within numerical tolerance.

---

## Final notes and justification

* The existing repo already has the model-fitting machinery and the filter helper; the job is mainly an integration, standardization (expectation DataFrame schema), YAML compatibility, and adding the transform primitives. Reusing `fit_smf_ols_models_and_summarize_adata()` is both pragmatic and robust because it already produces rich summaries you can repurpose. 
* The  config demonstrates this repo’s config-driven pattern — align the expectation-run schema to that pattern. Your YAML-driven workflows will then plug directly into the new scripts. 

---


