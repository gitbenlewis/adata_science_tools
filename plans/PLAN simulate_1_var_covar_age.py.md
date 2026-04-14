# Finish `simulate_1_var_covar_age.py`

1. Summary: fill out [simulate_1_var_covar_age.py](/home/ubuntu/projects/gitbenlewis/adata_science_tools/example_simulated_data/scripts/simulate_1_var_covar_age.py) as a config-driven runner that keeps the existing script scaffold, reads `simulate_1_var_covar_age_params` from [config.yaml](/home/ubuntu/projects/gitbenlewis/adata_science_tools/example_simulated_data/config/config.yaml), and uses `sim_covar_dependent_dataset(...)` from `_simulate_data` to generate one-feature, one-covariate Age-based `AnnData` outputs per named run.

2. Diff Summary: update the script to add a real run loop and a local import fallback when `code_library` is unavailable; update `example_simulated_data/config/config.yaml` to include defaults and at least one runnable example under `simulate_1_var_covar_age__runs`; leave the package-root exports and the simulator module unchanged.

3. Key Changes: preserve the current logging, dataclass, and script-level setup; keep the `code_library` import attempt, but on `ImportError` append `REPO_ROOT.parent.parent` to `sys.path`, then import `adata_science_tools` locally and import `sim_covar_dependent_dataset` from `adata_science_tools._simulate_data`; in `__main__`, read `DEFAULT_PARAMS = CFG["simulate_1_var_covar_age_params"].get("default_params") or {}` and `RUNS = CFG["simulate_1_var_covar_age_params"].get("simulate_1_var_covar_age__runs") or {}`, iterate with `ChainMap(run_values, DEFAULT_PARAMS, {"run": False})`, skip runs with `run: false`, derive `output_path` as `Path(chained.get("output_path", OUTPUT_DIR / run_key / run_key))`, and call `sim_covar_dependent_dataset(...)` with an age-specific mapping: `obs_key_list=["Age"]`, `obs_covar_dist_params={"Age": {"dist": "normal", "mean": age_mean, "stdev": age_stdev}}`, `var_names=[var_name]`, `betas=[beta]`, `yints=yint`, plus `n_obs`, `obs_names_prefix`, `random_seed`, `save_adata_dataset`, `also_return_adata`, and `save_obs_df`; if `save_obs_df` is true and `save_obs_df_path` is omitted, derive `output_path.with_name(f"{output_path.name}.obs_only")`; log the resolved params and resulting `adata`, `obs_df`, `var_df`, and `X` shapes; keep the script constrained to one covariate named `Age` and one feature, and raise a clear `ValueError` if config tries to supply multiple covariates, a list-like `var_name`, or a non-scalar `beta`. The config contract should be:
   ```yaml
   simulate_1_var_covar_age_params:
     repo_results_dir: /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_simulated_data/results/simulate_1_var_covar_age/
     default_params:
       n_obs: 100
       obs_names_prefix: obs_
       random_seed: 7
       age_mean: 50.0
       age_stdev: 10.0
       beta: 0.05
       yint: 10.0
       var_name: simulated_feature
       save_obs_df: false
       save_adata_dataset: true
       also_return_adata: true
     simulate_1_var_covar_age__runs:
       baseline:
         run: true
       older_shift:
         run: false
         age_mean: 65.0
         beta: 0.10
   ```

4. Test Plan: do not add a new script unit test in this pass, because preserving the current module-level script setup would make import-based testing disproportionately invasive; rely on the existing simulator coverage in [test_simulate_data.py](/home/ubuntu/projects/gitbenlewis/adata_science_tools/tests/test_simulate_data.py) for the library path, then use a script smoke test as the acceptance check: run `python example_simulated_data/scripts/simulate_1_var_covar_age.py`, verify that `baseline.h5ad`, `baseline.obs.csv`, `baseline.var.csv`, and `baseline.X.csv` are created under `.../results/simulate_1_var_covar_age/baseline/`, and confirm the saved `obs` has one `Age` column and the saved `var` has one row with `yint` and `beta_Age`.

5. Assumptions: keep the script Age-specific and use a normal Age distribution only; do not add noise, extra covariates, or top-level `adata_science_tools` exports in this pass; leave the current optional `run_GSEApy_wrapper` and `RNAseq_analysis` imports as best-effort logging only; leave `config_template.yaml` unchanged to keep the diff minimal unless you explicitly want template parity in the same change.
