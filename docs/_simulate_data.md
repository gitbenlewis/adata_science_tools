# `_simulate_data`

Simulation helpers for covariate-driven `AnnData` examples.

The current simulation API lives in [`_simulate_data/covar_dependent_feature.py`](../_simulate_data/covar_dependent_feature.py) and is re-exported from `adata_science_tools._simulate_data`.

## Main entry points

- `sim_observations_covars`
- `sim_covar_dependent_features`
- `sim_covar_dependent_dataset`

## `sim_observations_covars`

`sim_observations_covars(...)` creates an `obs_df` with one column per requested covariate.

```python
from adata_science_tools._simulate_data import sim_observations_covars

obs_df = sim_observations_covars(
    obs_key_list=["Age", "case_control"],
    obs_covar_dist_params={
        "Age": {"dist": "normal", "mean": 50.0, "stdev": 10.0},
        "case_control": {"dist": "binomial", "prob": 0.5},
    },
    n_obs=100,
    random_seed=7,
)
```

Important behavior:

- `obs_covar_dist_params` is keyed by covariate name, not by distribution name.
- Supported distributions are `normal` and `binomial`; the typo alias `bionomial` is also accepted.
- Normal draws are stored as float values.
- Binomial draws are stored as `0/1` integer values.
- Observation names use `obs_names_prefix` with 1-based indexing, for example `obs_1`, `obs_2`, `obs_3`.

## `sim_covar_dependent_features`

`sim_covar_dependent_features(...)` treats the columns of `obs_df` as predictors, coerces them to numeric values, and generates a linear feature matrix with optional additive residual noise:

`X = obs_matrix @ beta_matrix.T + yint + residual`

```python
from adata_science_tools._simulate_data import sim_covar_dependent_features

X, var_df, obs_df, adata = sim_covar_dependent_features(
    obs_df=obs_df,
    var_names=["simulated_feature"],
    betas=[0.05, 5.0],
    yints=10.0,
    residual_stdev=1.0,
    random_seed=7,
    also_return_adata=True,
    save_adata_dataset=False,
)
```

Important behavior:

- A 1D `betas` sequence must match the number of covariates and is broadcast across all simulated features.
- A 2D `betas` array must have shape `(n_vars, n_covars)`.
- Scalar `yints` values are broadcast across all simulated features.
- `residual_mean` and `residual_stdev` accept scalar values or 1D sequences of length `n_vars`.
- `var_df` uses `var_names` as its index and stores `yint`, one `beta_<covariate>` column per predictor, and the residual-noise settings used for each feature.
- With the default `residual_stdev=0.0`, the function remains deterministic for a fixed `obs_df`, `betas`, and `yints`.
- When `adata` is returned, `adata.X` stores the observed noisy feature values, `adata.layers["linear_mean"]` stores the noiseless linear predictor, and `adata.layers["residual"]` stores the realized residual term.
- Non-numeric predictor columns are rejected at this layer.

## `sim_covar_dependent_dataset`

`sim_covar_dependent_dataset(...)` is the wrapper that first simulates covariates and then simulates features from those covariates.

```python
from adata_science_tools._simulate_data import sim_covar_dependent_dataset

X, var_df, obs_df, adata = sim_covar_dependent_dataset(
    obs_key_list=["Age", "case_control"],
    obs_covar_dist_params={
        "Age": {"dist": "normal", "mean": 50.0, "stdev": 10.0},
        "case_control": {"dist": "binomial", "prob": 0.5},
    },
    n_obs=100,
    random_seed=7,
    var_names=["simulated_feature"],
    betas=[0.05, 5.0],
    yints=10.0,
    residual_stdev=1.0,
    save_adata_dataset=False,
)
```

Important behavior:

- The wrapper returns `(X, var_df, obs_df, adata)`.
- `AnnData` is created whenever `also_return_adata=True` or `save_adata_dataset=True`.
- Residual-noise settings are passed through to `sim_covar_dependent_features(...)`.
- Dataset export reuses the package save helper and writes `.h5ad`, `.obs.csv`, `.var.csv`, and `.X.csv` sidecars.
- When `adata.layers` are present, the same export path also writes one CSV per layer, so this simulator now emits `linear_mean` and `residual` sidecars alongside the main matrix export.

## Repo example workflow

The repository now includes a config-driven simulation and plotting example in `example_simulated_data/`.

- [`example_simulated_data/scripts/simulate_1_var_covar_age.py`](../example_simulated_data/scripts/simulate_1_var_covar_age.py) generates one feature, `simulated_feature`, from two predictors, `Age` and `case_control`, plus config-driven residual `y` variance.
- The numeric `case_control` backend used for simulation is relabeled to public string values `'case'` and `'control'` before the dataset is saved.
- [`example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py`](../example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py) loads the saved `.h5ad` and plots `simulated_feature` versus `Age` with `case_control` as both `hue` and `subset_key`.
- Both scripts are driven from [`example_simulated_data/config/config.yaml`](../example_simulated_data/config/config.yaml).

### Example config knobs

The example config currently exposes these main simulation controls:

- `age_mean` and `age_stdev`: control the center and spread of the `Age` covariate.
- `beta` or `beta_age`: controls the age slope for `simulated_feature`.
- `case_control_prob`: controls the fraction of observations assigned to the `case` group before relabeling from `1/0` to `'case'/'control'`.
- `beta_case_control`: controls the expected vertical shift between `case` and `control` in the simulated feature.
- `residual_mean` and `residual_stdev`: control the additive residual noise around the linear mean model.
- `random_seed`: keeps the full simulated dataset deterministic across reruns.

In the current default baseline config:

- `beta_case_control: 2.0` sets the mean case-control separation.
- `residual_stdev: 1.0` prevents subgroup points from falling exactly on their fitted lines.
- `case_control_prob: 0.5` targets an approximately balanced case/control split.

Run the example from the repo root with:

```bash
python example_simulated_data/scripts/simulate_1_var_covar_age.py
python example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py
```

The default baseline outputs are:

- [`baseline.h5ad`](../example_simulated_data/results/simulate_1_var_covar_age/baseline/baseline.h5ad)
- [`baseline.obs.csv`](../example_simulated_data/results/simulate_1_var_covar_age/baseline/baseline.obs.csv)
- [`baseline.var.csv`](../example_simulated_data/results/simulate_1_var_covar_age/baseline/baseline.var.csv)
- [`baseline.layer.linear_mean.csv`](../example_simulated_data/results/simulate_1_var_covar_age/baseline/baseline.layer.linear_mean.csv)
- [`baseline.layer.residual.csv`](../example_simulated_data/results/simulate_1_var_covar_age/baseline/baseline.layer.residual.csv)
- [`baseline.png`](../example_simulated_data/results/plot_dotplot_simulate_1_var_covar_age/baseline/baseline.png)
