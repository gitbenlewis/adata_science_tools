"""Simulate covariate-driven feature datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .._io._IO import save_dataset


LOGGER = logging.getLogger(__name__)

_save_dataset = save_dataset

__all__ = [
    "sim_observations_covars",
    "sim_covar_dependent_features",
    "sim_covar_dependent_dataset",
]


def sim_observations_covars(
    obs_key_list: Sequence[str] | None = None,
    obs_covar_dist_params: dict[str, dict[str, float | str]] | None = None,
    n_obs: int = 100,
    obs_names_prefix: str = "obs_",
    save_obs_df: bool = False,
    save_obs_df_path: str | Path = "obs_df",
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate an observation dataframe with covariates defined by distribution specs.

    Parameters
    ----------
    obs_key_list
        Ordered list of covariate names to generate.
    obs_covar_dist_params
        Mapping from covariate name to a specification like
        ``{"dist": "normal", "mean": 10, "stdev": 5}`` or
        ``{"dist": "binomial", "prob": 0.5}``.
    n_obs
        Number of observations to simulate.
    obs_names_prefix
        Prefix used to build a 1-based observation index.
    save_obs_df
        If True, write the simulated ``obs_df`` to CSV.
    save_obs_df_path
        Output path used when ``save_obs_df=True``.
    random_seed
        Optional seed for deterministic simulation.
    """
    if obs_key_list is None:
        obs_key_list = ["Age", "gender"]
    if obs_covar_dist_params is None:
        obs_covar_dist_params = {
            "Age": {"dist": "normal", "mean": 10, "stdev": 5},
            "gender": {"dist": "binomial", "prob": 0.5},
        }
    if isinstance(obs_key_list, str):
        raise TypeError("obs_key_list must be a sequence of covariate names, not a single string.")

    obs_key_list = list(obs_key_list)
    if not obs_key_list:
        raise ValueError("obs_key_list must contain at least one covariate name.")
    if len(set(obs_key_list)) != len(obs_key_list):
        raise ValueError("obs_key_list contains duplicate covariate names.")
    if not isinstance(obs_covar_dist_params, dict):
        raise TypeError("obs_covar_dist_params must be a dict keyed by covariate name.")
    if not isinstance(obs_names_prefix, str):
        raise TypeError("obs_names_prefix must be a string.")

    n_obs = int(n_obs)
    if n_obs <= 0:
        raise ValueError("n_obs must be a positive integer.")

    rng = np.random.default_rng(random_seed)
    obs_data: dict[str, np.ndarray] = {}

    for covariate_name in obs_key_list:
        if covariate_name not in obs_covar_dist_params:
            raise ValueError(
                f"obs_covar_dist_params is missing a distribution spec for covariate '{covariate_name}'."
            )
        dist_spec = obs_covar_dist_params[covariate_name]
        if not isinstance(dist_spec, dict):
            raise TypeError(
                f"Distribution spec for covariate '{covariate_name}' must be a dict."
            )
        if "dist" not in dist_spec:
            raise ValueError(
                f"Distribution spec for covariate '{covariate_name}' must include a 'dist' key."
            )

        dist_name = str(dist_spec["dist"]).lower()
        if dist_name == "normal":
            if "mean" not in dist_spec or "stdev" not in dist_spec:
                raise ValueError(
                    f"Normal distribution spec for covariate '{covariate_name}' requires 'mean' and 'stdev'."
                )
            mean = float(dist_spec["mean"])
            stdev = float(dist_spec["stdev"])
            if stdev < 0:
                raise ValueError(f"stdev must be >= 0 for covariate '{covariate_name}'.")
            obs_data[covariate_name] = rng.normal(loc=mean, scale=stdev, size=n_obs).astype(float)
            continue

        if dist_name in {"binomial", "bionomial"}:
            if "prob" not in dist_spec:
                raise ValueError(
                    f"Binomial distribution spec for covariate '{covariate_name}' requires 'prob'."
                )
            prob = float(dist_spec["prob"])
            if prob < 0 or prob > 1:
                raise ValueError(f"prob must be between 0 and 1 for covariate '{covariate_name}'.")
            obs_data[covariate_name] = rng.binomial(n=1, p=prob, size=n_obs).astype(int)
            continue

        raise ValueError(
            f"Unsupported distribution '{dist_spec['dist']}' for covariate '{covariate_name}'. "
            "Supported distributions are 'normal' and 'binomial'."
        )

    obs_index = [f"{obs_names_prefix}{idx}" for idx in range(1, n_obs + 1)]
    obs_df = pd.DataFrame(obs_data, index=obs_index)

    if save_obs_df:
        obs_path = Path(save_obs_df_path)
        if obs_path.suffix == "":
            obs_path = obs_path.with_suffix(".csv")
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving obs_df to %s", obs_path)
        obs_df.to_csv(obs_path)

    return obs_df


def sim_covar_dependent_features(
    obs_df: pd.DataFrame,
    var_names: Sequence[str] | str = ("covar_dependent_feature",),
    betas: Sequence[float] | Sequence[Sequence[float]] = (0.05, 5.0),
    yints: float | Sequence[float] = 10,
    also_return_adata: bool = True,
    save_adata_dataset: bool = True,
    output_path: str | Path | None = None,
    residual_dist: str = "normal",
    residual_mean: float | Sequence[float] = 0.0,
    residual_stdev: float | Sequence[float] = 0.0,
    random_seed: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, ad.AnnData | None]:
    """
    Simulate feature values from an observation dataframe of covariates.

    The generated feature matrix is the linear mean model plus optional residuals:
    ``X = obs_matrix @ beta_matrix.T + yint + residual``.
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise TypeError("obs_df must be a pandas.DataFrame.")
    if obs_df.empty:
        raise ValueError("obs_df must contain at least one observation.")
    if obs_df.shape[1] == 0:
        raise ValueError("obs_df must contain at least one covariate column.")

    predictor_names = obs_df.columns.tolist()
    numeric_obs_df = obs_df.apply(pd.to_numeric, errors="coerce")
    bad_columns = [column for column in predictor_names if numeric_obs_df[column].isna().any()]
    if bad_columns:
        raise TypeError(
            "obs_df contains non-numeric or missing predictor values in columns: "
            f"{bad_columns}."
        )
    predictor_matrix = numeric_obs_df.to_numpy(dtype=float, copy=True)
    n_obs, n_covars = predictor_matrix.shape

    if isinstance(var_names, str):
        var_names = [var_names]
    else:
        var_names = list(var_names)
    if not var_names:
        raise ValueError("var_names must contain at least one feature name.")
    if len(set(var_names)) != len(var_names):
        raise ValueError("var_names contains duplicate feature names.")

    n_vars = len(var_names)
    beta_array = np.asarray(betas, dtype=float)
    if beta_array.ndim == 1:
        if beta_array.shape[0] != n_covars:
            raise ValueError(
                f"1D betas must have length {n_covars} to match obs_df covariates; "
                f"got {beta_array.shape[0]}."
            )
        beta_matrix = np.tile(beta_array, (n_vars, 1))
    elif beta_array.ndim == 2:
        if beta_array.shape != (n_vars, n_covars):
            raise ValueError(
                f"2D betas must have shape ({n_vars}, {n_covars}); got {beta_array.shape}."
            )
        beta_matrix = beta_array
    else:
        raise ValueError("betas must be a 1D or 2D numeric sequence.")

    yint_array = np.asarray(yints, dtype=float)
    if yint_array.ndim == 0:
        yint_vector = np.full(n_vars, float(yint_array))
    elif yint_array.ndim == 1:
        if yint_array.shape[0] != n_vars:
            raise ValueError(
                f"1D yints must have length {n_vars} to match var_names; got {yint_array.shape[0]}."
            )
        yint_vector = yint_array
    else:
        raise ValueError("yints must be a scalar or a 1D numeric sequence.")

    residual_dist = str(residual_dist).lower()
    if residual_dist != "normal":
        raise ValueError(
            f"Unsupported residual_dist '{residual_dist}'. Supported residual distributions are: 'normal'."
        )

    residual_mean_array = np.asarray(residual_mean, dtype=float)
    if residual_mean_array.ndim == 0:
        residual_mean_vector = np.full(n_vars, float(residual_mean_array))
    elif residual_mean_array.ndim == 1:
        if residual_mean_array.shape[0] != n_vars:
            raise ValueError(
                f"1D residual_mean must have length {n_vars} to match var_names; "
                f"got {residual_mean_array.shape[0]}."
            )
        residual_mean_vector = residual_mean_array
    else:
        raise ValueError("residual_mean must be a scalar or a 1D numeric sequence.")

    residual_stdev_array = np.asarray(residual_stdev, dtype=float)
    if residual_stdev_array.ndim == 0:
        residual_stdev_vector = np.full(n_vars, float(residual_stdev_array))
    elif residual_stdev_array.ndim == 1:
        if residual_stdev_array.shape[0] != n_vars:
            raise ValueError(
                f"1D residual_stdev must have length {n_vars} to match var_names; "
                f"got {residual_stdev_array.shape[0]}."
            )
        residual_stdev_vector = residual_stdev_array
    else:
        raise ValueError("residual_stdev must be a scalar or a 1D numeric sequence.")
    if (residual_stdev_vector < 0).any():
        raise ValueError("residual_stdev must be >= 0 for all simulated features.")

    linear_mean = predictor_matrix @ beta_matrix.T
    linear_mean = linear_mean + yint_vector.reshape(1, n_vars)
    linear_mean = np.asarray(linear_mean, dtype=float).reshape(n_obs, n_vars)
    if np.all(residual_mean_vector == 0) and np.all(residual_stdev_vector == 0):
        residual_matrix = np.zeros((n_obs, n_vars), dtype=float)
    else:
        rng = np.random.default_rng(random_seed)
        residual_matrix = rng.normal(
            loc=residual_mean_vector.reshape(1, n_vars),
            scale=residual_stdev_vector.reshape(1, n_vars),
            size=(n_obs, n_vars),
        )
        residual_matrix = np.asarray(residual_matrix, dtype=float).reshape(n_obs, n_vars)

    X = linear_mean + residual_matrix

    var_df = pd.DataFrame(index=var_names)
    var_df["yint"] = yint_vector
    for idx, predictor_name in enumerate(predictor_names):
        var_df[f"beta_{predictor_name}"] = beta_matrix[:, idx]
    var_df["residual_dist"] = residual_dist
    var_df["residual_mean"] = residual_mean_vector
    var_df["residual_stdev"] = residual_stdev_vector

    adata: ad.AnnData | None = None
    if also_return_adata or save_adata_dataset:
        adata = ad.AnnData(X=X, obs=obs_df.copy(), var=var_df.copy())
        adata.layers["linear_mean"] = linear_mean.copy()
        adata.layers["residual"] = residual_matrix.copy()
        if save_adata_dataset:
            resolved_output_path = (
                Path.cwd() / "covar_dependent_dataset" if output_path is None else Path(output_path)
            )
            _save_dataset(adata, resolved_output_path, logger=LOGGER)

    return X, var_df, obs_df.copy(), adata


def sim_covar_dependent_dataset(
    obs_key_list: Sequence[str] | None = None,
    obs_covar_dist_params: dict[str, dict[str, float | str]] | None = None,
    n_obs: int = 100,
    obs_names_prefix: str = "obs_",
    save_obs_df: bool = False,
    save_obs_df_path: str | Path = "obs_df",
    random_seed: int | None = None,
    var_names: Sequence[str] | str = ("covar_dependent_feature",),
    betas: Sequence[float] | Sequence[Sequence[float]] = (0.05, 5.0),
    yints: float | Sequence[float] = 10,
    also_return_adata: bool = True,
    save_adata_dataset: bool = True,
    output_path: str | Path | None = None,
    residual_dist: str = "normal",
    residual_mean: float | Sequence[float] = 0.0,
    residual_stdev: float | Sequence[float] = 0.0,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, ad.AnnData | None]:
    """Simulate covariates first, then generate covariate-dependent features."""
    obs_df = sim_observations_covars(
        obs_key_list=obs_key_list,
        obs_covar_dist_params=obs_covar_dist_params,
        n_obs=n_obs,
        obs_names_prefix=obs_names_prefix,
        save_obs_df=save_obs_df,
        save_obs_df_path=save_obs_df_path,
        random_seed=random_seed,
    )
    feature_random_seed = None if random_seed is None else int(random_seed) + 1
    return sim_covar_dependent_features(
        obs_df=obs_df,
        var_names=var_names,
        betas=betas,
        yints=yints,
        also_return_adata=also_return_adata,
        save_adata_dataset=save_adata_dataset,
        output_path=output_path,
        residual_dist=residual_dist,
        residual_mean=residual_mean,
        residual_stdev=residual_stdev,
        # Use a derived seed so residual draws are deterministic without reusing
        # the exact covariate random stream from sim_observations_covars(...).
        random_seed=feature_random_seed,
    )
