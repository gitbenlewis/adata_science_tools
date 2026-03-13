"""Expectation-based covariate correction utilities."""

from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from patsy import dmatrix

from .._io._IO import save_dataset
from .._preprocessing._adata_row_operations import CFG_filter_adata_by_obs
from ._model_fit import fit_smf_ols_models_and_summarize_adata


__all__ = [
    "calculate_expectations",
    "reconstruct_expectation_model_spec",
    "convert_ols_summary_to_expectation_df",
    "save_expectation_model_files",
    "predict_expectation",
    "regress_out",
    "excess_expectation",
    "regression_expectation_correction_adata",
]


EXPECTATION_METADATA_COLUMNS = {
    "fit_formula",
    "fit_nobs",
    "fit_r2",
    "fit_ok",
    "fit_warning",
    "fit_method",
}

MODEL_SPEC_REQUIRED_KEYS = {
    "predictors",
    "formula_rhs",
    "design_terms",
    "coefficient_columns",
}


def _resolve_cfg_value(dataset_cfg: dict | None, key: str, explicit_value: Any) -> Any:
    if explicit_value is not None:
        return explicit_value
    if dataset_cfg is None:
        return None
    return dataset_cfg.get(key)


def _normalize_list_arg(value: Any, name: str, allow_none: bool = True) -> list[str] | None:
    if value is None:
        return None if allow_none else []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        raise TypeError(f"{name} must be a YAML list (e.g. ['Age', 'Gender']) not a single string.")
    raise TypeError(f"{name} must be list/tuple or None, got {type(value).__name__}.")


def _coerce_predictor_frame(obs_df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    predictor_df = obs_df.loc[:, predictors].copy()
    for predictor in predictors:
        series = predictor_df[predictor]
        if not series.notna().any():
            continue
        numeric_values = pd.to_numeric(series, errors="coerce")
        if numeric_values.notna().sum() == series.notna().sum():
            predictor_df[predictor] = numeric_values.astype(float)
    return predictor_df


def _filter_adata_for_expectation(
    adata: ad.AnnData,
    dataset_cfg: dict | None = None,
    filter_obs_boolean_column: str | None = None,
    filter_obs_column_key: str | None = None,
    filter_obs_column_values_list: list[Any] | None = None,
) -> ad.AnnData:
    if not any([dataset_cfg, filter_obs_boolean_column, filter_obs_column_key, filter_obs_column_values_list]):
        return adata
    return CFG_filter_adata_by_obs(
        adata,
        dataset_cfg=dataset_cfg,
        filter_obs_boolean_column=filter_obs_boolean_column,
        filter_obs_column_key=filter_obs_column_key,
        filter_obs_column_values_list=filter_obs_column_values_list,
        copy=True,
    )


def _make_formula_rhs(predictors: list[str]) -> str:
    if not predictors:
        raise ValueError("calculate_expectations requires at least one predictor.")
    return " + ".join(f'Q("{predictor}")' for predictor in predictors)


def _build_training_design_matrix(
    adata: ad.AnnData,
    predictors: list[str],
    formula_rhs: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictor_df = _coerce_predictor_frame(adata.obs, predictors)
    complete_cases = predictor_df.dropna(axis=0, how="any")
    if complete_cases.empty:
        raise ValueError("No rows remain after dropping missing predictor values.")
    complete_cases = complete_cases.copy()
    categorical_levels: dict[str, list[Any]] = {}
    for predictor in predictors:
        series = complete_cases[predictor]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = list(series.cat.categories)
        else:
            categories = list(pd.Index(series.dropna().unique()))
        categorical_levels[predictor] = categories
        complete_cases[predictor] = pd.Categorical(series, categories=categories)
    design_matrix = dmatrix(formula_rhs, complete_cases, return_type="dataframe")
    return design_matrix, complete_cases


def _extract_required_columns(summary_df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing expected columns while building {context}: {missing}")


def _summary_term_name(term: str) -> str:
    if term.startswith('Q("') and term.endswith('")'):
        return term[3:-2]
    return term


def _expectation_column_name(term: str) -> str:
    if term == "Intercept":
        return "intercept"
    clean_term = term
    if clean_term.startswith('Q("'):
        clean_term = clean_term.replace('Q("', "", 1).replace('")', "", 1)
    return clean_term


def _design_term_from_expectation_column(column: str, predictors: list[str]) -> str:
    if column == "intercept":
        return "Intercept"
    for predictor in predictors:
        if column == predictor:
            return f'Q("{predictor}")'
        if column.startswith(f"{predictor}["):
            return f'Q("{predictor}"){column[len(predictor):]}'
    raise KeyError(
        f"Unable to map expectation coefficient column '{column}' to one of predictors {predictors}."
    )


def _design_term_from_summary_term(summary_term: str, predictors: list[str]) -> str:
    if summary_term == "Intercept":
        return "Intercept"
    if summary_term in predictors:
        return f'Q("{summary_term}")'
    if summary_term.startswith('Q("'):
        return summary_term
    for predictor in predictors:
        if summary_term.startswith(f"{predictor}["):
            return f'Q("{predictor}"){summary_term[len(predictor):]}'
    raise KeyError(
        f"Unable to map summary coefficient term '{summary_term}' to one of predictors {predictors}."
    )


def _coefficient_columns_from_expectation_df(expectation_df: pd.DataFrame) -> list[str]:
    return [column for column in expectation_df.columns if column not in EXPECTATION_METADATA_COLUMNS]


def _default_model_spec_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".model_spec.yaml")


def _coerce_expectation_df_input(
    expectation_df: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(expectation_df, pd.DataFrame):
        return expectation_df, None
    if isinstance(expectation_df, (str, Path)):
        csv_path = Path(expectation_df)
        loaded_df = pd.read_csv(csv_path, index_col=0)
        return loaded_df, csv_path
    raise TypeError("expectation_df must be a pandas.DataFrame or CSV path.")


def _load_model_spec_yaml(model_spec_path: str | Path) -> dict[str, Any]:
    model_spec_path = Path(model_spec_path)
    try:
        with model_spec_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to parse model_spec YAML at {model_spec_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"model_spec YAML at {model_spec_path} must contain a mapping/dict.")
    return loaded


def _normalize_model_spec_dict(model_spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model_spec, dict):
        raise ValueError("model_spec must be a dict.")
    resolved = dict(model_spec)
    missing_keys = [key for key in MODEL_SPEC_REQUIRED_KEYS if key not in resolved]
    if missing_keys:
        raise KeyError(f"model_spec is missing required keys: {missing_keys}")
    fit_method = resolved.get("fit_method")
    if fit_method is not None and fit_method != "ols":
        raise ValueError("model_spec fit_method must be 'ols' when provided.")
    if resolved.get("categorical_levels") is None:
        resolved["categorical_levels"] = {}
    return resolved


def _resolve_model_spec_input(
    expectation_df: pd.DataFrame,
    model_spec: dict[str, Any] | str | Path | None,
    expectation_df_path: Path | None = None,
) -> dict[str, Any]:
    if model_spec is None:
        if "model_spec" in expectation_df.attrs:
            return _normalize_model_spec_dict(expectation_df.attrs["model_spec"])
        if expectation_df_path is not None:
            model_spec_path = _default_model_spec_path(expectation_df_path)
            if not model_spec_path.exists():
                raise FileNotFoundError(
                    f"model_spec was not provided and sibling model spec YAML was not found: {model_spec_path}"
                )
            return _normalize_model_spec_dict(_load_model_spec_yaml(model_spec_path))
        raise ValueError(
            "No model_spec provided and expectation_df.attrs['model_spec'] is missing. "
            "Pass a model_spec dict or YAML path."
        )
    if isinstance(model_spec, dict):
        return _normalize_model_spec_dict(model_spec)
    if isinstance(model_spec, (str, Path)):
        return _normalize_model_spec_dict(_load_model_spec_yaml(model_spec))
    raise TypeError("model_spec must be a dict or YAML path.")


def _to_yaml_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_yaml_safe_value(item) for item in value.tolist()]
    return value


def _reference_predictor_frame(
    predictors: list[str],
    reference_adata: ad.AnnData | None = None,
    reference_obs_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    if reference_adata is not None:
        return _coerce_predictor_frame(reference_adata.obs, predictors)
    if reference_obs_df is not None:
        return _coerce_predictor_frame(reference_obs_df, predictors)
    return None


def _infer_categorical_levels(
    predictors: list[str],
    coefficient_columns: list[str],
    categorical_levels: dict[str, list[Any]] | None = None,
    reference_adata: ad.AnnData | None = None,
    reference_obs_df: pd.DataFrame | None = None,
) -> dict[str, list[Any]]:
    resolved_levels = {
        predictor: list(levels)
        for predictor, levels in (categorical_levels or {}).items()
    }
    categorical_predictors = [
        predictor
        for predictor in predictors
        if any(column.startswith(f"{predictor}[") for column in coefficient_columns)
    ]
    if not categorical_predictors:
        return {}

    reference_df = _reference_predictor_frame(
        predictors,
        reference_adata=reference_adata,
        reference_obs_df=reference_obs_df,
    )
    for predictor in categorical_predictors:
        if predictor in resolved_levels:
            continue
        if reference_df is None:
            raise ValueError(
                f"categorical_levels or reference_adata/reference_obs_df is required to "
                f"reconstruct model_spec for categorical predictor '{predictor}'."
            )
        series = reference_df[predictor]
        if isinstance(series.dtype, pd.CategoricalDtype):
            resolved_levels[predictor] = list(series.cat.categories)
        else:
            resolved_levels[predictor] = list(pd.Index(series.dropna().unique()))
    return resolved_levels


def _build_expectation_table(
    summary_df: pd.DataFrame,
    model_name: str,
    design_terms: list[str],
    include_metadata: bool,
) -> pd.DataFrame:
    coef_prefix = f"{model_name}_Coef_"
    coefficient_columns = {
        term: _expectation_column_name(term)
        for term in design_terms
    }
    required_coef_columns = [f"{coef_prefix}{_summary_term_name(term)}" for term in design_terms]
    _extract_required_columns(summary_df, required_coef_columns, "expectation coefficients")

    expectation_df = pd.DataFrame(index=summary_df.index.copy())
    for term in design_terms:
        source_column = f"{coef_prefix}{_summary_term_name(term)}"
        target_column = coefficient_columns[term]
        expectation_df[target_column] = pd.to_numeric(summary_df[source_column], errors="coerce")

    if include_metadata:
        metadata_pairs = {
            "fit_formula": f"{model_name}_Formula",
            "fit_nobs": f"{model_name}_nobs",
            "fit_r2": f"{model_name}_R-squared",
            "fit_ok": f"{model_name}_Converged",
            "fit_warning": f"{model_name}_Warnings",
        }
        for target_column, source_column in metadata_pairs.items():
            if source_column in summary_df.columns:
                expectation_df[target_column] = summary_df[source_column]
        expectation_df["fit_method"] = "ols"
    return expectation_df


def _build_expectation_table_from_ols_summary(
    ols_summary_df: pd.DataFrame,
    predictors: list[str],
    model_name: str,
    include_metadata: bool,
) -> tuple[pd.DataFrame, list[str]]:
    coef_prefix = f"{model_name}_Coef_"
    design_terms: list[str] = []
    expectation_df = pd.DataFrame(index=ols_summary_df.index.copy())
    for column in ols_summary_df.columns:
        if not column.startswith(coef_prefix):
            continue
        summary_term = column[len(coef_prefix):]
        design_term = _design_term_from_summary_term(summary_term, predictors)
        design_terms.append(design_term)
        expectation_df[_expectation_column_name(design_term)] = pd.to_numeric(
            ols_summary_df[column],
            errors="coerce",
        )

    if not design_terms:
        raise KeyError(f"No coefficient columns found for model_name '{model_name}'.")

    if include_metadata:
        metadata_pairs = {
            "fit_formula": f"{model_name}_Formula",
            "fit_nobs": f"{model_name}_nobs",
            "fit_r2": f"{model_name}_R-squared",
            "fit_ok": f"{model_name}_Converged",
            "fit_warning": f"{model_name}_Warnings",
        }
        for target_column, source_column in metadata_pairs.items():
            if source_column in ols_summary_df.columns:
                expectation_df[target_column] = ols_summary_df[source_column]
        expectation_df["fit_method"] = "ols"
    return expectation_df, design_terms


def _build_model_spec(
    design_matrix: pd.DataFrame,
    training_predictor_df: pd.DataFrame,
    predictors: list[str],
    formula_rhs: str,
    model_name: str,
    layer: str | None,
    use_raw: bool,
) -> dict[str, Any]:
    return {
        "fit_method": "ols",
        "predictors": list(predictors),
        "formula_rhs": formula_rhs,
        "design_terms": design_matrix.columns.tolist(),
        "model_name": model_name,
        "layer": layer,
        "use_raw": use_raw,
        "categorical_levels": {
            predictor: list(training_predictor_df[predictor].cat.categories)
            for predictor in predictors
            if isinstance(training_predictor_df[predictor].dtype, pd.CategoricalDtype)
        },
        "coefficient_columns": {
            term: _expectation_column_name(term)
            for term in design_matrix.columns.tolist()
        },
    }


def reconstruct_expectation_model_spec(
    expectation_df: pd.DataFrame,
    predictors: list[str] | tuple[str, ...],
    *,
    model_name: str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    categorical_levels: dict[str, list[Any]] | None = None,
    reference_adata: ad.AnnData | None = None,
    reference_obs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Reconstruct ``model_spec`` for a loaded expectation coefficient table."""
    predictors = _normalize_list_arg(predictors, "predictors", allow_none=False)
    coefficient_columns = _coefficient_columns_from_expectation_df(expectation_df)
    if not coefficient_columns:
        raise ValueError("Expectation table does not contain any coefficient columns.")

    design_terms = [
        _design_term_from_expectation_column(column, predictors)
        for column in coefficient_columns
    ]
    resolved_categorical_levels = _infer_categorical_levels(
        predictors,
        coefficient_columns,
        categorical_levels=categorical_levels,
        reference_adata=reference_adata,
        reference_obs_df=reference_obs_df,
    )
    formula_rhs = _make_formula_rhs(predictors)
    return {
        "fit_method": "ols",
        "predictors": list(predictors),
        "formula_rhs": formula_rhs,
        "design_terms": design_terms,
        "model_name": model_name or expectation_df.attrs.get("model_name") or "expectation_fit",
        "layer": layer,
        "use_raw": use_raw,
        "categorical_levels": resolved_categorical_levels,
        "coefficient_columns": {
            design_term: coefficient_column
            for design_term, coefficient_column in zip(design_terms, coefficient_columns)
        },
    }


def convert_ols_summary_to_expectation_df(
    ols_summary_df: pd.DataFrame,
    predictors: list[str] | tuple[str, ...],
    *,
    model_name: str,
    layer: str | None = None,
    use_raw: bool = False,
    include_metadata: bool = True,
    categorical_levels: dict[str, list[Any]] | None = None,
    reference_adata: ad.AnnData | None = None,
    reference_obs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert an OLS summary table into an expectation coefficient table."""
    predictors = _normalize_list_arg(predictors, "predictors", allow_none=False)
    expectation_df, design_terms = _build_expectation_table_from_ols_summary(
        ols_summary_df,
        predictors=predictors,
        model_name=model_name,
        include_metadata=include_metadata,
    )
    model_spec = reconstruct_expectation_model_spec(
        expectation_df,
        predictors,
        model_name=model_name,
        layer=layer,
        use_raw=use_raw,
        categorical_levels=categorical_levels,
        reference_adata=reference_adata,
        reference_obs_df=reference_obs_df,
    )
    model_spec["design_terms"] = design_terms
    model_spec["coefficient_columns"] = {
        design_term: _expectation_column_name(design_term)
        for design_term in design_terms
    }
    expectation_df.attrs["model_spec"] = model_spec
    expectation_df.attrs["model_name"] = model_name
    return expectation_df


def _store_expectation_result(
    adata: ad.AnnData,
    expectation_df: pd.DataFrame,
    model_spec: dict[str, Any],
    model_name: str,
) -> None:
    if "expectation_model" not in adata.uns or not isinstance(adata.uns["expectation_model"], dict):
        adata.uns["expectation_model"] = {}
    adata.uns["expectation_model"][model_name] = {
        "table": expectation_df,
        "model_spec": model_spec,
        "fit_method": "ols",
        "predictors": list(model_spec["predictors"]),
        "design_terms": list(model_spec["design_terms"]),
        "formula_rhs": model_spec["formula_rhs"],
        "layer": model_spec["layer"],
        "use_raw": model_spec["use_raw"],
    }


def save_expectation_model_files(
    expectation_df: pd.DataFrame,
    csv_path: str | Path,
    *,
    model_spec: dict[str, Any] | str | Path | None = None,
    model_spec_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save an expectation coefficient table and model_spec YAML sidecar."""
    if not isinstance(expectation_df, pd.DataFrame):
        raise TypeError("expectation_df must be a pandas.DataFrame.")
    resolved_model_spec = _resolve_model_spec_input(expectation_df, model_spec)
    csv_path = Path(csv_path)
    model_spec_path = Path(model_spec_path) if model_spec_path is not None else _default_model_spec_path(csv_path)
    expectation_df.to_csv(csv_path, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
    with model_spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_yaml_safe_value(resolved_model_spec), handle, sort_keys=False)
    return csv_path, model_spec_path


def _prepare_prediction_frame(
    adata: ad.AnnData,
    predictors: list[str],
    baseline: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if baseline is None:
        missing_predictors = [predictor for predictor in predictors if predictor not in adata.obs.columns]
        if missing_predictors:
            raise KeyError(f"Predictor columns missing from adata.obs: {missing_predictors}")
        predictor_df = adata.obs.loc[:, predictors].copy()
    else:
        missing_predictors = [predictor for predictor in predictors if predictor not in baseline]
        if missing_predictors:
            raise KeyError(f"Baseline values missing for predictors: {missing_predictors}")
        predictor_df = pd.DataFrame(index=adata.obs_names)
        for predictor in predictors:
            predictor_df[predictor] = baseline[predictor]
    if predictor_df[predictors].isna().any(axis=None):
        raise ValueError("Prediction requires non-missing predictor values for all observations.")
    return _coerce_predictor_frame(predictor_df, predictors)


def _build_prediction_design(
    adata: ad.AnnData,
    model_spec: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> pd.DataFrame:
    predictors = list(model_spec["predictors"])
    predictor_df = _prepare_prediction_frame(adata, predictors, baseline=baseline)
    design_terms = list(model_spec["design_terms"])
    formula_rhs = model_spec.get("formula_rhs")
    categorical_levels = model_spec.get("categorical_levels", {})
    predictor_df = predictor_df.copy()
    for predictor, levels in categorical_levels.items():
        original_values = predictor_df[predictor].copy()
        original_non_null = original_values.notna()
        predictor_df[predictor] = pd.Categorical(original_values, categories=levels)
        unseen_mask = predictor_df[predictor].isna() & original_non_null
        if unseen_mask.any():
            bad_values = sorted({str(value) for value in pd.Index(original_values[unseen_mask]).tolist()})
            raise ValueError(
                f"Unable to build predictor design matrix for expectation prediction: "
                f"unseen category values for '{predictor}': {bad_values}"
            )
    try:
        if formula_rhs is None:
            raise ValueError("model_spec must include formula_rhs.")
        design_matrix = dmatrix(formula_rhs, predictor_df, return_type="dataframe", NA_action="raise")
    except Exception as exc:
        raise ValueError(f"Unable to build predictor design matrix for expectation prediction: {exc}") from exc

    unexpected_terms = [term for term in design_matrix.columns if term not in design_terms]
    missing_terms = [term for term in design_terms if term not in design_matrix.columns]
    if unexpected_terms or missing_terms:
        raise ValueError(
            f"Prediction design terms do not match fitted expectation model. "
            f"Missing: {missing_terms}; unexpected: {unexpected_terms}"
        )
    return design_matrix.loc[:, design_terms]


def _align_expectation_table(
    adata: ad.AnnData,
    expectation_df: pd.DataFrame,
    model_spec: dict[str, Any],
) -> pd.DataFrame:
    missing_features = [feature for feature in adata.var_names if feature not in expectation_df.index]
    if missing_features:
        raise KeyError(f"Expectation table is missing features required by adata.var_names: {missing_features[:10]}")
    aligned = expectation_df.reindex(adata.var_names)
    required_columns = [model_spec["coefficient_columns"][term] for term in model_spec["design_terms"]]
    missing_columns = [column for column in required_columns if column not in aligned.columns]
    if missing_columns:
        raise KeyError(f"Expectation table is missing coefficient columns: {missing_columns}")
    if aligned[required_columns].isna().any(axis=None):
        raise ValueError("Expectation coefficient table contains NaN values in required coefficient columns.")
    return aligned


def _matrix_from_adata(adata: ad.AnnData, layer: str | None = None) -> np.ndarray:
    matrix = adata.layers[layer] if layer is not None else adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def calculate_expectations(
    adata: ad.AnnData,
    covariates: list[str] | tuple[str, ...] | None = None,
    *,
    predictors: list[str] | tuple[str, ...] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    fit_method: str = "ols",
    feature_columns: list[str] | tuple[str, ...] | None = None,
    dataset_cfg: dict[str, Any] | None = None,
    filter_obs_boolean_column: str | None = None,
    filter_obs_column_key: str | None = None,
    filter_obs_column_values_list: list[Any] | tuple[Any, ...] | None = None,
    save_path: str | None = None,
    save_result_to_adata_uns_as_dict: bool | None = None,
    model_name: str | None = None,
    add_adata_var_column_key_list: list[str] | tuple[str, ...] | None = None,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """
    Fit per-feature OLS expectation models and return the coefficient table.

    The returned DataFrame stores the in-memory prediction spec in
    ``expectation_df.attrs["model_spec"]`` so the same object can be passed
    directly to :func:`predict_expectation`, :func:`regress_out`, and
    :func:`excess_expectation`.
    """
    if fit_method != "ols":
        raise ValueError("calculate_expectations currently supports fit_method='ols' only.")

    predictors = predictors if predictors is not None else covariates
    predictors = _resolve_cfg_value(dataset_cfg, "predictors", predictors)
    if predictors is None:
        predictors = _resolve_cfg_value(dataset_cfg, "covariates", covariates)
    predictors = _normalize_list_arg(predictors, "predictors", allow_none=False)
    if not predictors:
        raise ValueError("calculate_expectations requires at least one predictor.")

    feature_columns = _resolve_cfg_value(dataset_cfg, "feature_columns", feature_columns)
    feature_columns = _normalize_list_arg(feature_columns, "feature_columns") if feature_columns is not None else None
    filter_obs_column_values_list = _resolve_cfg_value(
        dataset_cfg,
        "filter_obs_column_values_list",
        filter_obs_column_values_list,
    )
    filter_obs_column_values_list = (
        _normalize_list_arg(filter_obs_column_values_list, "filter_obs_column_values_list")
        if filter_obs_column_values_list is not None
        else None
    )
    add_adata_var_column_key_list = _resolve_cfg_value(
        dataset_cfg,
        "add_adata_var_column_key_list",
        add_adata_var_column_key_list,
    )
    add_adata_var_column_key_list = (
        _normalize_list_arg(add_adata_var_column_key_list, "add_adata_var_column_key_list")
        if add_adata_var_column_key_list is not None
        else None
    )

    layer = _resolve_cfg_value(dataset_cfg, "layer", layer)
    save_path = _resolve_cfg_value(dataset_cfg, "save_path", save_path)
    model_name = _resolve_cfg_value(dataset_cfg, "model_name", model_name) or "expectation_fit"
    save_result_to_adata_uns_as_dict = _resolve_cfg_value(
        dataset_cfg,
        "save_result_to_adata_uns_as_dict",
        save_result_to_adata_uns_as_dict,
    )
    save_result_to_adata_uns_as_dict = bool(save_result_to_adata_uns_as_dict)
    filter_obs_boolean_column = _resolve_cfg_value(
        dataset_cfg,
        "filter_obs_boolean_column",
        filter_obs_boolean_column,
    )
    filter_obs_column_key = _resolve_cfg_value(dataset_cfg, "filter_obs_column_key", filter_obs_column_key)

    work_adata = _filter_adata_for_expectation(
        adata,
        dataset_cfg=dataset_cfg,
        filter_obs_boolean_column=filter_obs_boolean_column,
        filter_obs_column_key=filter_obs_column_key,
        filter_obs_column_values_list=filter_obs_column_values_list,
    )
    formula_rhs = _make_formula_rhs(predictors)
    design_matrix, training_predictor_df = _build_training_design_matrix(work_adata, predictors, formula_rhs)

    summary_df = fit_smf_ols_models_and_summarize_adata(
        work_adata,
        layer=layer,
        use_raw=use_raw,
        feature_columns=feature_columns,
        predictors=predictors,
        model_name=model_name,
        add_adata_var_column_key_list=add_adata_var_column_key_list,
        save_table=False,
        save_result_to_adata_uns_as_dict=False,
        include_fdr=False,
    )
    expectation_df = _build_expectation_table(
        summary_df,
        model_name=model_name,
        design_terms=design_matrix.columns.tolist(),
        include_metadata=include_metadata,
    )
    model_spec = _build_model_spec(
        design_matrix=design_matrix,
        training_predictor_df=training_predictor_df,
        predictors=predictors,
        formula_rhs=formula_rhs,
        model_name=model_name,
        layer=layer,
        use_raw=use_raw,
    )
    expectation_df.attrs["model_spec"] = model_spec
    expectation_df.attrs["model_name"] = model_name

    if save_path is not None:
        expectation_df.to_csv(save_path, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)

    if save_result_to_adata_uns_as_dict:
        _store_expectation_result(adata, expectation_df, model_spec, model_name)

    return expectation_df


def predict_expectation(
    adata: ad.AnnData,
    expectation_df: pd.DataFrame | str | Path,
    *,
    model_spec: dict[str, Any] | str | Path | None = None,
    include_intercept: bool = True,
    baseline: dict[str, Any] | None = None,
) -> np.ndarray:
    """Predict the expectation matrix from an in-memory table or CSV/YAML artifacts."""
    resolved_expectation_df, expectation_df_path = _coerce_expectation_df_input(expectation_df)
    resolved_model_spec = _resolve_model_spec_input(
        resolved_expectation_df,
        model_spec,
        expectation_df_path=expectation_df_path,
    )
    aligned_expectation_df = _align_expectation_table(adata, resolved_expectation_df, resolved_model_spec)
    design_matrix = _build_prediction_design(
        adata,
        model_spec=resolved_model_spec,
        baseline=baseline,
    )

    terms_to_use = list(resolved_model_spec["design_terms"])
    if not include_intercept and "Intercept" in terms_to_use:
        terms_to_use = [term for term in terms_to_use if term != "Intercept"]
    if not terms_to_use:
        return np.zeros((adata.n_obs, adata.n_vars), dtype=float)

    coefficient_columns = [resolved_model_spec["coefficient_columns"][term] for term in terms_to_use]
    coefficient_matrix = np.vstack([
        aligned_expectation_df[column].to_numpy(dtype=float)
        for column in coefficient_columns
    ])
    return np.asarray(design_matrix.loc[:, terms_to_use], dtype=float) @ coefficient_matrix


def regress_out(
    adata: ad.AnnData,
    expectation_df: pd.DataFrame | str | Path,
    *,
    model_spec: dict[str, Any] | str | Path | None = None,
    baseline: dict[str, Any] | None = None,
    flavor: str = "obs_minus_exp_covar",
    input_layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = False,
) -> ad.AnnData:
    """Apply covariate-only expectation correction from in-memory or file-backed inputs."""
    if flavor not in {"obs_minus_exp_covar", "obs_minus_exp_covar_baseline"}:
        raise ValueError(
            "regress_out supports flavors 'obs_minus_exp_covar' and "
            "'obs_minus_exp_covar_baseline' only."
        )
    target_adata = adata if inplace else adata.copy()
    output_layer = output_layer or flavor
    observed = _matrix_from_adata(target_adata, layer=input_layer)
    covariate_component = predict_expectation(
        target_adata,
        expectation_df,
        model_spec=model_spec,
        include_intercept=False,
    )

    if flavor == "obs_minus_exp_covar":
        corrected = observed - covariate_component
    else:
        if baseline is None:
            raise ValueError("obs_minus_exp_covar_baseline requires a baseline dict.")
        baseline_component = predict_expectation(
            target_adata,
            expectation_df,
            model_spec=model_spec,
            include_intercept=False,
            baseline=baseline,
        )
        corrected = observed - (covariate_component - baseline_component)

    target_adata.layers[output_layer] = corrected
    return target_adata


def excess_expectation(
    adata: ad.AnnData,
    expectation_df: pd.DataFrame | str | Path,
    *,
    model_spec: dict[str, Any] | str | Path | None = None,
    flavor: str = "obs_minus_exp_val",
    input_layer: str | None = None,
    output_layer: str | None = None,
    inplace: bool = False,
    eps: float | None = None,
) -> ad.AnnData:
    """Apply residual or ratio-based expectation transforms from in-memory or file-backed inputs."""
    valid_flavors = {
        "obs_minus_exp_val",
        "obs_over_exp",
        "log_obs_over_exp",
        "log2_obs_over_exp",
    }
    if flavor not in valid_flavors:
        raise ValueError(f"Unsupported excess_expectation flavor '{flavor}'.")
    if eps is not None and eps <= 0:
        raise ValueError("eps must be positive when provided.")

    target_adata = adata if inplace else adata.copy()
    output_layer = output_layer or flavor
    observed = _matrix_from_adata(target_adata, layer=input_layer)
    expected = predict_expectation(
        target_adata,
        expectation_df,
        model_spec=model_spec,
        include_intercept=True,
    )

    if flavor == "obs_minus_exp_val":
        result = observed - expected
    else:
        if eps is None and np.any(expected <= 0):
            raise ValueError("Expectation values must be strictly positive for ratio-based transforms.")
        numerator = observed
        denominator = expected
        if eps is not None:
            numerator = numerator + eps
            denominator = denominator + eps

        ratio = numerator / denominator
        if flavor == "obs_over_exp":
            result = ratio
        else:
            if eps is None and np.any(observed <= 0):
                raise ValueError("Observed values must be strictly positive for log expectation transforms.")
            if flavor == "log_obs_over_exp":
                result = np.log(ratio)
            else:
                result = np.log2(ratio)

    target_adata.layers[output_layer] = result
    return target_adata

def regression_expectation_correction_adata(
    adata: ad.AnnData,
    *,
    calculate_expectations_params: dict[str, Any] | None = None,
    regress_out_params: dict[str, Any] | None = None,
    predict_expectation_params: dict[str, Any] | None = None,
    excess_expectation_params: dict[str, Any] | None = None,
    expectation_save_path: str | Path | None = None,
    output_h5ad_path: str | Path | None = None,
    dataset_cfg: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Run expectation fitting plus covariate regression correction on one AnnData."""
    cfg_alias = kwargs.pop("CFG", None)
    cfg_like: dict[str, Any] = {}
    if isinstance(cfg_alias, dict):
        cfg_like.update(cfg_alias)
    if isinstance(dataset_cfg, dict):
        cfg_like.update(dataset_cfg)
    if kwargs:
        cfg_like.update(kwargs)

    log = logger or cfg_like.get("logger") or logging.getLogger(__name__)

    calc_params = copy.deepcopy(cfg_like.get("calculate_expectations_params") or {})
    regress_params = copy.deepcopy(cfg_like.get("regress_out_params") or {})
    predict_params = copy.deepcopy(cfg_like.get("predict_expectation_params") or {})
    excess_params = copy.deepcopy(cfg_like.get("excess_expectation_params") or {})

    if calculate_expectations_params is not None:
        calc_params = copy.deepcopy(calculate_expectations_params)
    if regress_out_params is not None:
        regress_params = copy.deepcopy(regress_out_params)
    if predict_expectation_params is not None:
        predict_params = copy.deepcopy(predict_expectation_params)
    if excess_expectation_params is not None:
        excess_params = copy.deepcopy(excess_expectation_params)

    resolved_expectation_save_path = expectation_save_path or cfg_like.get("expectation_save_path")
    if resolved_expectation_save_path is None and calc_params.get("save_path") is not None:
        resolved_expectation_save_path = calc_params.get("save_path")
    if "save_path" in calc_params:
        # Save once through the wrapper so the CSV and model_spec sidecar stay in sync.
        calc_params.pop("save_path")

    resolved_output_h5ad_path = output_h5ad_path or cfg_like.get("output_h5ad_path")
    if resolved_output_h5ad_path is None:
        run_out_dir = cfg_like.get("run_out_dir")
        filename = cfg_like.get("filename")
        if run_out_dir is not None and filename is not None:
            resolved_output_h5ad_path = Path(run_out_dir) / str(filename)

    if not regress_params:
        raise ValueError("regress_out_params is required.")

    expectation_df = regress_params.get("expectation_df")
    model_spec = regress_params.get("model_spec")
    if expectation_df is None and predict_params.get("expectation_df") is not None:
        expectation_df = predict_params.get("expectation_df")
    if model_spec is None and predict_params.get("model_spec") is not None:
        model_spec = predict_params.get("model_spec")

    if expectation_df is None:
        if not calc_params:
            raise ValueError(
                "calculate_expectations_params is required when regress_out_params does not provide expectation_df."
            )
        log.info("Fitting expectation model with calculate_expectations_params: %s", calc_params)
        expectation_df = calculate_expectations(adata, **calc_params)
        if resolved_expectation_save_path is not None:
            resolved_expectation_save_path = Path(resolved_expectation_save_path)
            resolved_expectation_save_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path, model_spec_path = save_expectation_model_files(
                expectation_df,
                resolved_expectation_save_path,
                model_spec=model_spec,
            )
            log.info("Saved expectation artifacts to %s and %s", csv_path, model_spec_path)

    regress_params.pop("expectation_df", None)
    regress_params.pop("model_spec", None)
    if regress_params.get("baseline") is None and predict_params.get("baseline") is not None:
        regress_params["baseline"] = copy.deepcopy(predict_params["baseline"])

    active_excess_params = {key: value for key, value in excess_params.items() if value is not None}
    if active_excess_params:
        raise NotImplementedError(
            "excess_expectation_params is accepted by the wrapper but not executed in this first pass."
        )

    log.info("Applying regress_out with regress_out_params: %s", regress_params)
    corrected_adata = regress_out(
        adata,
        expectation_df,
        model_spec=model_spec,
        inplace=False,
        **regress_params,
    )

    if resolved_output_h5ad_path is not None:
        resolved_output_h5ad_path = Path(resolved_output_h5ad_path)
        save_dataset(corrected_adata, resolved_output_h5ad_path, logger=log)

    return corrected_adata
