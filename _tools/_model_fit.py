# module imports
# module at : _tools/_model_fit.py
## updated 2026-02-24 added guard to skip features with no complete cases after dropping NaN/inf, and to record the reason for skipping in the model summary dataframe
## updated 2026-02-24 added guard to coerce numeric-like predictors to numeric dtype so they are treated as continuous in the formula instead of categorical dummies, and added this coercion step to both OLS and mixedlm functions
# updated 2026-03-13 add expectation model fit and correction
# updated 2026-03-19 added new _save_model_fit_results_csv(...) helper and switches all OLS and MixedLM CSV writes to index=False
from .. _io._IO import make_df_obs_adataX

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from contextlib import contextmanager as _contextmanager
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import warnings
import yaml
import threading as _threading


# Python 3.10 warning hooks are process-wide, so separate model-fit calls must not
# install overlapping capture contexts.
_MODEL_FIT_WARNING_LOCK = _threading.Lock()


def _validate_threads(threads):
    if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)):
        raise TypeError("threads must be a positive integer.")
    if threads < 1:
        raise ValueError("threads must be a positive integer.")
    return int(threads)


@_contextmanager
def _capture_model_fit_warnings(warning_state):
    if warning_state is None:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            yield caught_warnings
        return

    caught_warnings = []
    warning_state.caught_warnings = caught_warnings
    try:
        yield caught_warnings
    finally:
        del warning_state.caught_warnings


def _run_feature_fits(feature_columns, fit_feature, threads):
    """Run independent feature fits in input order with per-feature warnings."""
    threads = _validate_threads(threads)
    with _MODEL_FIT_WARNING_LOCK:
        if threads == 1:
            return [fit_feature(feature, None) for feature in feature_columns]

        warning_state = _threading.local()
        with warnings.catch_warnings():
            previous_showwarning = warnings.showwarning

            def route_warning(message, category, filename, lineno, file=None, line=None):
                caught_warnings = getattr(warning_state, "caught_warnings", None)
                if caught_warnings is None:
                    previous_showwarning(message, category, filename, lineno, file, line)
                    return
                caught_warnings.append(
                    warnings.WarningMessage(
                        message,
                        category,
                        filename,
                        lineno,
                        file=file,
                        line=line,
                    )
                )

            warnings.showwarning = route_warning
            warnings.simplefilter("always")

            def fit_threaded(feature):
                return fit_feature(feature, warning_state)

            with _ThreadPoolExecutor(max_workers=threads) as executor:
                return list(executor.map(fit_threaded, feature_columns))

# Helper inside the module (near top)
def _ensure_list(x, name):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        raise TypeError(f"{name} must be a YAML list (e.g. ['Age','Gender']) not a single string.")
    raise TypeError(f"{name} must be list/tuple or None, got {type(x).__name__}")


def _make_model_formula_rhs(predictors):
    if not predictors:
        return ""
    return " + ".join(f'Q("{predictor}")' for predictor in predictors)


def _model_spec_yaml_path(save_path):
    return Path(save_path).with_suffix(".model_spec.yaml")


def _coefficient_columns_from_results(results, model_name):
    coef_prefix = f"{model_name}_Coef_"
    return [column for column in results.columns if column.startswith(coef_prefix)]


def _build_model_fit_model_spec(
        results,
        *,
        fit_method,
        model_name,
        predictors,
        layer,
        use_raw,
        group=None,
        reml=None,
    ):
    coefficient_columns = _coefficient_columns_from_results(results, model_name)
    coef_prefix = f"{model_name}_Coef_"
    model_spec = {
        "fit_method": fit_method,
        "model_name": model_name,
        "predictors": list(predictors),
        "layer": layer,
        "use_raw": use_raw,
        "formula_rhs": _make_model_formula_rhs(predictors),
        "coefficient_terms": [column[len(coef_prefix):] for column in coefficient_columns],
        "coefficient_columns": coefficient_columns,
    }
    if group is not None:
        model_spec["group"] = group
    if reml is not None:
        model_spec["reml"] = bool(reml)
    return model_spec


def _save_model_spec_yaml(model_spec, save_path):
    model_spec_path = _model_spec_yaml_path(save_path)
    with model_spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(model_spec, handle, sort_keys=False)
    print(f"Saved model_spec YAML to {model_spec_path}")


def _save_model_fit_results_csv(results, save_path):
    import csv

    if "var_names" not in results.columns:
        raise ValueError("Model-fit results must include a 'var_names' column before saving.")

    results.to_csv(
        save_path,
        index=False,
        float_format="%.6f",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"Saved model-fit results to {save_path}")



def fit_smf_ols_models_and_summarize_wide(
        obs_X_df,
        feature_columns=None, 
        predictors=None,
        model_name='OLS',
        include_fdr=True,
        threads: int = 1,
    ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    threads = _validate_threads(threads)
    # pandas does not guarantee thread-safe copying from a shared DataFrame.
    # Keep only selection and the deep copy serial; preprocessing and fitting stay parallel.
    dataframe_copy_lock = _threading.Lock()
    # Threaded fits with the same complete-case rows can reuse one formula-built
    # predictor matrix. Each model still receives its own copy for thread safety.
    design_matrix_lock = _threading.Lock() if threads > 1 else None
    design_matrix_cache = {
        "values": None,
        "columns": None,
        "disabled": False,
    }
    prepared_predictor_frame = None
    prepared_predictor_complete_mask = None
    # Keep unusual labels, dtypes, and DataFrame subclasses on the established
    # per-feature preparation path.
    can_prepare_predictors = (
        threads > 1
        and type(obs_X_df) is pd.DataFrame
        and isinstance(feature_columns, list)
        and bool(feature_columns)
        and isinstance(predictors, list)
        and bool(predictors)
        and obs_X_df.columns.is_unique
        and all(
            isinstance(column, str)
            and column.isprintable()
            and '"' not in column
            and "\\" not in column
            for column in feature_columns + predictors
        )
        and len(feature_columns) == len(set(feature_columns))
        and len(predictors) == len(set(predictors))
        and set(feature_columns).isdisjoint(predictors)
        and all(
            column in obs_X_df.columns
            for column in feature_columns + predictors
        )
        and all(
            isinstance(obs_X_df.dtypes.loc[feature], np.dtype)
            and obs_X_df.dtypes.loc[feature].kind in {"f", "i", "u"}
            for feature in feature_columns
        )
    )
    if can_prepare_predictors:
        candidate_predictor_frame = None
        # A warning or failure must retain the original per-feature path so its
        # warning count and order remain unchanged.
        with _MODEL_FIT_WARNING_LOCK:
            with warnings.catch_warnings(record=True) as preparation_warnings:
                warnings.simplefilter("always")
                try:
                    candidate_predictor_frame = obs_X_df[predictors].copy(deep=True)
                    candidate_predictor_frame.replace(np.inf, np.nan, inplace=True)
                    candidate_predictor_frame.replace(-np.inf, np.nan, inplace=True)
                    for predictor in predictors:
                        series = candidate_predictor_frame[predictor]
                        if not series.notna().any():
                            continue
                        numeric_values = pd.to_numeric(series, errors="coerce")
                        if numeric_values.notna().sum() == series.notna().sum():
                            candidate_predictor_frame[predictor] = numeric_values.astype(float)
                    candidate_predictor_complete_mask = (
                        candidate_predictor_frame.notna().all(axis=1)
                    )
                except Exception:
                    candidate_predictor_frame = None
        if candidate_predictor_frame is not None and not preparation_warnings:
            prepared_predictor_frame = candidate_predictor_frame
            prepared_predictor_complete_mask = candidate_predictor_complete_mask

    def fit_feature(feature, warning_state):
        columns2keep = [feature] + predictors
        if prepared_predictor_frame is None:
            with dataframe_copy_lock:
                df = obs_X_df[columns2keep].copy(deep=True)
            # Scalar replacements avoid pandas' list-replacement path, which can raise
            # IndexError while updating Copy-on-Write block references.
            df.replace(np.inf, np.nan, inplace=True)
            df.replace(-np.inf, np.nan, inplace=True)
            # Coerce numeric-like predictors (e.g. Age loaded as strings/categories) to numeric
            # so formula terms remain continuous instead of categorical dummies.
            for predictor in predictors:
                series = df[predictor]
                if not series.notna().any():
                    continue
                numeric_values = pd.to_numeric(series, errors="coerce")
                if numeric_values.notna().sum() == series.notna().sum():
                    df[predictor] = numeric_values.astype(float)
            complete_case_mask = df.notna().all(axis=1)
        else:
            # Deep-copy both shared inputs under the existing pandas copy lock so
            # every worker continues with fully private DataFrame blocks.
            with dataframe_copy_lock:
                response_frame = obs_X_df[[feature]].copy(deep=True)
                predictor_frame = prepared_predictor_frame.copy(deep=True)
                predictor_complete_mask = prepared_predictor_complete_mask.copy(
                    deep=True
                )
            response_frame.replace(np.inf, np.nan, inplace=True)
            response_frame.replace(-np.inf, np.nan, inplace=True)
            complete_case_mask = (
                response_frame.notna().all(axis=1)
                & predictor_complete_mask
            )
            df = predictor_frame
            df.insert(0, feature, response_frame.iloc[:, 0])
            del predictor_frame, response_frame
        predictors_q = [f'Q("{p}")' for p in predictors]
        model_string = f'Q("{feature}") ~ {" + ".join(predictors_q)}'
        model_summary_formula = f'{feature} ~ {" + ".join(predictors)}'
        if not complete_case_mask.any():
            skipped_reason = (
                f"No complete-case rows after dropping NaN/inf for columns {columns2keep}."
            )
            return feature, (None, [], model_summary_formula), skipped_reason
        cache_eligible = (
            design_matrix_lock is not None
            and bool(predictors)
            and isinstance(feature, str)
            and feature.isprintable()
            and '"' not in feature
            and "\\" not in feature
        )
        direct_numeric_response = False
        if cache_eligible:
            response = df[feature]
            if (
                isinstance(response, pd.Series)
                and getattr(response.dtype, "kind", None) in {"f", "i", "u"}
            ):
                response_missing = response.isna()
                if response_missing.any():
                    missing_rows_have_complete_predictors = (
                        df.loc[response_missing, predictors]
                        .notna()
                        .all(axis=1)
                        .any()
                    )
                    cache_eligible = not missing_rows_have_complete_predictors
                direct_numeric_response = (
                    cache_eligible
                    and type(response) is pd.Series
                    and isinstance(response.dtype, np.dtype)
                )
            else:
                cache_eligible = False
        df = df.loc[complete_case_mask]
        skipped_reason = None
        with _capture_model_fit_warnings(warning_state) as caught_warnings:
            try:
                if not cache_eligible:
                    model = smf.ols(model_string, df).fit()
                else:
                    cached_design = None
                    fit_with_formula = False
                    with design_matrix_lock:
                        if design_matrix_cache["disabled"]:
                            fit_with_formula = True
                        elif design_matrix_cache["values"] is None:
                            warning_count = len(caught_warnings)
                            design_is_reusable = False
                            try:
                                design_df = patsy.dmatrix(
                                    _make_model_formula_rhs(predictors),
                                    df,
                                    return_type="dataframe",
                                )
                                design_is_reusable = (
                                    len(caught_warnings) == warning_count
                                    and design_df.index.equals(df.index)
                                )
                                if design_is_reusable:
                                    design_values = design_df.to_numpy(copy=False)
                                    design_values.setflags(write=False)
                            except Exception:
                                design_is_reusable = False
                            if not design_is_reusable:
                                # Let each full formula recreate any warning or
                                # row-selection behavior from the shared RHS.
                                del caught_warnings[warning_count:]
                                design_matrix_cache["disabled"] = True
                                fit_with_formula = True
                            else:
                                design_matrix_cache["values"] = design_values
                                design_matrix_cache["columns"] = tuple(
                                    design_df.columns
                                )
                                cached_design = (
                                    design_values,
                                    design_matrix_cache["columns"],
                                )
                        else:
                            cached_design = (
                                design_matrix_cache["values"],
                                design_matrix_cache["columns"],
                            )

                    if fit_with_formula:
                        model = smf.ols(model_string, df).fit()
                    elif cached_design is not None:
                        design_values, design_columns = cached_design
                        warning_count = len(caught_warnings)
                        try:
                            if direct_numeric_response:
                                # Patsy only casts ordinary numeric responses to
                                # float64 here, so avoid rebuilding that design.
                                response = df[feature].astype(float, copy=False)
                            else:
                                response_df = patsy.dmatrix(
                                    f'Q("{feature}") - 1',
                                    df,
                                    return_type="dataframe",
                                )
                                if (
                                    len(response_df.columns) != 1
                                    or not response_df.index.equals(df.index)
                                ):
                                    raise ValueError(
                                        "Cached OLS response design did not preserve rows."
                                    )
                                response = response_df.iloc[:, 0]
                            response.name = feature
                            feature_design_df = pd.DataFrame(
                                design_values.copy(),
                                index=df.index,
                                columns=design_columns,
                            )
                            model = sm.OLS(response, feature_design_df).fit()
                        except Exception:
                            # Preserve the established formula behavior for any
                            # response that the direct path cannot fit identically.
                            del caught_warnings[warning_count:]
                            model = smf.ols(model_string, df).fit()
            except Exception as e:
                model = None
                skipped_reason = f"{type(e).__name__}: {e}"
        return feature, (model, caught_warnings, model_summary_formula), skipped_reason

    # Store models and any fit warnings in dictionaries keyed by feature.
    models = {}
    skipped_reasons = {}
    for feature, model_data, skipped_reason in _run_feature_fits(
        feature_columns,
        fit_feature,
        threads,
    ):
        models[feature] = model_data
        if skipped_reason is not None:
            skipped_reasons[feature] = skipped_reason

    # make a results dataframe from the dict of models
    summary_rows = []
    for feature_name in feature_columns:
        model, caught_warnings, model_string = models[feature_name]
        if model is None:
            summary_rows.append(
                {
                    f'{model_name}_Formula': model_string,
                    f'{model_name}_Converged': False,
                    f'{model_name}_Warnings': skipped_reasons.get(
                        feature_name, "Model fit skipped."
                    ),
                }
            )
            continue
        converged = getattr(model, "converged", None)
        if converged is None and hasattr(model, "mle_retvals"):
            converged = model.mle_retvals.get("converged")
        if converged is None:
            converged = True  # OLS solves in closed form, so treat as converged by default
        warning_messages = "; ".join(f"{w.category.__name__}: {w.message}" for w in caught_warnings)
        confidence_intervals = model.conf_int().to_numpy(copy=False)
        residuals = model.resid
        jb, jb_p, skew, kurtosis = sm.stats.stattools.jarque_bera(residuals)
        omni, omni_p = sm.stats.stattools.omni_normtest(residuals)
        dw = sm.stats.durbin_watson(residuals)
        summary_data = {
            f'{model_name}_Log-Likelihood': model.llf,
            f'{model_name}_AIC': model.aic,
            f'{model_name}_BIC': model.bic,
            f'{model_name}_Formula': model_string,
            f'{model_name}_nobs': model.nobs,
            f'{model_name}_df_model': model.df_model,
            f'{model_name}_df_resid': model.df_resid,
            f'{model_name}_Scale': model.scale,
            f'{model_name}_Cov_Type': getattr(model, "cov_type", np.nan),
            f'{model_name}_Durbin_Watson': dw,
            f'{model_name}_Omnibus': omni,
            f'{model_name}_Omnibus_p': omni_p,
            f'{model_name}_Jarque_Bera': jb,
            f'{model_name}_Jarque_Bera_p': jb_p,
            f'{model_name}_Skew': skew,
            f'{model_name}_Kurtosis': kurtosis,
            f'{model_name}_Condition_Number': getattr(model, "condition_number", np.nan),
            f'{model_name}_R-squared': model.rsquared,
            f'{model_name}_Adj. R-squared': model.rsquared_adj,
        }
        f_pvalue = model.f_pvalue
        summary_data.update({
            f'{model_name}_F-statistic': model.fvalue if f_pvalue is not None else np.nan,
            f'{model_name}_P(F-statistic)': f_pvalue if f_pvalue is not None else np.nan,
            f'{model_name}_Converged': converged,
            f'{model_name}_Warnings': warning_messages if warning_messages else np.nan,
        })
        # Each access to these Statsmodels properties builds a pandas wrapper, so
        # materialize every result vector once before iterating over coefficients.
        parameters = model.params
        parameter_names = parameters.index
        parameter_values = parameters.to_numpy(copy=False)
        standard_errors = model.bse.to_numpy(copy=False)
        t_values = model.tvalues.to_numpy(copy=False)
        p_values = model.pvalues.to_numpy(copy=False)
        for parameter_index, param_name in enumerate(parameter_names):
            clean_param = param_name
            if clean_param.startswith('Q("') and clean_param.endswith('")'):
                clean_param = clean_param[3:-2]
            summary_data[f'{model_name}_Coef_{clean_param}'] = parameter_values[parameter_index]
            summary_data[f'{model_name}_StdErr_{clean_param}'] = standard_errors[parameter_index]
            summary_data[f'{model_name}_tStat_{clean_param}'] = t_values[parameter_index]
            summary_data[f'{model_name}_P>|t|_{clean_param}'] = p_values[parameter_index]
            ci_low, ci_high = confidence_intervals[parameter_index]
            summary_data[f'{model_name}_CI_low_{clean_param}'] = ci_low
            summary_data[f'{model_name}_CI_high_{clean_param}'] = ci_high
        summary_rows.append(summary_data)
    # make the final results dataframe
    results = pd.DataFrame(summary_rows, index=feature_columns)
    if include_fdr:
        pval_cols = [c for c in results.columns if c.startswith(f'{model_name}_P>|t|_')]
        for col in pval_cols:
            mask = results[col].notna()
            if not mask.any():
                continue
            _, qvals, _, _ = multipletests(results.loc[mask, col], method='fdr_bh')
            fdr_col = f'{col}_FDR'
            results[fdr_col] = np.nan
            results.loc[mask, fdr_col] = qvals
    var_names = feature_columns
    results['var_names'] = var_names 
    # place 'var_names' as the first column
    cols = results.columns.tolist()
    results = results[['var_names'] + [col for col in cols if col != 'var_names']]
    return results


def old_fit_smf_ols_models_and_summarize_adata(
        adata,layer=None,use_raw=False,
        feature_columns=None,
        predictors=None, 
        model_name='OLS_predictors',
        add_adata_var_column_key_list=None,
        save_table=False,
        save_path=None,
        save_result_to_adata_uns_as_dict=False,
        include_fdr=True,
            ):
    obs_X_df=make_df_obs_adataX(adata,layer=layer,use_raw=use_raw,include_obs=True,)
    feature_columns=feature_columns if feature_columns is not None else adata.var_names.tolist()
    results=fit_smf_ols_models_and_summarize_wide(obs_X_df, feature_columns, predictors, model_name=model_name, include_fdr=include_fdr)
    # convert numeric columns to numeric dtype
    num_cols = [
                col for col in results.columns
                if pd.to_numeric(results[col], errors="coerce").notna().all()
            ]
    if 'var_names' in num_cols:     # remove 'var_names' from num_cols
        num_cols.remove('var_names')
    results[num_cols] = results[num_cols].apply(pd.to_numeric)

    # add adata.var columns to the results dataframe if specified
    if add_adata_var_column_key_list is not None and adata is not None:
        # add adata.var columns to the results dataframe
        for var_col_key in add_adata_var_column_key_list:
            if var_col_key in adata.var.columns:
                var_col_values = adata.var[var_col_key]
                results = results.merge(var_col_values, left_index=True, right_index=True, how='left', suffixes=('', f'_{var_col_key}'))
            else:
                print(f"Warning: '{var_col_key}' not found in adata.var columns. Skipping this column.")

    # add results to adata.uns if specified
    if save_result_to_adata_uns_as_dict and adata is not None:
        key=f'OLS_model_results_{model_name}'
        if 'ols_model_results' not in adata.uns:
            adata.uns['ols_model_results'] = {}
        adata.uns['ols_model_results'][key] = results
        print(f"Added fit_smf_ols_models_and_summarize_wide  results to adata.uns['ols_model_results']['{key}']")

    # save the results dataframe to the save_path
    if save_table and save_path is not None:
        _save_model_fit_results_csv(results, save_path)
    return results

def fit_smf_ols_models_and_summarize_adata(
        adata,
        layer=None,
        use_raw=False,
        feature_columns=None,
        predictors=None,
        model_name='OLS_predictors',
        add_adata_var_column_key_list=None,
        save_table=False,
        save_model_spec_yaml: bool = False,
        save_path=None,
        save_result_to_adata_uns_as_dict=False,
        include_fdr=True,
        # --- new filter args ---
        dataset_cfg=None,
        filter_obs_boolean_column=None,
        filter_obs_column_key=None,
        filter_obs_column_values_list=None,
        filter_obs_copy=True,
        # when filtered internally, optionally also write results into the original adata.uns
        save_results_to_original_adata_uns: bool = False,
        # whether to return the filtered adata (work_adata) in addition to results
        return_filtered_adata: bool = False,
        threads: int = 1,
    ):
    """
    Fit OLS models for features in an AnnData and return a summary DataFrame.

    New behaviour:
      - If any of dataset_cfg, filter_obs_boolean_column, filter_obs_column_key,
        or filter_obs_column_values_list are provided, a filtered AnnData
        (work_adata) is created via CFG_filter_adata_by_obs and used for the fit.
      - If save_result_to_adata_uns_as_dict is True results are saved to
        work_adata.uns['ols_model_results'][f'OLS_model_results_{model_name}'].
      - If save_results_to_original_adata_uns is True and work_adata is a filtered
        copy, the same results are also saved into the original adata.uns.
      - return_filtered_adata=True will return (results, work_adata) instead of results.
      - threads>1 fits independent features concurrently while preserving result order.

    Backwards-compatible defaults preserve previous behaviour when no filter args are given.
    """
    # Local imports to avoid changing top-of-file imports and to keep the patch minimal.
    from .._preprocessing._adata_row_operations import CFG_filter_adata_by_obs
    from .._io._IO import make_df_obs_adataX
    import pandas as pd
    import numpy as np


    # If filter args provided, create a filtered work_adata
    if any([dataset_cfg, filter_obs_boolean_column, filter_obs_column_key, filter_obs_column_values_list]):
        work_adata = CFG_filter_adata_by_obs(
            adata,
            dataset_cfg=dataset_cfg,
            filter_obs_boolean_column=filter_obs_boolean_column,
            filter_obs_column_key=filter_obs_column_key,
            filter_obs_column_values_list=filter_obs_column_values_list,
            copy=filter_obs_copy,
        )
    else:
        work_adata = adata

    # Validate/normalize list-like inputs coming from YAML
    predictors = _ensure_list(predictors, "predictors")
    add_adata_var_column_key_list = _ensure_list(add_adata_var_column_key_list, "add_adata_var_column_key_list")
    if save_model_spec_yaml and (not save_table or save_path is None):
        raise ValueError("save_model_spec_yaml=True requires save_table=True and save_path to be set.")

    # Build the obs_X_df using the (possibly filtered) work_adata
    obs_X_df = make_df_obs_adataX(work_adata, layer=layer, use_raw=use_raw, include_obs=True,)
    feature_columns = feature_columns if feature_columns is not None else work_adata.var_names.tolist()

    # Delegate the heavy lifting to the wide version; threads=1 preserves serial fitting.
    results = fit_smf_ols_models_and_summarize_wide(
        obs_X_df,
        feature_columns,
        predictors,
        model_name=model_name,
        include_fdr=include_fdr,
        threads=threads,
    )
    model_spec = None
    if save_model_spec_yaml:
        model_spec = _build_model_fit_model_spec(
            results,
            fit_method="ols",
            model_name=model_name,
            predictors=predictors,
            layer=layer,
            use_raw=use_raw,
        )

    # convert numeric columns to numeric dtype where possible
    num_cols = [
                col for col in results.columns
                if pd.to_numeric(results[col], errors="coerce").notna().all()
            ]
    if 'var_names' in num_cols:
        num_cols.remove('var_names')
    if len(num_cols) > 0:
        results[num_cols] = results[num_cols].apply(pd.to_numeric)

    # add adata.var columns to the results dataframe if specified
    if add_adata_var_column_key_list and work_adata is not None:
        for var_col_key in add_adata_var_column_key_list:
            if var_col_key in work_adata.var.columns:
                var_col_values = work_adata.var[var_col_key]
                # merge on index (var_names expected to match adata.var index)
                results = results.merge(var_col_values, left_index=True, right_index=True, how='left', suffixes=('', f'_{var_col_key}'))
            else:
                print(f"Warning: '{var_col_key}' not found in work_adata.var columns. Skipping this column.")

    # add results to work_adata.uns if specified
    if save_result_to_adata_uns_as_dict and work_adata is not None:
        key = f'OLS_model_results_{model_name}'
        if 'ols_model_results' not in work_adata.uns:
            work_adata.uns['ols_model_results'] = {}
        work_adata.uns['ols_model_results'][key] = results
        print(f"Added fit_smf_ols_models_and_summarize_wide results to work_adata.uns['ols_model_results']['{key}']")
        if save_model_spec_yaml:
            if 'ols_model_specs' not in work_adata.uns:
                work_adata.uns['ols_model_specs'] = {}
            work_adata.uns['ols_model_specs'][key] = model_spec
            print(f"Added model_spec to work_adata.uns['ols_model_specs']['{key}']")

        # optionally also save into the original adata.uns (useful when work_adata is a filtered copy)
        if save_results_to_original_adata_uns and work_adata is not adata:
            if 'ols_model_results' not in adata.uns:
                adata.uns['ols_model_results'] = {}
            adata.uns['ols_model_results'][key] = results
            print(f"Also wrote results to original adata.uns['ols_model_results']['{key}']")
            if save_model_spec_yaml:
                if 'ols_model_specs' not in adata.uns:
                    adata.uns['ols_model_specs'] = {}
                adata.uns['ols_model_specs'][key] = model_spec
                print(f"Also wrote model_spec to original adata.uns['ols_model_specs']['{key}']")

    # save the results dataframe to the save_path if requested
    if save_table and save_path is not None:
        _save_model_fit_results_csv(results, save_path)
        if save_model_spec_yaml:
            _save_model_spec_yaml(model_spec, save_path)

    # return either results or (results, work_adata) if requested and work_adata is a filtered copy
    if return_filtered_adata and work_adata is not adata:
        return results, work_adata
    return results

def fit_smf_mixedlm_models_and_summarize_wide(
        obs_X_df,
        feature_columns=None, 
        predictors=None,
        group=None,
        model_name='mixedlm',
        reml=True,
        include_fdr=True,
        threads: int = 1,
    ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if predictors is None or len(predictors) == 0:
        raise ValueError("fit_smf_mixedlm_models_and_summarize_wide requires a non-empty predictors list.")
    if group is None:
        raise ValueError("fit_smf_mixedlm_models_and_summarize_wide requires a non-empty group column name.")

    # pandas does not guarantee thread-safe copying from a shared DataFrame.
    # Keep only selection and the deep copy serial; preprocessing and fitting stay parallel.
    dataframe_copy_lock = _threading.Lock()

    def fit_feature(feature, warning_state):
        columns2keep = [feature] + predictors + [group]
        missing_cols = [col for col in columns2keep if col not in obs_X_df.columns]
        if missing_cols:
            raise ValueError(
                f"[{model_name}] Missing required columns for feature '{feature}': {missing_cols}."
            )
        with dataframe_copy_lock:
            df = obs_X_df[columns2keep].copy(deep=True)
        # Scalar replacements avoid pandas' list-replacement path, which can raise
        # IndexError while updating Copy-on-Write block references.
        df.replace(np.inf, np.nan, inplace=True)
        df.replace(-np.inf, np.nan, inplace=True)
        # Coerce numeric-like predictors (e.g. Age loaded as strings/categories) to numeric
        # so formula terms remain continuous instead of categorical dummies.
        for predictor in predictors:
            series = df[predictor]
            if not series.notna().any():
                continue
            numeric_values = pd.to_numeric(series, errors="coerce")
            if numeric_values.notna().sum() == series.notna().sum():
                df[predictor] = numeric_values.astype(float)
        complete_case_mask = df.notna().all(axis=1)
        n_complete = int(complete_case_mask.sum())
        if n_complete == 0:
            missing_counts = df.isna().sum().to_dict()
            raise ValueError(
                f"[{model_name}] No complete-case rows for feature '{feature}' "
                f"with predictors {predictors} and group '{group}'. "
                f"Missing counts by column: {missing_counts}."
            )
        df = df.loc[complete_case_mask]
        n_groups = df[group].nunique(dropna=True)
        if n_groups < 2:
            raise ValueError(
                f"[{model_name}] Need at least 2 non-empty groups in '{group}' for feature '{feature}', "
                f"but found {n_groups} after complete-case filtering."
            )
        predictors_q = [f'Q("{p}")' for p in predictors]
        model_string = f'Q("{feature}") ~ {" + ".join(predictors_q)}'
        summary_formula = f'{feature} ~ {" + ".join(predictors)} | {group}'
        with _capture_model_fit_warnings(warning_state) as caught_warnings:
            model = smf.mixedlm(model_string, df, groups=df[group]).fit(reml=reml)
        return feature, (model, caught_warnings, summary_formula)

    # Store models and any fit warnings in a dictionary keyed by feature.
    models = dict(
        _run_feature_fits(
            feature_columns,
            fit_feature,
            threads,
        )
    )

    # make a results dataframe from the dict of models
    summary_rows = []
    for feature_name in feature_columns:
        model, caught_warnings, model_string = models[feature_name]
        converged = getattr(model, "converged", None)
        if converged is None and hasattr(model, "mle_retvals"):
            converged = model.mle_retvals.get("converged")
        if converged is None:
            converged = True  # OLS solves in closed form, so treat as converged by default
        warning_messages = [f"{w.category.__name__}: {w.message}" for w in caught_warnings]
        ci = model.conf_int()
        summary_data = {
            f'{model_name}_Log-Likelihood': model.llf,
            f'{model_name}_reml': model.reml,
            f'{model_name}_AIC': model.aic,
            f'{model_name}_BIC': model.bic,
            f'{model_name}_Formula': model_string,
            f'{model_name}_nobs': model.nobs,
            f'{model_name}_n_groups': len(pd.unique(model.model.groups)),
            f'{model_name}_Method': "REML" if model.reml else "ML",
            f'{model_name}_Scale': model.scale,
            f'{model_name}_Converged': converged,
            f'{model_name}_Warnings': "; ".join(warning_messages) if warning_messages else np.nan,
        }
        if getattr(model, "cov_re", None) is not None:
            for re_name, var in zip(model.cov_re.index, np.diag(model.cov_re)):
                summary_data[f'{model_name}_Var_RE_{re_name}'] = var
        summary_data[f'{model_name}_Var_Residual'] = model.scale
        for param_name in model.params.index:
            clean_param = param_name
            if clean_param.startswith('Q("') and clean_param.endswith('")'):
                clean_param = clean_param[3:-2]
            summary_data[f'{model_name}_Coef_{clean_param}'] = model.params[param_name]
            summary_data[f'{model_name}_StdErr_{clean_param}'] = model.bse[param_name]
            summary_data[f'{model_name}_tStat_{clean_param}'] = model.tvalues[param_name]
            summary_data[f'{model_name}_P>|z|_{clean_param}'] = model.pvalues[param_name]
            ci_low, ci_high = ci.loc[param_name]
            summary_data[f'{model_name}_CI_low_{clean_param}'] = ci_low
            summary_data[f'{model_name}_CI_high_{clean_param}'] = ci_high
        random_effects = {}
        try:
            random_effects = getattr(model, "random_effects", {})
        except ValueError as e:
            # mixedlm can fail to invert a singular covariance matrix; keep going but note it
            warning_messages.append(f"Random effects unavailable: {e}")
        for grp_label, random_effect in random_effects.items():
            for re_name, re_val in random_effect.items():
                clean_re = re_name
                if clean_re.startswith('Q("') and clean_re.endswith('")'):
                    clean_re = clean_re[3:-2]
                summary_data[f'{model_name}_RE_{grp_label}_{clean_re}'] = re_val
        summary_data[f'{model_name}_Warnings'] = "; ".join(warning_messages) if warning_messages else np.nan
        summary_rows.append(summary_data)

    # make the final results dataframe
    results = pd.DataFrame(summary_rows, index=feature_columns)
    if include_fdr:
        pval_cols = [c for c in results.columns if c.startswith(f'{model_name}_P>|z|_')]
        for col in pval_cols:
            mask = results[col].notna()
            if not mask.any():
                continue
            _, qvals, _, _ = multipletests(results.loc[mask, col], method='fdr_bh')
            fdr_col = f'{col}_FDR'
            results[fdr_col] = np.nan
            results.loc[mask, fdr_col] = qvals
    var_names = feature_columns
    results['var_names'] = var_names 
    # place 'var_names' as the first column
    cols = results.columns.tolist()
    results = results[['var_names'] + [col for col in cols if col != 'var_names']]
    return results


def old_fit_smf_mixedlm_models_and_summarize_adata(
        adata,layer=None,use_raw=False,
        feature_columns=None,
        predictors=None,
        group=None,
        model_name='mixedlm_predictors',
        reml=True,
        add_adata_var_column_key_list=None,
        save_table=False,
        save_path=None,
        save_result_to_adata_uns_as_dict=False,
        include_fdr=True,
            ):
    obs_X_df=make_df_obs_adataX(adata,layer=layer,use_raw=use_raw,include_obs=True,)
    feature_columns=feature_columns if feature_columns is not None else adata.var_names.tolist()
    results=fit_smf_mixedlm_models_and_summarize_wide(obs_X_df, feature_columns, predictors, group=group,model_name=model_name,reml=reml, include_fdr=include_fdr)
    # convert numeric columns to numeric dtype
    num_cols = [
                col for col in results.columns
                if pd.to_numeric(results[col], errors="coerce").notna().all()
            ]
    if 'var_names' in num_cols:     # remove 'var_names' from num_cols
        num_cols.remove('var_names')
    results[num_cols] = results[num_cols].apply(pd.to_numeric)

    # add adata.var columns to the results dataframe if specified
    if add_adata_var_column_key_list is not None and adata is not None:
        # add adata.var columns to the results dataframe
        for var_col_key in add_adata_var_column_key_list:
            if var_col_key in adata.var.columns:
                var_col_values = adata.var[var_col_key]
                results = results.merge(var_col_values, left_index=True, right_index=True, how='left', suffixes=('', f'_{var_col_key}'))
            else:
                print(f"Warning: '{var_col_key}' not found in adata.var columns. Skipping this column.")

    # add results to adata.uns if specified
    if save_result_to_adata_uns_as_dict and adata is not None:
        key=f'mixedlm_model_results_{model_name}'
        if 'mixedlm_model_results' not in adata.uns:
            adata.uns['mixedlm_model_results'] = {}
        adata.uns['mixedlm_model_results'][key] = results
        print(f"Added fit_smf_mixedlm_models_and_summarize_wide  results to adata.uns['mixedlm_model_results']['{key}']")

    # save the results dataframe to the save_path
    if save_table and save_path is not None:
        _save_model_fit_results_csv(results, save_path)

    return results

def fit_smf_mixedlm_models_and_summarize_adata(
        adata,
        layer=None,
        use_raw=False,
        feature_columns=None,
        predictors=None,
        group=None,
        model_name='mixedlm_predictors',
        reml=True,
        add_adata_var_column_key_list=None,
        save_table=False,
        save_model_spec_yaml: bool = False,
        save_path=None,
        save_result_to_adata_uns_as_dict=False,
        include_fdr=True,
        # --- new filter args ---
        dataset_cfg=None,
        filter_obs_boolean_column=None,
        filter_obs_column_key=None,
        filter_obs_column_values_list=None,
        filter_obs_copy=True,
        # when filtered internally, optionally also write results into the original adata.uns
        save_results_to_original_adata_uns: bool = False,
        # whether to return the filtered adata (work_adata) in addition to results
        return_filtered_adata: bool = False,
        threads: int = 1,
    ):
    """
    Fit MixedLM models for features in an AnnData and return a summary DataFrame.

    New behaviour:
      - If any of dataset_cfg, filter_obs_boolean_column, filter_obs_column_key,
        or filter_obs_column_values_list are provided, a filtered AnnData
        (work_adata) is created via CFG_filter_adata_by_obs and used for the fit.
      - If save_result_to_adata_uns_as_dict is True results are saved to
        work_adata.uns['mixedlm_model_results'][f'mixedlm_model_results_{model_name}'].
      - If save_results_to_original_adata_uns is True and work_adata is a filtered
        copy, the same results are also saved into the original adata.uns.
      - return_filtered_adata=True will return (results, work_adata) instead of results.
      - threads>1 fits independent features concurrently while preserving result order.

    Backwards-compatible defaults preserve previous behaviour when no filter args are given.
    """
    # Local imports to avoid changing top-of-file imports and to keep the patch minimal.
    from .._preprocessing._adata_row_operations import CFG_filter_adata_by_obs
    from .._io._IO import make_df_obs_adataX
    import pandas as pd
    import numpy as np

    # If any filter args provided, create a filtered work_adata using the repo helper
    if any([dataset_cfg, filter_obs_boolean_column, filter_obs_column_key, filter_obs_column_values_list]):
        work_adata = CFG_filter_adata_by_obs(
            adata,
            dataset_cfg=dataset_cfg,
            filter_obs_boolean_column=filter_obs_boolean_column,
            filter_obs_column_key=filter_obs_column_key,
            filter_obs_column_values_list=filter_obs_column_values_list,
            copy=filter_obs_copy,
        )
    else:
        work_adata = adata

    # Validate/normalize list-like inputs coming from YAML
    predictors = _ensure_list(predictors, "predictors")
    add_adata_var_column_key_list = _ensure_list(add_adata_var_column_key_list, "add_adata_var_column_key_list")
    if save_model_spec_yaml and (not save_table or save_path is None):
        raise ValueError("save_model_spec_yaml=True requires save_table=True and save_path to be set.")

    # group is required for mixedlm; validate early with a clear error
    if group is None:
        raise ValueError("fit_smf_mixedlm_models_and_summarize_adata requires a 'group' argument (the grouping column name in adata.obs).")

    # Build the obs_X_df using the (possibly filtered) work_adata
    obs_X_df = make_df_obs_adataX(work_adata, layer=layer, use_raw=use_raw, include_obs=True,)
    feature_columns = feature_columns if feature_columns is not None else work_adata.var_names.tolist()

    # Delegate to the wide-version which contains the per-feature model-fitting logic
    results = fit_smf_mixedlm_models_and_summarize_wide(
        obs_X_df,
        feature_columns,
        predictors,
        group=group,
        model_name=model_name,
        reml=reml,
        include_fdr=include_fdr,
        threads=threads,
    )
    model_spec = None
    if save_model_spec_yaml:
        model_spec = _build_model_fit_model_spec(
            results,
            fit_method="mixedlm",
            model_name=model_name,
            predictors=predictors,
            layer=layer,
            use_raw=use_raw,
            group=group,
            reml=reml,
        )

    # convert numeric columns to numeric dtype where possible
    num_cols = [
                col for col in results.columns
                if pd.to_numeric(results[col], errors="coerce").notna().all()
            ]
    if 'var_names' in num_cols:
        num_cols.remove('var_names')
    if len(num_cols) > 0:
        results[num_cols] = results[num_cols].apply(pd.to_numeric)

    # add adata.var columns to the results dataframe if specified
    if add_adata_var_column_key_list and work_adata is not None:
        for var_col_key in add_adata_var_column_key_list:
            if var_col_key in work_adata.var.columns:
                var_col_values = work_adata.var[var_col_key]
                results = results.merge(var_col_values, left_index=True, right_index=True, how='left', suffixes=('', f'_{var_col_key}'))
            else:
                print(f"Warning: '{var_col_key}' not found in work_adata.var columns. Skipping this column.")

    # add results to work_adata.uns if specified
    if save_result_to_adata_uns_as_dict and work_adata is not None:
        key = f'mixedlm_model_results_{model_name}'
        if 'mixedlm_model_results' not in work_adata.uns:
            work_adata.uns['mixedlm_model_results'] = {}
        work_adata.uns['mixedlm_model_results'][key] = results
        print(f"Added fit_smf_mixedlm_models_and_summarize_wide results to work_adata.uns['mixedlm_model_results']['{key}']")
        if save_model_spec_yaml:
            if 'mixedlm_model_specs' not in work_adata.uns:
                work_adata.uns['mixedlm_model_specs'] = {}
            work_adata.uns['mixedlm_model_specs'][key] = model_spec
            print(f"Added model_spec to work_adata.uns['mixedlm_model_specs']['{key}']")

        # optionally also save into the original adata.uns (useful when work_adata is a filtered copy)
        if save_results_to_original_adata_uns and work_adata is not adata:
            if 'mixedlm_model_results' not in adata.uns:
                adata.uns['mixedlm_model_results'] = {}
            adata.uns['mixedlm_model_results'][key] = results
            print(f"Also wrote results to original adata.uns['mixedlm_model_results']['{key}']")
            if save_model_spec_yaml:
                if 'mixedlm_model_specs' not in adata.uns:
                    adata.uns['mixedlm_model_specs'] = {}
                adata.uns['mixedlm_model_specs'][key] = model_spec
                print(f"Also wrote model_spec to original adata.uns['mixedlm_model_specs']['{key}']")

    # save the results dataframe to the save_path if requested
    if save_table and save_path is not None:
        _save_model_fit_results_csv(results, save_path)
        if save_model_spec_yaml:
            _save_model_spec_yaml(model_spec, save_path)

    # return either results or (results, work_adata) if requested and work_adata is a filtered copy
    if return_filtered_adata and work_adata is not adata:
        return results, work_adata
    return results
