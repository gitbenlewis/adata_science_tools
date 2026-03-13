# module imports
# module at : /home/ubuntu/projects/gitbenlewis/adata_science_tools/_tools/_model_fit.py
## updated 2026-02-24 added guard to skip features with no complete cases after dropping NaN/inf, and to record the reason for skipping in the model summary dataframe
## updated 2026-02-24 added guard to coerce numeric-like predictors to numeric dtype so they are treated as continuous in the formula instead of categorical dummies, and added this coercion step to both OLS and mixedlm functions
# updated 2026-03-13 add expectation model fit and correction

from .. _io._IO import make_df_obs_adataX

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import warnings

# Helper inside the module (near top)
def _ensure_list(x, name):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        raise TypeError(f"{name} must be a YAML list (e.g. ['Age','Gender']) not a single string.")
    raise TypeError(f"{name} must be list/tuple or None, got {type(x).__name__}")



def fit_smf_ols_models_and_summarize_wide(
        obs_X_df,
        feature_columns=None, 
        predictors=None,
        model_name='OLS',
        include_fdr=True,
    ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Store models and any fit warnings in a dictionary keyed by feature
    models = {}
    skipped_reasons = {}
    for feature in feature_columns:
        columns2keep = [feature] + predictors
        df = obs_X_df[columns2keep].replace([np.inf, -np.inf], np.nan)
        # Coerce numeric-like predictors (e.g. Age loaded as strings/categories) to numeric
        # so formula terms remain continuous instead of categorical dummies.
        for predictor in predictors:
            series = df[predictor]
            if not series.notna().any():
                continue
            numeric_values = pd.to_numeric(series, errors="coerce")
            if numeric_values.notna().sum() == series.notna().sum():
                df[predictor] = numeric_values.astype(float)
        predictors_q = [f'Q("{p}")' for p in predictors]
        model_string = f'Q("{feature}") ~ {" + ".join(predictors_q)}'
        model_summary_formula = f'{feature} ~ {" + ".join(predictors)}'
        complete_case_mask = df.notna().all(axis=1)
        if not complete_case_mask.any():
            skipped_reasons[feature] = (
                f"No complete-case rows after dropping NaN/inf for columns {columns2keep}."
            )
            models[feature] = (None, [], model_summary_formula)
            continue
        df = df.loc[complete_case_mask]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                models[feature] = (smf.ols(model_string, df).fit(), caught_warnings, model_summary_formula)
            except Exception as e:
                skipped_reasons[feature] = f"{type(e).__name__}: {e}"
                models[feature] = (None, caught_warnings, model_summary_formula)

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
        ci = model.conf_int()
        jb, jb_p, skew, kurtosis = sm.stats.stattools.jarque_bera(model.resid)
        omni, omni_p = sm.stats.stattools.omni_normtest(model.resid)
        dw = sm.stats.durbin_watson(model.resid)
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
            f'{model_name}_F-statistic': model.fvalue if model.f_pvalue is not None else np.nan,
            f'{model_name}_P(F-statistic)': model.f_pvalue if model.f_pvalue is not None else np.nan,
            f'{model_name}_Converged': converged,
            f'{model_name}_Warnings': warning_messages if warning_messages else np.nan,
        }
        for param_name in model.params.index:
            clean_param = param_name
            if clean_param.startswith('Q("') and clean_param.endswith('")'):
                clean_param = clean_param[3:-2]
            summary_data[f'{model_name}_Coef_{clean_param}'] = model.params[param_name]
            summary_data[f'{model_name}_StdErr_{clean_param}'] = model.bse[param_name]
            summary_data[f'{model_name}_tStat_{clean_param}'] = model.tvalues[param_name]
            summary_data[f'{model_name}_P>|t|_{clean_param}'] = model.pvalues[param_name]
            ci_low, ci_high = ci.loc[param_name]
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
        import csv
        results.to_csv(save_path,float_format="%.6f", quoting=csv.QUOTE_MINIMAL,)
        print(f"Saved results  fit_smf_ols_models_and_summarize_wide results to {save_path}")
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

    # Build the obs_X_df using the (possibly filtered) work_adata
    obs_X_df = make_df_obs_adataX(work_adata, layer=layer, use_raw=use_raw, include_obs=True,)
    feature_columns = feature_columns if feature_columns is not None else work_adata.var_names.tolist()

    # Delegate the heavy lifting to the wide-version (unchanged behavior)
    results = fit_smf_ols_models_and_summarize_wide(obs_X_df, feature_columns, predictors, model_name=model_name, include_fdr=include_fdr)

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

        # optionally also save into the original adata.uns (useful when work_adata is a filtered copy)
        if save_results_to_original_adata_uns and work_adata is not adata:
            if 'ols_model_results' not in adata.uns:
                adata.uns['ols_model_results'] = {}
            adata.uns['ols_model_results'][key] = results
            print(f"Also wrote results to original adata.uns['ols_model_results']['{key}']")

    # save the results dataframe to the save_path if requested
    if save_table and save_path is not None:
        import csv
        results.to_csv(save_path, float_format="%.6f", quoting=csv.QUOTE_MINIMAL,)
        print(f"Saved results fit_smf_ols_models_and_summarize_wide results to {save_path}")

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
    ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if predictors is None or len(predictors) == 0:
        raise ValueError("fit_smf_mixedlm_models_and_summarize_wide requires a non-empty predictors list.")
    if group is None:
        raise ValueError("fit_smf_mixedlm_models_and_summarize_wide requires a non-empty group column name.")

    # Store models and any fit warnings in a dictionary keyed by feature
    models = {}
    for feature in feature_columns:
        columns2keep = [feature] + predictors + [group]
        missing_cols = [col for col in columns2keep if col not in obs_X_df.columns]
        if missing_cols:
            raise ValueError(
                f"[{model_name}] Missing required columns for feature '{feature}': {missing_cols}."
            )
        df = obs_X_df[columns2keep].replace([np.inf, -np.inf], np.nan)
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
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            models[feature] = (smf.mixedlm(model_string, df, groups=df[group]).fit(reml=reml), caught_warnings, summary_formula)

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
        import csv
        results.to_csv(save_path,float_format="%.6f", quoting=csv.QUOTE_MINIMAL,)
        print(f"Saved results  fit_smf_mixedlm_models_and_summarize_wide results to {save_path}")

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

    Backwards-compatible defaults preserve previous behaviour when no filter args are given.
    """
    # Local imports to avoid changing top-of-file imports and to keep the patch minimal.
    from .._preprocessing._adata_row_operations import CFG_filter_adata_by_obs
    from .._io._IO import make_df_obs_adataX
    import pandas as pd
    import numpy as np

    # small helper to validate YAML list-like inputs (allows list/tuple or None; disallows single string)
    def _ensure_list(x, name):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, str):
            raise TypeError(f"{name} must be a YAML list (e.g. ['Age','Gender']) not a single string.")
        raise TypeError(f"{name} must be list/tuple or None, got {type(x).__name__}")

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

        # optionally also save into the original adata.uns (useful when work_adata is a filtered copy)
        if save_results_to_original_adata_uns and work_adata is not adata:
            if 'mixedlm_model_results' not in adata.uns:
                adata.uns['mixedlm_model_results'] = {}
            adata.uns['mixedlm_model_results'][key] = results
            print(f"Also wrote results to original adata.uns['mixedlm_model_results']['{key}']")

    # save the results dataframe to the save_path if requested
    if save_table and save_path is not None:
        import csv
        results.to_csv(save_path, float_format="%.6f", quoting=csv.QUOTE_MINIMAL,)
        print(f"Saved results fit_smf_mixedlm_models_and_summarize_wide results to {save_path}")

    # return either results or (results, work_adata) if requested and work_adata is a filtered copy
    if return_filtered_adata and work_adata is not adata:
        return results, work_adata
    return results
