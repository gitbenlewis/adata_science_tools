# module imports
from .. anndata_io._IO import make_df_obs_adataX

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
import warnings


def fit_smf_ols_models_and_summarize_wide(
        obs_X_df,
        feature_columns=None, 
        predictors=None,
        model_name='OLS',
    ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Store models and any fit warnings in a dictionary keyed by feature
    models = {}
    for feature in feature_columns:
        columns2keep = [feature] + predictors
        df = obs_X_df[columns2keep]
        predictors_q = [f'Q("{p}")' for p in predictors]
        model_string = f'Q("{feature}") ~ {" + ".join(predictors_q)}'
        model_summary_formula = f'{feature} ~ {" + ".join(predictors)}'
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            models[feature] = (smf.ols(model_string, df).fit(), caught_warnings, model_summary_formula)

    # make a results dataframe from the dict of models
    summary_rows = []
    for feature_name in feature_columns:
        model, caught_warnings, model_string = models[feature_name]
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

    summary_df_single_row = pd.DataFrame(summary_rows, index=feature_columns)
    return summary_df_single_row


def fit_smf_ols_models_and_summarize_adata(adata,layer=None,use_raw=False, feature_columns=None, predictors=None, model_name='OLS_predictors'):
    obs_X_df=make_df_obs_adataX(adata,layer=layer,use_raw=use_raw,include_obs=True,)
    feature_columns=feature_columns if feature_columns is not None else adata.var_names.tolist()
    summary_df=fit_smf_ols_models_and_summarize_wide(obs_X_df, feature_columns, predictors, model_name=model_name)
    return summary_df



def fit_smf_mixedlm_models_and_summarize_wide(
        obs_X_df,
        feature_columns=None, 
        predictors=None,
        group=None,
        model_name='mixedlm',
        reml=True,
    ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Store models and any fit warnings in a dictionary keyed by feature
    models = {}
    for feature in feature_columns:
        columns2keep = [feature] + predictors+ [group]
        df = obs_X_df[columns2keep]
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
        warning_messages = "; ".join(f"{w.category.__name__}: {w.message}" for w in caught_warnings)
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
            f'{model_name}_Warnings': warning_messages if warning_messages else np.nan,
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
        for grp_label, random_effect in getattr(model, "random_effects", {}).items():
            for re_name, re_val in random_effect.items():
                clean_re = re_name
                if clean_re.startswith('Q("') and clean_re.endswith('")'):
                    clean_re = clean_re[3:-2]
                summary_data[f'{model_name}_RE_{grp_label}_{clean_re}'] = re_val
        summary_rows.append(summary_data)

    summary_df_single_row = pd.DataFrame(summary_rows, index=feature_columns)
    return summary_df_single_row


def fit_smf_mixedlm_models_and_summarize_adata(adata,layer=None,use_raw=False, feature_columns=None, predictors=None, group=None,model_name='mixedlm_predictors',reml=True,):
    obs_X_df=make_df_obs_adataX(adata,layer=layer,use_raw=use_raw,include_obs=True,)
    feature_columns=feature_columns if feature_columns is not None else adata.var_names.tolist()
    summary_df=fit_smf_mixedlm_models_and_summarize_wide(obs_X_df, feature_columns, predictors, group=group,model_name=model_name,reml=reml)
    return summary_df
