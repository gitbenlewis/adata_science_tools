''' _stat_tests.py '''
# module at /home/ubuntu/projects/gitbenlewis/adata_science_tools/anndata_tools/_stat_tests.py 

def diff_test(adata, layer=None, use_raw=False,
            groupby_key=None, groupby_key_target_values=[None], groupby_key_ref_values=[None],
            comparison_col_tag='_target_ref',
            nested_groupby_key_target_values=[(None,None)], nested_groupby_key_ref_values= [(None,None)],
            nested_comparison_col_tag='_target_con_ref_con',
            sortby=None,ascending=False,
            tests=['ttest_ind', 'ttest_rel','mannwhitneyu', 'WilcoxonSigned','ttest_rel_nested','WilcoxonSigned_nested'],
            pair_by_key=None ,
            add_values2results= False,
            add_adata_var_column_key_list=None,
            save_table=False,
            save_path=None,
            save_result_to_adata_uns_as_dict=False,
            logger=None,
            log_inputs=True,
            log_level="INFO",
            save_log=True,
            x_df=None,
            var_df=None,
            obs_df=None,

            ):
    """
    #### update4d 2025-12-27 added a logging and a save_log option
    #### updated 2025-12-11 changed pvals_corrected to pvals_FDR to be more explicit
    #### updated 2025-12-11 store values in adata.uns as str to avoid issues with saving lists in anndata
    #### updated 2025-12-11 to add save options and adata.var columns to results 
    #### ## updated 2025-05-29 sort by hypothesis stats
    #### ## updated 2025-05-28 added the hypothesis stats to the results DataFrame
    ## updated 2025-05-28 added option to use raw data from adata.raw if available
    ## updated 2025-03-12 fix WilcoxonSigned to paired not ranksums
    Perform various statistical comparisons (independent, paired, and nested paired tests) 
    between groups or conditions in an AnnData object (or x_df/obs_df/var_df inputs that 
    are converted to AnnData). This function can handle optional baseline or control 
    conditions (nested comparisons) and provides both parametric 
    (t-tests) and non-parametric (Mann-Whitney U, Wilcoxon Signed-rank) tests. It also 
    computes normality tests (Shapiro-Wilk and Kolmogorov-Smirnov) to help guide test 
    selection.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing observations (rows in adata.obs) and variables 
        (columns in adata.var). The primary data can be in `adata.X` or in a specified 
        `layer`. Ignored if x_df/obs_df/var_df are provided.
    layer : str, optional
        Key in `adata.layers` to use as the data matrix. If None, `adata.X` is used.
        Ignored if x_df/obs_df/var_df are provided.
    x_df : pandas.DataFrame or str or Path, optional
        Alternative data matrix (observations x variables). If provided, `obs_df` and
        `var_df` are required. If a string/Path, it is read with
        `pandas.read_csv(index_col=0)`.
    var_df : pandas.DataFrame or str or Path, optional
        Alternative variables DataFrame. Index must match `x_df` columns. If a string/Path,
        it is read with `pandas.read_csv(index_col=0)`.
    obs_df : pandas.DataFrame or str or Path, optional
        Alternative observations DataFrame. Index must match `x_df` index. If a string/Path,
        it is read with `pandas.read_csv(index_col=0)`.
    groupby_key : str, optional
        Column in `adata.obs` that defines the groups/conditions to compare.
        Required for all tests except when explicitly doing nested comparisons with
        `ttest_rel_nested` or `WilcoxonSigned_nested`.
    groupby_key_target_values : list, optional
        The target group(s) you want to compare against a reference. For example, 
        `['drug']` or `['treatmentA', 'treatmentB']`.
    groupby_key_ref_values : list, optional
        The reference group(s) to compare the target against. If None, all other 
        categories in `groupby_key` not specified in `groupby_key_target_values` will 
        be used.
    comparison_col_tag : str, optional
        Label appended to result columns for standard group-vs-group comparisons 
        (e.g., `_target_ref`).
    nested_groupby_key_target_values : list of tuples, optional
        For nested comparisons (e.g., target vs. its control), each tuple is 
        (target_category, target_control_category). For example, `[('drug', 'predose')]`.
    nested_groupby_key_ref_values : list of tuples, optional
        For nested comparisons (e.g., reference vs. its control), each tuple is 
        (reference_category, reference_control_category). For example, `[('vehicle', 'predoseVeh')]`.
    nested_comparison_col_tag : str, optional
        Label appended to result columns for nested group comparisons 
        (e.g., `_target_con_ref_con`).
    sortby : str, optional
        Column name by which to sort the final results DataFrame. If None, the function 
        attempts to sort by a relevant p-value column (in a priority order).
    tests : list of str, optional
        List of tests to perform. Possible values:
        - `'ttest_ind'` (unpaired/independent t-test)
        - `'ttest_rel'` (paired t-test)
        - `'mannwhitneyu'` (Mann-Whitney U test, unpaired)
        - `'WilcoxonSigned'` (Wilcoxon Signed-rank, paired)
        - `'ttest_rel_nested'` (nested paired t-test: (target - target_control) vs. (ref - ref_control))
        - `'WilcoxonSigned_nested'` (nested Wilcoxon: (target - target_control) vs. (ref - ref_control))
    pair_by_key : str, optional
        The column in `adata.obs` used to pair observations for paired tests 
        (e.g., animal ID or subject ID).
    add_values2results : bool, optional
        If True, additional columns with the raw, paired, or nested differences 
        (sorted by `pair_by_key`) are stored in the results.
    logger : logging.Logger, optional
        Logger to use for function output. Defaults to a module logger.
    log_inputs : bool, optional
        If True, logs the input arguments at INFO level.
    log_level : int or str, optional
        Logging level to set on the chosen logger (e.g., "INFO", "DEBUG"). Defaults to "INFO".
        If no handlers are attached to the module logger, a StreamHandler is added.
    save_log : bool, optional
        If True, writes logs to a file at the same location as save_path with ".log" appended.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by variable names (e.g., gene/analyte names). The columns 
        included depend on which tests and comparisons were performed. Below is a summary 
        of the possible columns:

        **General Columns**
        - ``var_names`` : Identifier for each variable (e.g., gene, protein, or metabolite).

        **Basic Group Statistics**
        - ``mean:{groupby_key_target}`` : Mean expression for the target group.
        - ``mean:{groupby_key_ref}`` : Mean expression for the reference group.
        - ``CVpct:{groupby_key_target}`` : Coefficient of variation (%) for the target group.
        - ``CVpct:{groupby_key_ref}`` : Coefficient of variation (%) for the reference group.
        - ``PCTchange{comparison_col_tag}`` : Percent change = (Mean target - Mean ref)/(Mean ref)*100.
        - ``fc{comparison_col_tag}`` : Fold change (target / reference).
        - ``l2fc{comparison_col_tag}`` : Log2 fold change.

        **Optional Raw/Paired/Nested Values** (only if ``add_values2results=True``)
        - ``{target_value}_values`` : Sorted expression values for the target group.
        - ``{ref_value}_values`` : Sorted expression values for the reference group.
        - ``{target_value}_minus_{ref_value}_values`` : Element-wise difference between the paired target and reference.
        - ``{pair_by_key}_order`` : The order of categories used to align paired samples.
        - For nested tests, analogous columns appear for the target-control, ref-control, and the nested difference:
          - ``{target}_values``, ``{target_control}_values``
          - ``{ref}_values``, ``{ref_control}_values``
          - ``{target}_minus_{target_control}_values``
          - ``{ref}_minus_{ref_control}_values``
          - ``{target}_diffcontrol_minus_{ref}_diffcontrol_values``
        
        **Normality Tests** (Shapiro-Wilk and Kolmogorov-Smirnov)
        - For independent groups (e.g., ``ttest_ind``, ``mannwhitneyu``):
          - ``shapiro_pvals: {groupby_key}_{target_value}`` 
          - ``ks_pvals: {groupby_key}_{target_value}``
          - ``shapiro_pvals: {groupby_key}_{ref_value}``
          - ``ks_pvals: {groupby_key}_{ref_value}``
        - For paired tests (e.g., ``ttest_rel``, ``WilcoxonSigned``):
          - ``shapiro_pvals: paired_diff ({groupby_key}_{target_value} - {groupby_key}_{ref_value})``
          - ``ks_pvals: paired_diff ({groupby_key}_{target_value} - {groupby_key}_{ref_value})``
        - For nested paired tests (e.g., ``ttest_rel_nested``, ``WilcoxonSigned_nested``):
          - ``shapiro_pvals: paired_NESTED_diffcontrol {target_value}_{ref_value}``
          - ``ks_pvals: paired_NESTED_diffcontrol {target_value}_{ref_value}``

        **Statistical Test Columns**
        - *Independent t-test* (``ttest_ind``):
          - ``ttest_ind_pvals{comparison_col_tag}``
          - ``ttest_ind_pvals_corrected{comparison_col_tag}``
        - *Mann-Whitney U* (``mannwhitneyu``):
          - ``mannwhitneyu_pvals{comparison_col_tag}``
          - ``mannwhitneyu_pvals_corrected{comparison_col_tag}``
        - *Paired t-test* (``ttest_rel``):
          - ``ttest_rel_pvals{comparison_col_tag}``
          - ``ttest_rel_pvals_corrected{comparison_col_tag}``
          - ``ttest_rel_mean_paired_fc{comparison_col_tag}``
          - ``ttest_rel_mean_paired_l2fc{comparison_col_tag}``
        - *Wilcoxon Signed-rank* (``WilcoxonSigned``):
          - ``WilcoxonSigned_pvals{comparison_col_tag}``
          - ``WilcoxonSigned_pvals_corrected{comparison_col_tag}``
          - ``WilcoxonSigned_mean_paired_l2fc{comparison_col_tag}``
        - *Nested Paired t-test* (``ttest_rel_nested``):
          - ``ttest_rel_nested_pvals{nested_comparison_col_tag}``
          - ``ttest_rel_nested_pvals_corrected{nested_comparison_col_tag}``
          - ``ttest_rel_nested_mean_paired_fcfc{nested_comparison_col_tag}``
          - ``ttest_rel_nested_mean_paired_l2fcfc{nested_comparison_col_tag}``
        - *Nested Wilcoxon Signed-rank* (``WilcoxonSigned_nested``):
          - ``WilcoxonSigned_nested_pvals{nested_comparison_col_tag}``
          - ``WilcoxonSigned_nested_pvals_corrected{nested_comparison_col_tag}``
          - ``WilcoxonSigned_nested_mean_paired_fcfc{nested_comparison_col_tag}``
          - ``WilcoxonSigned_nested_mean_paired_l2fcfc{nested_comparison_col_tag}``
    
    Notes
    -----
    - If a test (or certain groups) is not performed, the corresponding columns will not appear.
    - By default, the DataFrame is sorted by a primary p-value column based on the tests 
      requested, falling back to the next-available p-value column if the chosen one is not present.
    - The function automatically applies an FDR (Benjamini-Hochberg) correction to p-values.
    - The normality test columns (Shapiro and KS) can help decide whether parametric or non-parametric 
      tests are more appropriate.

    Examples
    --------
    # Example usage:
    diff_test(
         adata,
         layer=None,
         groupby_key="Treatment",
         groupby_key_target_values=["drug"],
         groupby_key_ref_values=["vehicle"],
         comparison_col_tag="_drug_vs_vehicle",
         nested_groupby_key_target_values=[('drug','predoseDrug')], nested_groupby_key_ref_values= [('vehicle','predoseVeh')], 
         nested_comparison_col_tag='_nested_predose_baseline_drug_vehicle',
         tests=['ttest_ind', 'ttest_rel','mannwhitneyu', 'WilcoxonSigned','ttest_rel_nested','WilcoxonSigned_nested'],
         pair_by_key="AnimalID"
     )
    # Example using CSVs instead of adata:
    diff_test(
         adata=None,
         x_df="x.csv",
         obs_df="obs.csv",
         var_df="var.csv",
         groupby_key="Treatment",
         groupby_key_target_values=["drug"],
         groupby_key_ref_values=["vehicle"],
         tests=['ttest_ind'],
     )
    """
    import logging
    import os
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    import pandas as pd
    import anndata

    log = logger or logging.getLogger(__name__)
    if log_level is not None:
        log.setLevel(log_level)
        if log.handlers:
            for handler in log.handlers:
                handler.setLevel(log_level)
        elif logger is None:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
            log.addHandler(handler)
            log.propagate = False
    if save_log:
        if save_path is None:
            raise ValueError("save_path is required when save_log is True.")
        log_path = f"{save_path}.log"
        log_path_abs = os.path.abspath(log_path)
        has_log_file = False
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                handler_path = os.path.abspath(getattr(handler, "baseFilename", ""))
                if handler_path == log_path_abs:
                    has_log_file = True
                    break
        if not has_log_file:
            file_handler = logging.FileHandler(log_path, 
                                               mode="w" # ouverwrite existing log file
                                               )
            if log_level is not None:
                file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
            log.addHandler(file_handler)
        log.info(f"diff_test log start: {datetime.now().isoformat(timespec='seconds')}")

    def _coerce_df(df_like, name):
        if df_like is None:
            return None
        if isinstance(df_like, pd.DataFrame):
            log.info(f"Using provided {name} with shape {df_like.shape}")
            return df_like
        if isinstance(df_like, (str, Path)):
            df = pd.read_csv(df_like, index_col=0)
            log.info(f"Loaded {name} from {df_like} with shape {df.shape}")
            return df
        raise TypeError(f"{name} must be a pandas.DataFrame or CSV path.")

    x_df = _coerce_df(x_df, "x_df")
    var_df = _coerce_df(var_df, "var_df")
    obs_df = _coerce_df(obs_df, "obs_df")

    uses_alt_inputs = any(v is not None for v in (x_df, var_df, obs_df))
    if not uses_alt_inputs and adata is None:
        raise ValueError("Please provide an adata object or x_df/var_df/obs_df inputs.")
    if uses_alt_inputs:
        if x_df is None or var_df is None or obs_df is None:
            raise ValueError("x_df, var_df, and obs_df must all be provided when using DataFrame/CSV inputs.")
        if adata is not None:
            log.info("x_df/var_df/obs_df provided; ignoring adata.")
        missing_obs = x_df.index.difference(obs_df.index)
        missing_xobs = obs_df.index.difference(x_df.index)
        if len(missing_obs) or len(missing_xobs):
            raise ValueError(
                f"obs_df index must match x_df index; missing in obs_df={len(missing_obs)}, "
                f"missing in x_df={len(missing_xobs)}"
            )
        if not x_df.index.equals(obs_df.index):
            obs_df = obs_df.reindex(x_df.index)
        missing_var = x_df.columns.difference(var_df.index)
        missing_xvar = var_df.index.difference(x_df.columns)
        if len(missing_var) or len(missing_xvar):
            raise ValueError(
                f"var_df index must match x_df columns; missing in var_df={len(missing_var)}, "
                f"missing in x_df={len(missing_xvar)}"
            )
        if not x_df.columns.equals(var_df.index):
            var_df = var_df.reindex(x_df.columns)
        adata = anndata.AnnData(X=x_df.to_numpy(), obs=obs_df.copy(), var=var_df.copy())
        layer = None
        use_raw = False
        log.info(f"Constructed AnnData from x_df/obs_df/var_df with shape {adata.shape}.")

    def _summarize_df_input(df_like):
        if isinstance(df_like, pd.DataFrame):
            return f"DataFrame shape={df_like.shape}"
        return df_like
    if log_inputs and log.isEnabledFor(logging.INFO):
        args_items = [
            ('adata', adata),
            ('layer', layer),
            ('use_raw', use_raw),
            ('x_df', _summarize_df_input(x_df)),
            ('var_df', _summarize_df_input(var_df)),
            ('obs_df', _summarize_df_input(obs_df)),
            ('groupby_key', groupby_key),
            ('groupby_key_target_values', groupby_key_target_values),
            ('groupby_key_ref_values', groupby_key_ref_values),
            ('comparison_col_tag', comparison_col_tag),
            ('nested_groupby_key_target_values', nested_groupby_key_target_values),
            ('nested_groupby_key_ref_values', nested_groupby_key_ref_values),
            ('nested_comparison_col_tag', nested_comparison_col_tag),
            ('sortby', sortby),
            ('ascending', ascending),
            ('tests', tests),
            ('pair_by_key', pair_by_key),
            ('add_values2results', add_values2results),
            ('add_adata_var_column_key_list', add_adata_var_column_key_list),
            ('save_table', save_table),
            ('save_path', save_path),
            ('save_result_to_adata_uns_as_dict', save_result_to_adata_uns_as_dict),
            ('logger', logger),
            ('log_inputs', log_inputs),
            ('log_level', log_level),
            ('save_log', save_log),
        ]
        args_lines = "\n".join([f"  {key}: {value}" for key, value in args_items])
        log.info(f"diff_test args:\n{args_lines}")

    def _safe_shapiro(vec):
        n = vec.shape[0]
        if n < 3 or n > 5000:
            return 'n<3_or_n>5000'
        return stats.shapiro(vec).pvalue

    def _aligned_pairs(X, adata, idx_a, idx_b, pair_key, label_a, label_b):
        ids_a = adata.obs.loc[idx_a, pair_key]
        ids_b = adata.obs.loc[idx_b, pair_key]
        common_ids = ids_a[ids_a.isin(ids_b)].unique()
        if len(common_ids) == 0:
            raise ValueError(f"No overlapping pairs on '{pair_key}' for {label_a} vs {label_b}.")
        mask_a = ids_a.isin(common_ids)
        mask_b = ids_b.isin(common_ids)
        Xa = X[idx_a][mask_a]
        Xb = X[idx_b][mask_b]
        order_a = np.argsort(adata.obs.loc[idx_a, pair_key][mask_a].cat.codes)
        order_b = np.argsort(adata.obs.loc[idx_b, pair_key][mask_b].cat.codes)
        Xa = Xa[order_a]
        Xb = Xb[order_b]
        pair_order_a = adata.obs.loc[idx_a, pair_key][mask_a].iloc[order_a].astype(str).tolist()
        pair_order_b = adata.obs.loc[idx_b, pair_key][mask_b].iloc[order_b].astype(str).tolist()
        log.info(
            f"Aligned pairs for {label_a} vs {label_b}: {len(common_ids)} overlapping IDs; "
            f"first 30 IDs: {pair_order_a[:30]}"
        )
        if Xa.shape[0] != Xb.shape[0] or pair_order_a != pair_order_b:
            raise ValueError(f"Mismatched pairing on '{pair_key}' for {label_a} vs {label_b}.")
        return Xa, Xb, pair_order_a

    ### ensure arguments are correct

    # if either or groupby_key  is None return error message
    if groupby_key is None:
            raise ValueError("Please provide a groupby key (column in adata.obs).")
    # Ensure `groupby` is categorical
    adata.obs[groupby_key] = adata.obs[groupby_key].astype("category")
    log.info(f"Groupby key '{groupby_key}' categories: {adata.obs[groupby_key].cat.categories.tolist()}")
    log.info(f"Number of observations per group in '{groupby_key}':\n{adata.obs[groupby_key].value_counts()}")
     # if tests is empty return error message or not a list return error message
    if not tests or not isinstance(tests, list):
        raise ValueError("Please provide a list of tests to perform.")
    # if if 'ttest_rel_nested' is not in tests: than groupby_key_target_values must have a value
    if 'ttest_rel_nested' not in tests and groupby_key_target_values is None:
        # if  groupby_key_target_values is None return error message
        raise ValueError("Please provide a groupby_key_target_values")
    if groupby_key_ref_values is None:
        log.info(f"groupby_key_ref_values is None, using all other values as groupby_key_ref_values")
        groupby_key_ref_values=[x for x in adata.obs[groupby_key].unique() if x not in groupby_key_target_values]
        log.info(f"Using groupby_key_ref_values: {groupby_key_ref_values}")
    # paired/nested tests require a pairing key
    paired_tests = {'ttest_rel', 'WilcoxonSigned', 'ttest_rel_nested', 'WilcoxonSigned_nested'}
    needs_pairing = any(t in tests for t in paired_tests)
    log.info(f"Requested tests: {tests}")
    log.info(f"Pairing required: {needs_pairing} (pair_by_key={pair_by_key})")
    if needs_pairing:
        if pair_by_key is None:
            raise ValueError("pair_by_key is required for paired or nested tests.")
        if pair_by_key not in adata.obs.columns:
            raise ValueError(f"pair_by_key '{pair_by_key}' not found in adata.obs.")
        if adata.obs[pair_by_key].isna().any():
            raise ValueError("pair_by_key contains missing values; paired tests require complete pairs.")
        adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        log.info(f"Pairing key '{pair_by_key}' has {adata.obs[pair_by_key].nunique()} unique IDs.")
        log.info(f"Pairing by key '{pair_by_key}' with categories: {adata.obs[pair_by_key].cat.categories.tolist()}")
        log.info(f"Number of unique pairs in '{pair_by_key}': {adata.obs[pair_by_key].nunique()}")
        log.info(f"Pairing key '{pair_by_key}' value counts:\n{adata.obs[pair_by_key].value_counts()}")

    ### #) extract the data matrix from the adata object and clean it
    # Select the data matrix
    #X = adata.layers[layer] if layer else adata.X
    data_source = None
    if use_raw and adata.raw is not None:
        X = adata.raw.X if layer is None else adata.raw.layers[layer]
        if layer:
            log.info(f"Using raw data from adata.raw.{layer}.")
            data_source = f"adata.raw.layers.{layer}"
        else:
            log.info(f"Using raw data from adata.raw.X.")
            data_source = "adata.raw.X"
    else:
        # Use the specified layer or the main data matrix
        if layer is not None and layer in adata.layers:
            X = adata.layers[layer]
            log.info(f"Using data from adata.layers.{layer}.")
            data_source = f"adata.layers.{layer}"
        else:
            X = adata.X
            log.info(f"Using data from adata.X.")
            data_source = "adata.X"
    if hasattr(X, "toarray"):  # Convert sparse matrix to dense if necessary
        X = X.toarray()
    log.info(f"Data matrix loaded from {data_source} with shape {X.shape}, dtype {X.dtype}")
    # Remove genes (columns) with zero expression across all cells
    log.info(f"Data matrix shape before removing zero-expression variables: {X.shape}")
    _X_shape_before_zero_removal=X.shape
    non_zero_genes = ~np.all(X == 0, axis=0)
    X = X[:, non_zero_genes]
    var_names = adata.var_names[non_zero_genes]
    log.info(f"Data matrix shape after removing zero-expression variables: {X.shape}")
    log.info(f"Removed { _X_shape_before_zero_removal[1] - X.shape[1]} zero-expression variables (genes).")

    ### Initialize results DataFrame
    results = pd.DataFrame({"var_names": var_names, }, index=var_names)

    group1_idx = adata.obs[groupby_key].isin(groupby_key_target_values)
    group2_idx = adata.obs[groupby_key].isin(groupby_key_ref_values)
    data1 = X[group1_idx].copy()
    data2 = X[group2_idx].copy()
    log.info(
        f"Group sizes for {groupby_key}: target={groupby_key_target_values} n={int(group1_idx.sum())}, "
        f"ref={groupby_key_ref_values} n={int(group2_idx.sum())}"
    )

    if groupby_key_target_values:
        # Boolean indexing to select cells in each group
        group1_idx = adata.obs[groupby_key].isin(groupby_key_target_values)
        group2_idx = adata.obs[groupby_key].isin(groupby_key_ref_values)

        data1 = X[group1_idx].copy()
        data2 = X[group2_idx].copy()

        # Compute mean expressions for each group (vectorized) and coefficient of variation
        mean_data1 = np.mean(data1, axis=0)
        cv_data1 = (np.std(data1, axis=0,ddof=1)/np.mean(data1, axis=0))*100
        mean_data2 = np.mean(data2, axis=0)
        cv_data2 = (np.std(data2, axis=0,ddof=1)/np.mean(data2, axis=0))*100

        # Compute  fold change with vectorized operation and handle zeros
        epsilon = 1e-9  # Small constant to avoid log issues
        valid_indices = (mean_data1 > 0) & (mean_data2 > 0)
        fc = np.empty(mean_data1.shape)
        fc.fill(np.nan)  # Initialize with NaN
        fc[valid_indices] = ((mean_data1[valid_indices] + epsilon) / (mean_data2[valid_indices] + epsilon))

        # Compute log fold change with vectorized operation and handle zeros
        epsilon = 1e-9  # Small constant to avoid log issues
        valid_indices = (mean_data1 > 0) & (mean_data2 > 0)
        logfc = np.empty(mean_data1.shape)
        logfc.fill(np.nan)  # Initialize with NaN
        logfc[valid_indices] = np.log2((mean_data1[valid_indices] + epsilon) / (mean_data2[valid_indices] + epsilon))

        # Compute PCT_diff with vectorized operation and handle zeros
        epsilon = 1e-9  # Small constant to avoid log issues
        valid_indices = (mean_data1 > 0) & (mean_data2 > 0)
        pct_diff = np.empty(mean_data1.shape)
        pct_diff.fill(np.nan)  # Initialize with NaN
        pct_diff[valid_indices] = ((mean_data1[valid_indices] - mean_data2[valid_indices]) / (mean_data2[valid_indices] + epsilon))*100


        # Add means and l2fc to the results DataFrame

        # make labels group1 and group2 labels
        _groupby_key_target_values_str='_'.join(groupby_key_target_values)
        _groupby_key_ref_values_str='_'.join(groupby_key_ref_values)
        group1_label = f"{groupby_key}_{_groupby_key_target_values_str}"
        group2_label = f"{groupby_key}_{_groupby_key_ref_values_str}"


        results[f'mean:{group1_label}'] =mean_data1
        results[f'mean:{group2_label}'] =mean_data2
        results[f'CVpct:{group1_label}'] =cv_data1
        results[f'CVpct:{group2_label}'] =cv_data2
        results[f'PCTchange{comparison_col_tag}'] =pct_diff
        results[f'fc{comparison_col_tag}'] =fc
        results[f'l2fc{comparison_col_tag}'] =logfc

    if add_values2results:
        ### add values to the results dataframe
        # if 'ttest_rel_nested' or 'wilcox_nested' in tests: than groupby_key_target_values must have a value and use to order the data values
        if 'ttest_rel_nested' in tests or 'WilcoxonSigned_nested' in tests:
            # first compute the nested difference
            # Boolean indexing to select cells in each group
            group_target_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][0]])
            group_targetControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][1]])
            group_ref_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][0]])
            group_refControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][1]])
            # Select the data matrix for each group
            data_target_rel, data_targetControl_rel, pair_order_target = _aligned_pairs(
                X, adata, group_target_idx, group_targetControl_idx, pair_by_key,
                nested_groupby_key_target_values[0][0], nested_groupby_key_target_values[0][1]
            )
            data_ref_rel, data_refControl_rel, pair_order_ref = _aligned_pairs(
                X, adata, group_ref_idx, group_refControl_idx, pair_by_key,
                nested_groupby_key_ref_values[0][0], nested_groupby_key_ref_values[0][1]
            )
            if pair_order_target != pair_order_ref:
                raise ValueError(f"Mismatched pairing on '{pair_by_key}' between target/control and ref/control.")
            pair_order = pair_order_target
            # Compute the difference for each pair 
            target_diff = data_target_rel - data_targetControl_rel
            ref_diff = data_ref_rel - data_refControl_rel
            nested_diff = target_diff - ref_diff

            # paired percentage change  target-ref
            pct_diff_target_ref_rel = np.mean(((data_target_rel - data_ref_rel) / (data_ref_rel + 1e-9))*100, axis=0)
            # add the paired percentage change for the target_diff and ref_diff
            pct_diff_targetControl_target_rel = np.mean(((data_target_rel - data_targetControl_rel) / (data_targetControl_rel + 1e-9))*100, axis=0)
            # add the percentage change values to the data frame
            results[f'paired_PCTchange{comparison_col_tag}'] = pct_diff_target_ref_rel
            results[f'paired_PCTchange_NESTED_diffcontrol{comparison_col_tag}'] = pct_diff_targetControl_target_rel

            # add values to the results dataframe
            results[f'{nested_groupby_key_target_values[0][0]}_values'] = data_target_rel.T.astype(str).tolist()
            results[f'{nested_groupby_key_target_values[0][1]}_values'] = data_targetControl_rel.T.astype(str).tolist()
            results[f'{nested_groupby_key_ref_values[0][0]}_values'] = data_ref_rel.T.astype(str).tolist()
            results[f'{nested_groupby_key_ref_values[0][1]}_values'] = data_refControl_rel.T.astype(str).tolist()
            # add diff values to the results dataframe
            # target-ref
            results[f'{nested_groupby_key_target_values[0][0]}_minus_{nested_groupby_key_ref_values[0][0]}_values'] = (data_target_rel - data_ref_rel).T.astype(str).tolist() 
            # target-control 
            results[f'{nested_groupby_key_target_values[0][0]}_minus_{nested_groupby_key_target_values[0][1]}_values'] = target_diff.T.astype(str).tolist()
            # ref-control 
            results[f'{nested_groupby_key_ref_values[0][0]}_minus_{nested_groupby_key_ref_values[0][1]}_values'] = ref_diff.T.astype(str).tolist()
            # nested diff (target-control) - (ref-control)
            results[f'{nested_groupby_key_target_values[0][0]}_diffcontrol_minus_{nested_groupby_key_ref_values[0][0]}_diffcontrol_values'] = nested_diff.T.astype(str).tolist()
            # add cvs for differences with control 
            cv_target_ref_diff= (np.std(data_target_rel - data_ref_rel, axis=0,ddof=1)/np.abs(np.mean(data_target_rel - data_ref_rel, axis=0)))*100
            cv_target_diff = (np.std(target_diff, axis=0,ddof=1)/np.abs(np.mean(target_diff, axis=0)))*100
            cv_ref_diff = (np.std(ref_diff, axis=0,ddof=1)/np.abs(np.mean(ref_diff, axis=0)))*100
            cv_nested_diff = (np.std(nested_diff, axis=0,ddof=1)/np.abs(np.mean(nested_diff, axis=0)))*100
            results[f'CVpct:{nested_groupby_key_target_values[0][0]}_minus_{nested_groupby_key_ref_values[0][0]}'] = cv_target_ref_diff
            results[f'CVpct:{nested_groupby_key_target_values[0][0]}_minus_{nested_groupby_key_target_values[0][1]}'] = cv_target_diff
            results[f'CVpct:{nested_groupby_key_ref_values[0][0]}_minus_{nested_groupby_key_ref_values[0][1]}'] = cv_ref_diff
            results[f'CVpct:'f'{nested_groupby_key_target_values[0][0]}_diffcontrol_minus_{nested_groupby_key_ref_values[0][0]}_diffcontrol_values'] = cv_nested_diff

            # add add the pair_by_key categories order to the results dataframe
            results[f'{pair_by_key}_order'] = [pair_order] * len(results)
            log.info(
                f"add_values2results: nested values stored for "
                f"{nested_groupby_key_target_values[0]} vs {nested_groupby_key_ref_values[0]}; "
                f"pair order stored in '{pair_by_key}_order'"
            )

        elif 'ttest_rel' in tests or 'WilcoxonSigned' in tests:
            # Ensure `pair_by_key` is categorical for proper sorting
            if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
                adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
            group1_idx = adata.obs[groupby_key].isin(groupby_key_target_values)
            group2_idx = adata.obs[groupby_key].isin(groupby_key_ref_values)
            data1_rel, data2_rel, pair_order = _aligned_pairs(
                X, adata, group1_idx, group2_idx, pair_by_key,
                groupby_key_target_values, groupby_key_ref_values
            )
            data1_rel_data2_rel_diff=data1_rel - data2_rel

            # paired percentage change  target-ref
            pct_diff_target_ref_rel = np.mean(((data1_rel - data2_rel) / (data2_rel + 1e-9))*100, axis=0)
            # add the percentage change values to the data frame
            results[f'paired_PCTchange{comparison_col_tag}'] = pct_diff_target_ref_rel
            # add values to the results dataframe
            # target
            results[f'{groupby_key_target_values[0]}_values'] = data1_rel.T.astype(str).tolist()
            # ref
            results[f'{groupby_key_ref_values[0]}_values'] = data2_rel.T.astype(str).tolist()
            # target-ref
            results[f'{groupby_key_target_values[0]}_minus_{groupby_key_ref_values[0]}_values'] = data1_rel_data2_rel_diff.T.astype(str).tolist()
            # add add the pair_by_key categories order to the results dataframe
            results[f'{pair_by_key}_order'] = [pair_order] * len(results)
            log.info(
                f"add_values2results: paired values stored for {groupby_key_target_values[0]} vs "
                f"{groupby_key_ref_values[0]}; pair order stored in '{pair_by_key}_order'"
            )
            # add cvs for differences with target-ref 
            cv_target_ref_diff= (np.std(data1_rel - data2_rel, axis=0,ddof=1)/np.abs(np.mean(data1_rel - data2_rel, axis=0)))*100
            results[f'CVpct:{groupby_key_target_values[0]}_minus_{groupby_key_ref_values[0]}'] = cv_target_ref_diff
        else:
            # add values to the results dataframe
            # target
            results[f'{groupby_key_target_values[0]}_values'] = data1.T.astype(str).tolist()
            # ref
            results[f'{groupby_key_ref_values[0]}_values'] = data2.T.astype(str).tolist()
            if sortby is not None and sortby in adata.obs.columns:
                order_key = sortby
                order_series = adata.obs[sortby]
            else:
                order_key = adata.obs_names.name or "obs_name"
                order_series = pd.Series(adata.obs_names, index=adata.obs.index)
            target_order = order_series[group1_idx].astype(str).tolist()
            ref_order = order_series[group2_idx].astype(str).tolist()
            results[f'{groupby_key_target_values[0]}_{order_key}_order'] = [target_order] * len(results)
            results[f'{groupby_key_ref_values[0]}_{order_key}_order'] = [ref_order] * len(results)
            log.info(
                f"add_values2results: independent values stored for {groupby_key_target_values[0]} vs "
                f"{groupby_key_ref_values[0]}; order_key='{order_key}'"
            )

    ### Perform statistical tests
    nested_group_sizes_logged = False
    if 'ttest_ind' in tests:
        # Perform vectorized t-tests
        t_stat, t_test_pvals = stats.ttest_ind(data1, data2, equal_var=False, axis=0)
        # Multiple testing correction
        _, t_test_pvals_corrected, _, _ = multipletests(t_test_pvals[np.isfinite(t_test_pvals)], method="fdr_bh")
        n_finite_ttest_ind = int(np.isfinite(t_test_pvals).sum())
        log.info(f"ttest_ind: corrected {n_finite_ttest_ind} of {len(t_test_pvals)} p-values with FDR.")
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        t_test_pvals_corrected_full = np.full_like(t_test_pvals, np.nan, dtype=float)
        t_test_pvals_corrected_full[np.isfinite(t_test_pvals)] = t_test_pvals_corrected
        # Add p-values to the results DataFrame
        results[f'ttest_ind_stat{comparison_col_tag}']=t_stat
        results[f'ttest_ind_pvals{comparison_col_tag}'] = t_test_pvals
        results[f'ttest_ind_pvals_FDR{comparison_col_tag}'] = t_test_pvals_corrected_full
    if 'mannwhitneyu' in tests:
        # Perform Mann-Whitney U tests with continuity correction
        u_statistic, u_test_pvals = stats.mannwhitneyu(data1, data2, axis=0, alternative='two-sided', use_continuity=True)
        #u_test_pvals = np.array([
        #    stats.mannwhitneyu(data1[:, i], data2[:, i], alternative='two-sided', use_continuity=True).pvalue
        #    for i in range(data1.shape[1])])
        # Multiple testing correction
        _, u_test_pvals_corrected, _, _ = multipletests(u_test_pvals[np.isfinite(u_test_pvals)], method="fdr_bh")
        n_finite_mannwhitneyu = int(np.isfinite(u_test_pvals).sum())
        log.info(f"mannwhitneyu: corrected {n_finite_mannwhitneyu} of {len(u_test_pvals)} p-values with FDR.")
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        u_test_pvals_corrected_full = np.full_like(u_test_pvals, np.nan, dtype=float)
        u_test_pvals_corrected_full[np.isfinite(u_test_pvals)] = u_test_pvals_corrected
        # Add p-values to the results DataFrame
        results[f'mannwhitneyu_stat{comparison_col_tag}'] = u_statistic
        results[f'mannwhitneyu_pvals{comparison_col_tag}'] = u_test_pvals
        results[f'mannwhitneyu_pvals_FDR{comparison_col_tag}'] = u_test_pvals_corrected_full

    ### add a shapiro and ks test for normality for the target and ref groups  for idependent between-subjects designs
    if 'ttest_ind' in tests  or 'mannwhitneyu' in tests:
        # data1 shapiro test # ks test
        #shapiro_stat_group1, shapiro_pvals_group1 = stats.shapiro(data1,axis=0)
        #ks_stat_group1, ks_pvals_group1 = stats.kstest(data1, 'norm',axis=0)
        shapiro_pvals_group1 =[_safe_shapiro(data1[:, i]) for i in range(data1.shape[1])]
        ks_pvals_group1 = [
            stats.kstest(
                data1[:, i],
                lambda x, mean=data1[:, i].mean(), std=data1[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(data1.shape[1])
        ]
        # data2 shapiro test # ks test
        #shapiro_stat_group2, shapiro_pvals_group2 = stats.shapiro(data2,axis=0)
        #ks_stat_group2, ks_pvals_group2 = stats.kstest(data2, 'norm',axis=0)
        shapiro_pvals_group2 =[_safe_shapiro(data2[:, i]) for i in range(data2.shape[1])]
        #ks_pvals_group2 = [stats.kstest(data2[:, i], 'norm').pvalue for i in range(data2.shape[1])]
        ks_pvals_group2 = [
            stats.kstest(
                data2[:, i],
                lambda x, mean=data2[:, i].mean(), std=data2[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(data2.shape[1])
        ]
        skip_shapiro_group1 = shapiro_pvals_group1.count('n<3_or_n>5000')
        skip_shapiro_group2 = shapiro_pvals_group2.count('n<3_or_n>5000')
        log.info(
            f"Shapiro skipped (n<3_or_n>5000): {group1_label} {skip_shapiro_group1}/{len(shapiro_pvals_group1)}, "
            f"{group2_label} {skip_shapiro_group2}/{len(shapiro_pvals_group2)}"
        )
        # Add p-values to the results DataFrame 
        results[f'shapiro_pvals: {group1_label}'] = shapiro_pvals_group1
        results[f'ks_pvals: {group1_label}'] = ks_pvals_group1
        results[f'shapiro_pvals: {group2_label}'] = shapiro_pvals_group2
        results[f'ks_pvals: {group2_label}'] = ks_pvals_group2

    if 'ttest_rel' in tests:
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        data1_rel, data2_rel, _ = _aligned_pairs(
            X, adata, group1_idx, group2_idx, pair_by_key,
            groupby_key_target_values, groupby_key_ref_values
        )
        data1_rel_data2_rel_diff= (data1_rel - data2_rel)
        # Perform vectorized t-tests
        t_rel_stat, t_rel_test_pvals = stats.ttest_rel(data1_rel, data2_rel, axis=0)
        # Multiple testing correction
        _, t_rel_test_pvals_corrected, _, _ = multipletests(t_rel_test_pvals[np.isfinite(t_rel_test_pvals)], method="fdr_bh")
        n_finite_ttest_rel = int(np.isfinite(t_rel_test_pvals).sum())
        log.info(f"ttest_rel: corrected {n_finite_ttest_rel} of {len(t_rel_test_pvals)} p-values with FDR.")
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        t_rel_test_pvals_corrected_full = np.full_like(t_rel_test_pvals, np.nan, dtype=float)
        t_rel_test_pvals_corrected_full[np.isfinite(t_rel_test_pvals)] = t_rel_test_pvals_corrected
        # calculate mean log2 fold change for the paired data
        mean_fc_rel = np.mean(((data1_rel + 1e-9) / (data2_rel + 1e-9)), axis=0)
        # calculate mean log2 fold change for the paired data
        mean_log2_fc_rel = np.mean(np.log2((data1_rel + 1e-9) / (data2_rel + 1e-9)), axis=0)
        # Add p-values to the results DataFrame
        results[f'ttest_rel_stat{comparison_col_tag}'] = t_rel_stat
        results[f'ttest_rel_pvals{comparison_col_tag}'] = t_rel_test_pvals
        results[f'ttest_rel_pvals_FDR{comparison_col_tag}'] = t_rel_test_pvals_corrected_full
        results[f'ttest_rel_mean_paired_fc{comparison_col_tag}'] =mean_fc_rel
        results[f'ttest_rel_mean_paired_l2fc{comparison_col_tag}'] =mean_log2_fc_rel
    if 'WilcoxonSigned' in tests:
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        data1_rel, data2_rel, _ = _aligned_pairs(
            X, adata, group1_idx, group2_idx, pair_by_key,
            groupby_key_target_values, groupby_key_ref_values
        )
        # Perform Wilcoxon rank-sum tests
        #w_stat, w_test_pvals = stats.ranksums(data1_rel, data2_rel)
        w_stat, w_test_pvals = stats.wilcoxon(data1_rel, data2_rel,alternative='two-sided',axis=0,correction=True)
        # Multiple testing correction
        _, w_test_pvals_corrected, _, _ = multipletests(w_test_pvals[np.isfinite(w_test_pvals)], method="fdr_bh")
        n_finite_wilcoxon = int(np.isfinite(w_test_pvals).sum())
        log.info(f"WilcoxonSigned: corrected {n_finite_wilcoxon} of {len(w_test_pvals)} p-values with FDR.")
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        w_test_pvals_corrected_full = np.full_like(w_test_pvals, np.nan, dtype=float)
        w_test_pvals_corrected_full[np.isfinite(w_test_pvals)] = w_test_pvals_corrected
        # calculate mean  fold change for the paired data
        mean_fc_rel = np.mean(((data1_rel + 1e-9) / (data2_rel + 1e-9)), axis=0)
        # calculate mean log2 fold change for the paired data
        mean_log2_fc_rel = np.mean(np.log2((data1_rel + 1e-9) / (data2_rel + 1e-9)), axis=0)
        # Add p-values to the results DataFrame
        results[f'WilcoxonSigned_stat{comparison_col_tag}'] = w_stat
        results[f'WilcoxonSigned_pvals{comparison_col_tag}'] = w_test_pvals
        results[f'WilcoxonSigned_pvals_FDR{comparison_col_tag}'] = w_test_pvals_corrected_full
        results[f'WilcoxonSigned_mean_paired_fc{comparison_col_tag}'] =mean_fc_rel
        results[f'WilcoxonSigned_mean_paired_l2fc{comparison_col_tag}'] =mean_log2_fc_rel

    ### add a shapiro and ks test for normality of the difference between the (target - ref) groups  for  within-subjects designs
    if 'ttest_rel' in tests  or 'WilcoxonSigned' in tests:
        #data1_rel_data2_rel_diff=data1_rel - data2_rel
        # data1_rel_data2_rel_diff shapiro test # ks test
        #shapiro_stat_data1_rel_data2_rel_diff, shapiro_pvals_data1_rel_data2_rel_diff= stats.shapiro(data1_rel_data2_rel_diff,axis=0)
        #ks_stat_data1_rel_data2_rel_diff, ks_pvals_data1_rel_data2_rel_diff = stats.kstest(data1_rel_data2_rel_diff, 'norm',axis=0)
        shapiro_pvals_data1_rel_data2_rel_diff =[_safe_shapiro(data1_rel_data2_rel_diff[:, i]) for i in range(data1_rel_data2_rel_diff.shape[1])]
        ks_pvals_data1_rel_data2_rel_diff = [
            stats.kstest(
                data1_rel_data2_rel_diff[:, i],
                lambda x, mean=data1_rel_data2_rel_diff[:, i].mean(), std=data1_rel_data2_rel_diff[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(data1_rel_data2_rel_diff.shape[1])
        ]
        skip_shapiro_paired = shapiro_pvals_data1_rel_data2_rel_diff.count('n<3_or_n>5000')
        log.info(
            f"Shapiro skipped (n<3_or_n>5000): paired_diff ({group1_label}-{group2_label}) "
            f"{skip_shapiro_paired}/{len(shapiro_pvals_data1_rel_data2_rel_diff)}"
        )
        # Add p-values to the results DataFrame
        results[f'shapiro_pvals: paired_diff ({group1_label}-{group2_label})'] = shapiro_pvals_data1_rel_data2_rel_diff
        results[f'ks_pvals: paired_diff ({group1_label}-{group2_label})'] = ks_pvals_data1_rel_data2_rel_diff

    if 'ttest_rel_nested' in tests:
        # first compute the nested difference
        # Boolean indexing to select cells in each group
        group_target_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][0]])
        group_targetControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][1]])
        group_ref_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][0]])
        group_refControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][1]])
        if not nested_group_sizes_logged:
            log.info(
                f"Nested group sizes for {groupby_key}: target={nested_groupby_key_target_values[0]} "
                f"n_target={int(group_target_idx.sum())}, n_target_control={int(group_targetControl_idx.sum())}, "
                f"ref={nested_groupby_key_ref_values[0]} n_ref={int(group_ref_idx.sum())}, "
                f"n_ref_control={int(group_refControl_idx.sum())}"
            )
            nested_group_sizes_logged = True
        # Select the data matrix for each group
        data_target_rel, data_targetControl_rel, pair_order_target = _aligned_pairs(
            X, adata, group_target_idx, group_targetControl_idx, pair_by_key,
            nested_groupby_key_target_values[0][0], nested_groupby_key_target_values[0][1]
        )
        data_ref_rel, data_refControl_rel, pair_order_ref = _aligned_pairs(
            X, adata, group_ref_idx, group_refControl_idx, pair_by_key,
            nested_groupby_key_ref_values[0][0], nested_groupby_key_ref_values[0][1]
        )
        if pair_order_target != pair_order_ref:
            raise ValueError(f"Mismatched pairing on '{pair_by_key}' between target/control and ref/control.")
        # Compute the difference for each pair and run the paired t-test
        target_diff = data_target_rel - data_targetControl_rel
        ref_diff = data_ref_rel - data_refControl_rel
        # Perform vectorized t-tests
        t_rel_nested_stat, t_rel_nested_test_pvals = stats.ttest_rel(target_diff, ref_diff, axis=0)
        # Multiple testing correction
        _, t_rel_nested_test_pvals_corrected, _, _ = multipletests(t_rel_nested_test_pvals[np.isfinite(t_rel_nested_test_pvals)], method="fdr_bh")
        n_finite_ttest_rel_nested = int(np.isfinite(t_rel_nested_test_pvals).sum())
        log.info(
            f"ttest_rel_nested: corrected {n_finite_ttest_rel_nested} of {len(t_rel_nested_test_pvals)} p-values with FDR."
        )
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        t_rel_nested_test_pvals_corrected_full = np.full_like(t_rel_nested_test_pvals, np.nan, dtype=float)
        t_rel_nested_test_pvals_corrected_full[np.isfinite(t_rel_nested_test_pvals)] = t_rel_nested_test_pvals_corrected
        # calculate mean fold change for the nested paired test data
        mean_fc_rel_nested = np.mean(((
            ((data_target_rel + 1e-9) / (data_targetControl_rel + 1e-9))/((data_ref_rel + 1e-9) / (data_refControl_rel + 1e-9)))
            ), axis=0)
        # calculate mean log2 fold change for the nested paired test data
        mean_log2_fc_rel_nested = np.mean(np.log2((
            ((data_target_rel + 1e-9) / (data_targetControl_rel + 1e-9))/((data_ref_rel + 1e-9) / (data_refControl_rel + 1e-9)))
            ), axis=0)
        # Add p-values to the results DataFrame
        results[f'ttest_rel_nested_stat{nested_comparison_col_tag}'] = t_rel_nested_stat
        results[f'ttest_rel_nested_pvals{nested_comparison_col_tag}'] = t_rel_nested_test_pvals
        results[f'ttest_rel_nested_pvals_FDR{nested_comparison_col_tag}'] = t_rel_nested_test_pvals_corrected_full
        results[f'ttest_rel_nested_mean_paired_fcfc{nested_comparison_col_tag}'] =mean_fc_rel_nested
        results[f'ttest_rel_nested_mean_paired_l2fcfc{nested_comparison_col_tag}'] =mean_log2_fc_rel_nested
    if 'WilcoxonSigned_nested' in tests:
        # first compute the nested difference
        # Boolean indexing to select cells in each group
        group_target_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][0]])
        group_targetControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][1]])
        group_ref_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][0]])
        group_refControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][1]])
        if not nested_group_sizes_logged:
            log.info(
                f"Nested group sizes for {groupby_key}: target={nested_groupby_key_target_values[0]} "
                f"n_target={int(group_target_idx.sum())}, n_target_control={int(group_targetControl_idx.sum())}, "
                f"ref={nested_groupby_key_ref_values[0]} n_ref={int(group_ref_idx.sum())}, "
                f"n_ref_control={int(group_refControl_idx.sum())}"
            )
            nested_group_sizes_logged = True
        # Select the data matrix for each group
        data_target_rel, data_targetControl_rel, pair_order_target = _aligned_pairs(
            X, adata, group_target_idx, group_targetControl_idx, pair_by_key,
            nested_groupby_key_target_values[0][0], nested_groupby_key_target_values[0][1]
        )
        data_ref_rel, data_refControl_rel, pair_order_ref = _aligned_pairs(
            X, adata, group_ref_idx, group_refControl_idx, pair_by_key,
            nested_groupby_key_ref_values[0][0], nested_groupby_key_ref_values[0][1]
        )
        if pair_order_target != pair_order_ref:
            raise ValueError(f"Mismatched pairing on '{pair_by_key}' between target/control and ref/control.")
        # Compute the difference for each pair and run the paired Wilcoxon rank-sum test
        target_diff = data_target_rel - data_targetControl_rel
        ref_diff = data_ref_rel - data_refControl_rel
        # Perform Wilcoxon rank-sum tests
        #w_nested_stat, w_nested_test_pvals = stats.ranksums(target_diff, ref_diff) ### wrong drrrdadrr: this breaks the pairing
        # Compute paired Wilcoxon on per-feature differences (keeps pairing intact)
        w_nested_stat, w_nested_test_pvals = stats.wilcoxon(
            target_diff,
            ref_diff,
            axis=0,
            alternative="two-sided",
            correction=True,   # keep continuity correction as before
            zero_method="wilcox",  # or "pratt" if you want to keep zeros
        )
        # Multiple testing correction
        _, w_nested_test_pvals_corrected, _, _ = multipletests(w_nested_test_pvals[np.isfinite(w_nested_test_pvals)], method="fdr_bh")
        n_finite_wilcoxon_nested = int(np.isfinite(w_nested_test_pvals).sum())
        log.info(
            f"WilcoxonSigned_nested: corrected {n_finite_wilcoxon_nested} of {len(w_nested_test_pvals)} p-values with FDR."
        )
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        w_nested_test_pvals_full_corrected = np.full_like(w_nested_test_pvals, np.nan, dtype=float)
        w_nested_test_pvals_full_corrected[np.isfinite(w_nested_test_pvals)] = w_nested_test_pvals_corrected

        # calculate mean fold change for the nested paired test data
        mean_fc_rel_nested = np.mean(((
            ((data_target_rel + 1e-9) / (data_targetControl_rel + 1e-9))/((data_ref_rel + 1e-9) / (data_refControl_rel + 1e-9)))
            ), axis=0)
        
        # calculate mean log2 fold change for the nested paired test data
        mean_log2_fc_rel_nested = np.mean(np.log2((
            ((data_target_rel + 1e-9) / (data_targetControl_rel + 1e-9))/((data_ref_rel + 1e-9) / (data_refControl_rel + 1e-9)))
            ), axis=0)
        # Add p-values to the results DataFrame
        results[f'WilcoxonSigned_nested_stat{nested_comparison_col_tag}'] = w_nested_stat
        results[f'WilcoxonSigned_nested_pvals{nested_comparison_col_tag}'] = w_nested_test_pvals
        results[f'WilcoxonSigned_nested_pvals_FDR{nested_comparison_col_tag}'] = w_nested_test_pvals_full_corrected
        results[f'WilcoxonSigned_nested_mean_paired_fcfc{nested_comparison_col_tag}'] =mean_fc_rel_nested
        results[f'WilcoxonSigned_nested_mean_paired_l2fcfc{nested_comparison_col_tag}'] =mean_log2_fc_rel_nested
    ### add a shapiro and ks test for normality for the nested difference [(Target-control) - (Ref-control)] data for  within-subjects design with baseline control
    if 'ttest_rel_nested' in tests or 'WilcoxonSigned_nested' in tests:
        nested_diff = target_diff - ref_diff
        # (Target-control)-(Ref-control) shapiro test # ks test
        #shapiro_stat_nested_diff, shapiro_pvals_nested_diff = stats.shapiro(nested_diff,axis=0)
        #ks_stat_nested_diff, ks_pvals_nested_diff = stats.kstest(nested_diff, 'norm',axis=0)
        shapiro_pvals_nested_diff =[_safe_shapiro(nested_diff[:, i]) for i in range(nested_diff.shape[1])]
        ks_pvals_nested_diff = [
            stats.kstest(
                nested_diff[:, i],
                lambda x, mean=nested_diff[:, i].mean(), std=nested_diff[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(nested_diff.shape[1])
        ]
        skip_shapiro_nested = shapiro_pvals_nested_diff.count('n<3_or_n>5000')
        log.info(
            f"Shapiro skipped (n<3_or_n>5000): paired_NESTED_diffcontrol "
            f"{nested_groupby_key_target_values[0][0]}_{nested_groupby_key_ref_values[0][0]} "
            f"{skip_shapiro_nested}/{len(shapiro_pvals_nested_diff)}"
        )
        # Add p-values to the results DataFrame
        # f'{nested_groupby_key_target_values[0][0]}_{nested_groupby_key_ref_values[0][0]}_NESTED_diff_values'
        results[f'shapiro_pvals: paired_NESTED_diffcontrol {nested_groupby_key_target_values[0][0]}_{nested_groupby_key_ref_values[0][0]}'] = shapiro_pvals_nested_diff
        results[f'ks_pvals: paired_NESTED_diffcontrol {nested_groupby_key_target_values[0][0]}_{nested_groupby_key_ref_values[0][0]}'] = ks_pvals_nested_diff

        
    ### Sort the results dataframe sortby
    # assign a column to sortby if not provided
    if sortby is None:
        if f'ttest_rel_nested_stat{nested_comparison_col_tag}' in results.columns:
            sortby = f'ttest_rel_nested_stat{nested_comparison_col_tag}'
        elif f'WilcoxonSigned_nested_stat{nested_comparison_col_tag}' in results.columns:
            sortby = f'WilcoxonSigned_nested_stat{nested_comparison_col_tag}'
        elif f'ttest_rel_stat{comparison_col_tag}' in results.columns:
            sortby = f'ttest_rel_stat{comparison_col_tag}'
        elif f'ttest_ind_stat{comparison_col_tag}' in results.columns:
            sortby = f'ttest_ind_stat{comparison_col_tag}'
        elif f'mannwhitneyu_stat{comparison_col_tag}' in results.columns:
            sortby = f'mannwhitneyu_stat{comparison_col_tag}'

    # Sort the results by absolute value of the selected hypothesis test statistic
    if sortby is not None and sortby in results.columns:
        results.sort_values(sortby, ascending=ascending, inplace=True, key=lambda x: x.abs(), na_position='last')
        top_var = results.index[0] if len(results.index) else None
        log.info(f"Sorted results by abs({sortby}), ascending={ascending}; top var_names={top_var}")
    elif sortby is None:
        log.info("sortby not provided and no default statistic available; results not sorted.")
    else:
        log.info(f"sortby '{sortby}' not in results columns; results not sorted.")


    # add results to adata.uns if specified
    #if save_result_to_adata_uns_as_dict and adata is not None:
    #    key=f'{groupby_key}_{_groupby_key_target_values_str}_over_{_groupby_key_ref_values_str}'
    #    if 'diff_test_results' not in adata.uns:
    #        adata.uns['diff_test_results'] = {}
    #    adata.uns['diff_test_results'][key] = results
    #    print(f"Added diff test results to adata.uns['diff_test_results']['{key}']")

    _groupby_key_target_values_str = '_'.join([str(v) for v in (groupby_key_target_values or []) if v is not None])
    if not _groupby_key_target_values_str:
        _groupby_key_target_values_str = '_'.join([str(v) for v in (nested_groupby_key_target_values[0] if nested_groupby_key_target_values else []) if v is not None]) or 'target'
    _groupby_key_ref_values_str = '_'.join([str(v) for v in (groupby_key_ref_values or []) if v is not None])
    if not _groupby_key_ref_values_str:
        _groupby_key_ref_values_str = '_'.join([str(v) for v in (nested_groupby_key_ref_values[0] if nested_groupby_key_ref_values else []) if v is not None]) or 'ref'

    if save_result_to_adata_uns_as_dict and adata is not None:
        def _to_json_if_listlike(x):
            import json
            if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
                return json.dumps([None if (isinstance(v, float) and np.isnan(v)) else v for v in list(x)])
            return x  # leave scalars alone
        key = f'{groupby_key}_{_groupby_key_target_values_str}_over_{_groupby_key_ref_values_str}'
        if 'diff_test_results' not in adata.uns:
            adata.uns['diff_test_results'] = {}
        results_for_uns = results.copy()
        # stringify list-in-cell columns so h5ad can write them
        list_cols = [c for c in results_for_uns.columns if c.endswith('_values') or c.endswith('_order')]
        log.info(
            f"Saving results to adata.uns['diff_test_results']['{key}'] "
            f"(rows={results_for_uns.shape[0]}, cols={results_for_uns.shape[1]}, list_cols={len(list_cols)})"
        )
        for c in list_cols:
            results_for_uns[c] = results_for_uns[c].apply(_to_json_if_listlike).astype(str)
        adata.uns['diff_test_results'][key] = results_for_uns
        log.info(f"Added diff test results to adata.uns['diff_test_results']['{key}']")


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
                log.warning(f"'{var_col_key}' not found in adata.var columns. Skipping this column.")

    # save the results dataframe to the save_path
    if save_table and save_path is not None:
        import csv
        results.to_csv(save_path,
                       #float_format="%.6f",
                        quoting=csv.QUOTE_MINIMAL,)
        log.info(f"Saved results diff test results to {save_path} (rows={results.shape[0]}, cols={results.shape[1]})")

    if save_log:
        log.info(f"diff_test log end: {datetime.now().isoformat(timespec='seconds')}")

    return results
