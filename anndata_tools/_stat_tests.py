
def diff_test(adata, layer=None, use_raw=False,groupby_key=None, groupby_key_target_values=[None], groupby_key_ref_values=[None],comparison_col_tag='_target_ref',
                      nested_groupby_key_target_values=[(None,None)], nested_groupby_key_ref_values= [(None,None)],   nested_comparison_col_tag='_target_con_ref_con',
                       sortby=None,tests=['ttest_ind', 'ttest_rel','mannwhitneyu', 'WilcoxonSigned','ttest_rel_nested','WilcoxonSigned_nested'],pair_by_key=None ,add_values2results= False):
    """
    #### ## updated 2025-05-29 sort by hypothesis stats
    #### ## updated 2025-05-28 added the hypothesis stats to the results DataFrame
    ## updated 2025-05-28 added option to use raw data from adata.raw if available
    ## updated 2025-03-12 fix WilcoxonSigned to paired not ranksums
    Perform various statistical comparisons (independent, paired, and nested paired tests) 
    between groups or conditions in an AnnData object. This function can handle optional 
    baseline or control conditions (nested comparisons) and provides both parametric 
    (t-tests) and non-parametric (Mann-Whitney U, Wilcoxon Signed-rank) tests. It also 
    computes normality tests (Shapiro-Wilk and Kolmogorov-Smirnov) to help guide test 
    selection.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing observations (rows in adata.obs) and variables 
        (columns in adata.var). The primary data can be in `adata.X` or in a specified 
        `layer`.
    layer : str, optional
        Key in `adata.layers` to use as the data matrix. If None, `adata.X` is used.
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
    """
    import numpy as np
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    ### ensure arguments are correct

    # if either or groupby_key  is None return error message
    if groupby_key is None:
            raise ValueError("Please provide a groupby key (column in adata.obs).")
    # Ensure `groupby` is categorical
    adata.obs[groupby_key] = adata.obs[groupby_key].astype("category")
     # if tests is empty return error message or not a list return error message
    if not tests or not isinstance(tests, list):
        raise ValueError("Please provide a list of tests to perform.")
    # if if 'ttest_rel_nested' is not in tests: than groupby_key_target_values must have a value
    if 'ttest_rel_nested' not in tests and groupby_key_target_values is None:
        # if  groupby_key_target_values is None return error message
        raise ValueError("Please provide a groupby_key_target_values")
    if groupby_key_ref_values is None:
        print('groupby_key_ref_values is None, using all other values as groupby_key_ref_values')
        groupby_key_ref_values=[x for x in adata.obs[groupby_key].unique() if x not in groupby_key_target_values]

    ### extract the data matrix from the adata object and clean it
    # Select the data matrix
    #X = adata.layers[layer] if layer else adata.X
    if use_raw and adata.raw is not None:
        X = adata.raw.X if layer is None else adata.raw.layers[layer]
        print(f'Using raw data from adata.raw.{layer}.' if layer else 'Using raw data from adata.raw.X.')
    else:
        # Use the specified layer or the main data matrix
        if layer is not None and layer in adata.layers:
            X = adata.layers[layer]
            print(f'Using data from adata.layers.{layer}.')
        else:
            X = adata.X
            print('Using data from adata.X.')
    if hasattr(X, "toarray"):  # Convert sparse matrix to dense if necessary
        X = X.toarray()
    # Remove genes (columns) with zero expression across all cells
    non_zero_genes = ~np.all(X == 0, axis=0)
    X = X[:, non_zero_genes]
    var_names = adata.var_names[non_zero_genes]

    ### Initialize results DataFrame
    results = pd.DataFrame({"var_names": var_names, }, index=var_names)

    group1_idx = adata.obs[groupby_key].isin(groupby_key_target_values)
    group2_idx = adata.obs[groupby_key].isin(groupby_key_ref_values)
    data1 = X[group1_idx].copy()
    data2 = X[group2_idx].copy()

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
            data_target=X[group_target_idx].copy()
            data_targetControl=X[group_targetControl_idx].copy()
            data_ref=X[group_ref_idx].copy()
            data_refControl=X[group_refControl_idx].copy()
            # sort the data by the pair_by_key
            # Ensure `pair_by_key` is categorical for proper sorting
            if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
                adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
            # Sort the data by `pair_by_key`
            data_target_rel = data_target[np.argsort(adata.obs.loc[group_target_idx, pair_by_key].cat.codes)]
            data_targetControl_rel = data_targetControl[np.argsort(adata.obs.loc[group_targetControl_idx, pair_by_key].cat.codes)]
            data_ref_rel = data_ref[np.argsort(adata.obs.loc[group_ref_idx, pair_by_key].cat.codes)]
            data_refControl_rel = data_refControl[np.argsort(adata.obs.loc[group_refControl_idx, pair_by_key].cat.codes)]
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
            results[f'{nested_groupby_key_target_values[0][0]}_values'] = data_target_rel.T.tolist()
            results[f'{nested_groupby_key_target_values[0][1]}_values'] = data_targetControl_rel.T.tolist()
            results[f'{nested_groupby_key_ref_values[0][0]}_values'] = data_ref_rel.T.tolist()
            results[f'{nested_groupby_key_ref_values[0][1]}_values'] = data_refControl_rel.T.tolist()
            # add diff values to the results dataframe
            # target-ref
            results[f'{nested_groupby_key_target_values[0][0]}_minus_{nested_groupby_key_ref_values[0][0]}_values'] = (data_target_rel - data_ref_rel).T.tolist() 
            # target-control 
            results[f'{nested_groupby_key_target_values[0][0]}_minus_{nested_groupby_key_target_values[0][1]}_values'] = target_diff.T.tolist()
            # ref-control 
            results[f'{nested_groupby_key_ref_values[0][0]}_minus_{nested_groupby_key_ref_values[0][1]}_values'] = ref_diff.T.tolist()
            # nested diff (target-control) - (ref-control)
            results[f'{nested_groupby_key_target_values[0][0]}_diffcontrol_minus_{nested_groupby_key_ref_values[0][0]}_diffcontrol_values'] = nested_diff.T.tolist()
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
            results[f'{pair_by_key}_order'] = [adata.obs[pair_by_key].cat.categories.tolist()] * len(results)
        elif 'ttest_rel' in tests or 'WilcoxonSigned' in tests:
            # Ensure `pair_by_key` is categorical for proper sorting
            if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
                adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
            group1_idx = adata.obs[groupby_key].isin(groupby_key_target_values)
            group2_idx = adata.obs[groupby_key].isin(groupby_key_ref_values)
            data1 = X[group1_idx].copy()
            data2 = X[group2_idx].copy()
            # sort the data by the pair_by_key
            data1_rel = data1[np.argsort(adata.obs.loc[group1_idx,pair_by_key].cat.codes)]
            data2_rel = data2[np.argsort(adata.obs.loc[group2_idx,pair_by_key].cat.codes)]
            data1_rel_data2_rel_diff=data1_rel - data2_rel

            # paired percentage change  target-ref
            pct_diff_target_ref_rel = np.mean(((data1_rel - data2_rel) / (data2_rel + 1e-9))*100, axis=0)
            # add the percentage change values to the data frame
            results[f'paired_PCTchange{comparison_col_tag}'] = pct_diff_target_ref_rel
            # add values to the results dataframe
            # target
            results[f'{groupby_key_target_values[0]}_values'] = data1_rel.T.tolist()
            # ref
            results[f'{groupby_key_ref_values[0]}_values'] = data2_rel.T.tolist()
            # target-ref
            results[f'{groupby_key_target_values[0]}_minus_{groupby_key_ref_values[0]}_values'] = data1_rel_data2_rel_diff.T.tolist()
            # add add the pair_by_key categories order to the results dataframe
            results[f'{pair_by_key}_order'] = [adata.obs[pair_by_key].cat.categories.tolist()] * len(results)
            # add cvs for differences with target-ref 
            cv_target_ref_diff= (np.std(data1_rel - data2_rel, axis=0,ddof=1)/np.abs(np.mean(data1_rel - data2_rel, axis=0)))*100
            results[f'CVpct:{groupby_key_target_values[0]}_minus_{groupby_key_ref_values[0]}'] = cv_target_ref_diff
        else:
            # add values to the results dataframe
            # target
            results[f'{groupby_key_target_values[0]}_values'] = data1.T.tolist()
            # ref
            results[f'{groupby_key_ref_values[0]}_values'] = data2.T.tolist()
            print('warning groupby_key_target_values and groupby_key_ref_values not used')


    ### Perform statistical tests
    if 'ttest_ind' in tests:
        # Perform vectorized t-tests
        t_stat, t_test_pvals = stats.ttest_ind(data1, data2, equal_var=False, axis=0)
        # Multiple testing correction
        _, t_test_pvals_corrected, _, _ = multipletests(t_test_pvals[np.isfinite(t_test_pvals)], method="fdr_bh")
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        t_test_pvals_corrected_full = np.full_like(t_test_pvals, np.nan, dtype=float)
        t_test_pvals_corrected_full[np.isfinite(t_test_pvals)] = t_test_pvals_corrected
        # Add p-values to the results DataFrame
        results[f'ttest_ind_stat{comparison_col_tag}']=t_stat
        results[f'ttest_ind_pvals{comparison_col_tag}'] = t_test_pvals
        results[f'ttest_ind_pvals_corrected{comparison_col_tag}'] = t_test_pvals_corrected_full
    if 'mannwhitneyu' in tests:
        # Perform Mann-Whitney U tests with continuity correction
        u_statistic, u_test_pvals = stats.mannwhitneyu(data1, data2, axis=0, alternative='two-sided', use_continuity=True)
        #u_test_pvals = np.array([
        #    stats.mannwhitneyu(data1[:, i], data2[:, i], alternative='two-sided', use_continuity=True).pvalue
        #    for i in range(data1.shape[1])])
        # Multiple testing correction
        _, u_test_pvals_corrected, _, _ = multipletests(u_test_pvals[np.isfinite(u_test_pvals)], method="fdr_bh")
        # Create a full array to store the corrected p-values, keeping NaNs where they were
        u_test_pvals_corrected_full = np.full_like(u_test_pvals, np.nan, dtype=float)
        u_test_pvals_corrected_full[np.isfinite(u_test_pvals)] = u_test_pvals_corrected
        # Add p-values to the results DataFrame
        results[f'mannwhitneyu_stat{comparison_col_tag}'] = u_statistic
        results[f'mannwhitneyu_pvals{comparison_col_tag}'] = u_test_pvals
        results[f'mannwhitneyu_pvals_corrected{comparison_col_tag}'] = u_test_pvals_corrected_full

    ### add a shapiro and ks test for normality for the target and ref groups  for idependent between-subjects designs
    if 'ttest_ind' in tests  or 'mannwhitneyu' in tests:
        # data1 shapiro test # ks test
        #shapiro_stat_group1, shapiro_pvals_group1 = stats.shapiro(data1,axis=0)
        #ks_stat_group1, ks_pvals_group1 = stats.kstest(data1, 'norm',axis=0)
        shapiro_pvals_group1 =[stats.shapiro(data1[:, i]).pvalue for i in range(data1.shape[1])]
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
        shapiro_pvals_group2 =[stats.shapiro(data2[:, i]).pvalue for i in range(data2.shape[1])]
        #ks_pvals_group2 = [stats.kstest(data2[:, i], 'norm').pvalue for i in range(data2.shape[1])]
        ks_pvals_group2 = [
            stats.kstest(
                data2[:, i],
                lambda x, mean=data2[:, i].mean(), std=data2[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(data2.shape[1])
        ]
        # Add p-values to the results DataFrame 
        results[f'shapiro_pvals: {group1_label}'] = shapiro_pvals_group1
        results[f'ks_pvals: {group1_label}'] = ks_pvals_group1
        results[f'shapiro_pvals: {group2_label}'] = shapiro_pvals_group2
        results[f'ks_pvals: {group2_label}'] = ks_pvals_group2

    if 'ttest_rel' in tests:
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        # sort the data by the pair_by_key
        data1_rel = data1[np.argsort(adata.obs.loc[group1_idx,pair_by_key].cat.codes)]
        data2_rel = data2[np.argsort(adata.obs.loc[group2_idx,pair_by_key].cat.codes)]
        data1_rel_data2_rel_diff= (data1_rel - data2_rel)
        # Perform vectorized t-tests
        t_rel_stat, t_rel_test_pvals = stats.ttest_rel(data1_rel, data2_rel, axis=0)
        # Multiple testing correction
        _, t_rel_test_pvals_corrected, _, _ = multipletests(t_rel_test_pvals[np.isfinite(t_rel_test_pvals)], method="fdr_bh")
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
        results[f'ttest_rel_pvals_corrected{comparison_col_tag}'] = t_rel_test_pvals_corrected_full
        results[f'ttest_rel_mean_paired_fc{comparison_col_tag}'] =mean_fc_rel
        results[f'ttest_rel_mean_paired_l2fc{comparison_col_tag}'] =mean_log2_fc_rel
    if 'WilcoxonSigned' in tests:
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        # sort the data by the pair_by_key
        data1_rel = data1[np.argsort(adata.obs.loc[group1_idx,pair_by_key].cat.codes)]
        data2_rel = data2[np.argsort(adata.obs.loc[group2_idx,pair_by_key].cat.codes)]
        # Perform Wilcoxon rank-sum tests
        #w_stat, w_test_pvals = stats.ranksums(data1_rel, data2_rel)
        w_stat, w_test_pvals = stats.wilcoxon(data1_rel, data2_rel,alternative='two-sided',axis=0,correction=True)
        # Multiple testing correction
        _, w_test_pvals_corrected, _, _ = multipletests(w_test_pvals[np.isfinite(w_test_pvals)], method="fdr_bh")
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
        results[f'WilcoxonSigned_pvals_corrected{comparison_col_tag}'] = w_test_pvals_corrected_full
        results[f'WilcoxonSigned_mean_paired_l2fc{comparison_col_tag}'] =mean_fc_rel
        results[f'WilcoxonSigned_mean_paired_l2fc{comparison_col_tag}'] =mean_log2_fc_rel

    ### add a shapiro and ks test for normality of the differenve between the (target - ref) groups  for  within-subjects designs
    if 'ttest_rel' in tests  or 'WilcoxonSigned' in tests:
        #data1_rel_data2_rel_diff=data1_rel - data2_rel
        # data1_rel_data2_rel_diff shapiro test # ks test
        #shapiro_stat_data1_rel_data2_rel_diff, shapiro_pvals_data1_rel_data2_rel_diff= stats.shapiro(data1_rel_data2_rel_diff,axis=0)
        #ks_stat_data1_rel_data2_rel_diff, ks_pvals_data1_rel_data2_rel_diff = stats.kstest(data1_rel_data2_rel_diff, 'norm',axis=0)
        shapiro_pvals_data1_rel_data2_rel_diff =[stats.shapiro(data1_rel_data2_rel_diff[:, i]).pvalue for i in range(data1_rel_data2_rel_diff.shape[1])]
        ks_pvals_data1_rel_data2_rel_diff = [
            stats.kstest(
                data1_rel_data2_rel_diff[:, i],
                lambda x, mean=data1_rel_data2_rel_diff[:, i].mean(), std=data1_rel_data2_rel_diff[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(data1_rel_data2_rel_diff.shape[1])
        ]
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
        # Select the data matrix for each group
        data_target=X[group_target_idx].copy()
        data_targetControl=X[group_targetControl_idx].copy()
        data_ref=X[group_ref_idx].copy()
        data_refControl=X[group_refControl_idx].copy()
        # sort the data by the pair_by_key
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        # Sort the data by `pair_by_key`
        data_target_rel = data_target[np.argsort(adata.obs.loc[group_target_idx, pair_by_key].cat.codes)]
        data_targetControl_rel = data_targetControl[np.argsort(adata.obs.loc[group_targetControl_idx, pair_by_key].cat.codes)]
        data_ref_rel = data_ref[np.argsort(adata.obs.loc[group_ref_idx, pair_by_key].cat.codes)]
        data_refControl_rel = data_refControl[np.argsort(adata.obs.loc[group_refControl_idx, pair_by_key].cat.codes)]
        # Compute the difference for each pair and run the paired t-test
        target_diff = data_target_rel - data_targetControl_rel
        ref_diff = data_ref_rel - data_refControl_rel
        # Perform vectorized t-tests
        t_rel_nested_stat, t_rel_nested_test_pvals = stats.ttest_rel(target_diff, ref_diff, axis=0)
        # Multiple testing correction
        _, t_rel_nested_test_pvals_corrected, _, _ = multipletests(t_rel_nested_test_pvals[np.isfinite(t_rel_nested_test_pvals)], method="fdr_bh")
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
        results[f'ttest_rel_nested_pvals_corrected{nested_comparison_col_tag}'] = t_rel_nested_test_pvals_corrected_full
        results[f'ttest_rel_nested_mean_paired_fcfc{nested_comparison_col_tag}'] =mean_fc_rel_nested
        results[f'ttest_rel_nested_mean_paired_l2fcfc{nested_comparison_col_tag}'] =mean_log2_fc_rel_nested
    if 'WilcoxonSigned_nested' in tests:
        # first compute the nested difference
        # Boolean indexing to select cells in each group
        group_target_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][0]])
        group_targetControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_target_values[0][1]])
        group_ref_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][0]])
        group_refControl_idx = adata.obs[groupby_key].isin([nested_groupby_key_ref_values[0][1]])
        # Select the data matrix for each group
        data_target=X[group_target_idx].copy()
        data_targetControl=X[group_targetControl_idx].copy()
        data_ref=X[group_ref_idx].copy()
        data_refControl=X[group_refControl_idx].copy()
        # sort the data by the pair_by_key
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        # Sort the data by `pair_by_key`
        data_target_rel = data_target[np.argsort(adata.obs.loc[group_target_idx, pair_by_key].cat.codes)]
        data_targetControl_rel = data_targetControl[np.argsort(adata.obs.loc[group_targetControl_idx, pair_by_key].cat.codes)]
        data_ref_rel = data_ref[np.argsort(adata.obs.loc[group_ref_idx, pair_by_key].cat.codes)]
        data_refControl_rel = data_refControl[np.argsort(adata.obs.loc[group_refControl_idx, pair_by_key].cat.codes)]
        # Compute the difference for each pair and run the paired Wilcoxon rank-sum test
        target_diff = data_target_rel - data_targetControl_rel
        ref_diff = data_ref_rel - data_refControl_rel
        # Perform Wilcoxon rank-sum tests
        w_nested_stat, w_nested_test_pvals = stats.ranksums(target_diff, ref_diff)
        # Multiple testing correction
        _, w_nested_test_pvals_corrected, _, _ = multipletests(w_nested_test_pvals[np.isfinite(w_nested_test_pvals)], method="fdr_bh")
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
        results[f'WilcoxonSigned_nested_pvals_corrected{nested_comparison_col_tag}'] = w_nested_test_pvals_full_corrected
        results[f'WilcoxonSigned_nested_mean_paired_fcfc{nested_comparison_col_tag}'] =mean_fc_rel_nested
        results[f'WilcoxonSigned_nested_mean_paired_l2fcfc{nested_comparison_col_tag}'] =mean_log2_fc_rel_nested
    ### add a shapiro and ks test for normality for the nested difference [(Target-control) - (Ref-control)] data for  within-subjects design with baseline control
    if 'ttest_rel_nested' in tests or 'wilcox_nested' in tests:
        nested_diff = target_diff - ref_diff
        # (Target-control)-(Ref-control) shapiro test # ks test
        #shapiro_stat_nested_diff, shapiro_pvals_nested_diff = stats.shapiro(nested_diff,axis=0)
        #ks_stat_nested_diff, ks_pvals_nested_diff = stats.kstest(nested_diff, 'norm',axis=0)
        shapiro_pvals_nested_diff =[stats.shapiro(nested_diff[:, i]).pvalue for i in range(nested_diff.shape[1])]
        ks_pvals_nested_diff = [
            stats.kstest(
                nested_diff[:, i],
                lambda x, mean=nested_diff[:, i].mean(), std=nested_diff[:, i].std(ddof=1): stats.norm.cdf(x, loc=mean, scale=std)
            ).pvalue
            for i in range(nested_diff.shape[1])
        ]
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
            sortby = f'wilcox_nested_stat{nested_comparison_col_tag}'
        elif f'ttest_rel_stat{comparison_col_tag}' in results.columns:
            sortby = f'ttest_rel_stat{comparison_col_tag}'
        elif f'ttest_ind_stat{comparison_col_tag}' in results.columns:
            sortby = f'ttest_ind_stat{comparison_col_tag}'
        elif f'mannwhitneyu_stat{comparison_col_tag}' in results.columns:
            sortby = f'mannwhitneyu_stat{comparison_col_tag}'

    # Sort the results by absolute value of the selected hypothesis test statistic
    if sortby is not None and sortby in results.columns:
        results.sort_values(sortby, ascending=False, inplace=True, key=lambda x: x.abs(), na_position='last')

    return results