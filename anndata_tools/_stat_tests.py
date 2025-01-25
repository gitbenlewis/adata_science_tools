
def diff_exp_soma(adata, layer=None, groupby_key=None, groupby_key_target_values=[None], groupby_key_ref_values=[None],comparison_col_tag='_target_ref',
                      nested_groupby_key_target_values=[(None,None)], nested_groupby_key_ref_values= [(None,None)],   nested_comparison_col_tag='_target_con_ref_con',
                       sortby=None,tests=['ttest_ind', 'ttest_rel','mannwhitneyu', 'wilcox','ttest_rel_nested','wilcox_nested'],pair_by_key=None ):
    """
    Perform differential expression analysis on a somalogic dataset.
    Parameters
    ----------
    adata : AnnData object
    groupby : str
        The key of the observation grouping to consider.
    group1 : str
        The name of the first group.
    group2 : str
        The name of the second group.
    layer : str, optional (default: None)
        The key of the layer to use. If not specified, defaults to adata.X.

    Returns
    -------
    A dataframe with the gene names, t-test p-value, t-test log fold change, and corrected p-value for each gene.
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
    X = adata.layers[layer] if layer else adata.X
    if hasattr(X, "toarray"):  # Convert sparse matrix to dense if necessary
        X = X.toarray()
    # Remove genes (columns) with zero expression across all cells
    non_zero_genes = ~np.all(X == 0, axis=0)
    X = X[:, non_zero_genes]
    var_names = adata.var_names[non_zero_genes]

    ### Initialize results DataFrame
    results = pd.DataFrame({"var_names": var_names, }, index=var_names)

    if groupby_key_target_values:
        # Boolean indexing to select cells in each group
        group1_idx = adata.obs[groupby_key].isin(groupby_key_target_values)
        group2_idx = adata.obs[groupby_key].isin(groupby_key_ref_values)

        data1 = X[group1_idx].copy()
        data2 = X[group2_idx].copy()

        # Compute mean expressions for each group (vectorized)
        mean_data1 = np.mean(data1, axis=0)
        mean_data2 = np.mean(data2, axis=0)

        # Compute log fold change with vectorized operation and handle zeros
        epsilon = 1e-9  # Small constant to avoid log issues
        valid_indices = (mean_data1 > 0) & (mean_data2 > 0)
        logfc = np.empty(mean_data1.shape)
        logfc.fill(np.nan)  # Initialize with NaN
        logfc[valid_indices] = np.log2((mean_data1[valid_indices] + epsilon) / (mean_data2[valid_indices] + epsilon))

        # Add means and l2fc to the results DataFrame

        # make labels group1 and group2 labels
        _groupby_key_target_values_str='_'.join(groupby_key_target_values)
        _groupby_key_ref_values_str='_'.join(groupby_key_ref_values)
        group1_label = f"{groupby_key}_{_groupby_key_target_values_str}"
        group2_label = f"{groupby_key}_{_groupby_key_ref_values_str}"
        # make new columns in the results dataframe
        results[f'{group1_label}:mean'] =mean_data1
        results[f'{group2_label}:mean'] =mean_data2
        results[f'l2fc{comparison_col_tag}'] =logfc




    ### Perform statistical tests
    if 'ttest_ind' in tests:
        # Perform vectorized t-tests
        t_stat, t_test_pvals = stats.ttest_ind(data1, data2, equal_var=False, axis=0)
        # Multiple testing correction
        _, t_test_pvals_corrected, _, _ = multipletests(t_test_pvals, method="fdr_bh")
        # Add p-values to the results DataFrame
        results[f'ttest_ind_pvals{comparison_col_tag}'] = t_test_pvals
        results[f'ttest_ind_pvals_corrected{comparison_col_tag}'] = t_test_pvals_corrected
    if 'ttest_rel' in tests:
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        # sort the data by the pair_by_key
        data1_rel = data1[np.argsort(adata.obs.loc[group1_idx,pair_by_key].cat.codes)]
        data2_rel = data2[np.argsort(adata.obs.loc[group2_idx,pair_by_key].cat.codes)]
        # Perform vectorized t-tests
        t_rel_stat, t_rel_test_pvals = stats.ttest_rel(data1_rel, data2_rel, axis=0)
        # Multiple testing correction
        _, t_rel_test_pvals_corrected, _, _ = multipletests(t_rel_test_pvals, method="fdr_bh")
        # calculate mean log2 fold change for the paired data
        mean_log2_fc_rel = np.mean(np.log2((data1_rel + 1e-9) / (data2_rel + 1e-9)), axis=0)
        # Add p-values to the results DataFrame
        results[f'ttest_rel_pvals{comparison_col_tag}'] = t_rel_test_pvals
        results[f'ttest_rel_pvals_corrected{comparison_col_tag}'] = t_rel_test_pvals_corrected
        results[f'ttest_rel_mean_paired_l2fc{comparison_col_tag}'] =mean_log2_fc_rel
    if 'mannwhitneyu' in tests:
        # Perform Mann-Whitney U tests with continuity correction
        u_test_pvals = [
            stats.mannwhitneyu(data1[:, i], data2[:, i], alternative='two-sided',).pvalue
            for i in range(data1.shape[1])]
        # Multiple testing correction
        _, u_test_pvals_corrected, _, _ = multipletests(u_test_pvals, method="fdr_bh")
        # Add p-values to the results DataFrame
        results[f'mannwhitneyu_pvals{comparison_col_tag}'] = u_test_pvals
        results[f'mannwhitneyu_pvals_corrected{comparison_col_tag}'] = u_test_pvals_corrected
    if 'wilcox' in tests:
        # Ensure `pair_by_key` is categorical for proper sorting
        if not isinstance(adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
            adata.obs[pair_by_key] = adata.obs[pair_by_key].astype('category')
        # sort the data by the pair_by_key
        data1_rel = data1[np.argsort(adata.obs.loc[group1_idx,pair_by_key].cat.codes)]
        data2_rel = data2[np.argsort(adata.obs.loc[group2_idx,pair_by_key].cat.codes)]
        # Perform Wilcoxon rank-sum tests
        w_stat, w_test_pvals = stats.ranksums(data1_rel, data2_rel)
        # Multiple testing correction
        _, w_test_pvals_corrected, _, _ = multipletests(w_test_pvals, method="fdr_bh")
        # calculate mean log2 fold change for the paired data
        mean_log2_fc_rel = np.mean(np.log2((data1_rel + 1e-9) / (data2_rel + 1e-9)), axis=0)
        # Add p-values to the results DataFrame
        results[f'wilcox_pvals{comparison_col_tag}'] = w_test_pvals
        results[f'wilcox_pvals_corrected{comparison_col_tag}'] = w_test_pvals_corrected
        results[f'wilcox_mean_paired_l2fc{comparison_col_tag}'] =mean_log2_fc_rel
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
        _, t_rel_nested_test_pvals_corrected, _, _ = multipletests(t_rel_nested_test_pvals, method="fdr_bh")
        # calculate mean log2 fold change for the nested paired test data
        mean_log2_fc_rel_nested = np.mean(np.log2((
            ((data_target_rel + 1e-9) / (data_targetControl_rel + 1e-9))/((data_ref_rel + 1e-9) / (data_refControl_rel + 1e-9)))
            ), axis=0)
        # Add p-values to the results DataFrame
        results[f'ttest_rel_nested_pvals{nested_comparison_col_tag}'] = t_rel_nested_test_pvals
        results[f'ttest_rel_nested_pvals_corrected{nested_comparison_col_tag}'] = t_rel_nested_test_pvals_corrected
        results[f'ttest_rel_nested_mean_paired_l2fc{nested_comparison_col_tag}'] =mean_log2_fc_rel_nested
    if 'wilcox_nested' in tests:
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
        _, w_nested_test_pvals_corrected, _, _ = multipletests(w_nested_test_pvals, method="fdr_bh")
        # calculate mean log2 fold change for the nested paired test data
        mean_log2_fc_rel_nested = np.mean(np.log2((
            ((data_target_rel + 1e-9) / (data_targetControl_rel + 1e-9))/((data_ref_rel + 1e-9) / (data_refControl_rel + 1e-9)))
            ), axis=0)
        # Add p-values to the results DataFrame
        results[f'wilcox_nested_pvals{nested_comparison_col_tag}'] = w_nested_test_pvals
        results[f'wilcox_nested_pvals_corrected{nested_comparison_col_tag}'] = w_nested_test_pvals_corrected
        results[f'wilcox_nested_mean_paired_l2fc{nested_comparison_col_tag}'] =mean_log2_fc_rel_nested
        




    ### Sort the results dataframe sortby
    # assign a column to sortby if not provided
    if sortby is None:
        if f'ttest_ind_pvals{comparison_col_tag}' in results.columns:
            sortby = f'ttest_ind_pvals{comparison_col_tag}'
        elif f'ttest_rel_pvals{comparison_col_tag}' in results.columns:
            sortby = f'ttest_rel_pvals{comparison_col_tag}'
        elif f'mannwhitneyu_pvals{comparison_col_tag}' in results.columns:
            sortby = f'mannwhitneyu_pvals{comparison_col_tag}'
    # Sort the results by corrected p-value
    if sortby in results.columns:
        results = results.sort_values(sortby)

    return results