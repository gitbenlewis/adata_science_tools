####################################################################################################################

def average_feature_expression(adata, groupby_key, layer=None, use_raw=False, log1p=False, zscore=False, subtract_mean=True):
    """
    Calculate the average feature expression for observations sharing the same metadata.

    Parameters:
    adata (AnnData): AnnData object containing gene expression data.
    groupby_key (str): Key in adata.obs to group by (e.g., cell type).
    layer (str, optional): Key of the layer in adata to use for the expression data. If None, uses adata.X.
    use_raw (bool, optional): If True, use adata.raw for the expression data. Default is False.
    log1p (bool, optional): If True, apply log1p transformation to the data before averaging. Default is False.
    zscore (bool, optional): If True, apply Z-score scaling to the data before averaging. Default is False.
    subtract_mean (bool, optional): If True, subtract the mean from each feature. Default is False.

    Returns:
    pd.DataFrame: DataFrame with average feature expression, where rows are groups and columns are features (genes).
    """
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler

    # Select the appropriate data matrix
    if use_raw:
        if layer is not None:
            raise ValueError("Cannot specify a layer when use_raw is True")
        data_matrix = adata.raw.X
        var_names = adata.raw.var_names
    else:
        if layer:
            data_matrix = adata.layers[layer]
        else:
            data_matrix = adata.X
        var_names = adata.var_names

    # Apply log1p transformation if specified
    if log1p:
        if sp.issparse(data_matrix):
            data_matrix = data_matrix.log1p()
        else:
            data_matrix = np.log1p(data_matrix)
    # Apply Z-score scaling if specified
    if zscore:
        # Subtract mean if specified
        if subtract_mean:
            if sp.issparse(data_matrix):
                mean = np.array(data_matrix.mean(axis=0)).flatten()
                data_matrix = data_matrix - mean
            else:
                mean = np.mean(data_matrix, axis=0)
                data_matrix = data_matrix - mean
        scaler = StandardScaler(with_mean=not sp.issparse(data_matrix))
        data_matrix = np.asarray(data_matrix)
        data_matrix = scaler.fit_transform(data_matrix)

    # Extract group labels and unique groups
    group_labels = adata.obs[groupby_key]
    unique_groups = adata.obs[groupby_key].cat.categories  # Preserve the order of categories

    # Initialize an empty list to hold the average expressions
    avg_expression_list = []

    # Iterate over each group to calculate the mean expression
    for group in unique_groups:
        group_indices = np.where(group_labels == group)[0]
        group_data = data_matrix[group_indices, :]

        if sp.issparse(group_data):
            group_mean = group_data.mean(axis=0).A1  # Use .A1 to get a flat array from sparse matrix
        else:
            group_mean = np.mean(group_data, axis=0)
        
        # Ensure the group_mean is a flat 1D array
        group_mean = np.asarray(group_mean).flatten()

        # Debugging step: Print the shape of group_mean
        #print(f"Group: {group}, group_mean shape: {group_mean.shape}")

        avg_expression_list.append(group_mean.flatten())

    # Convert the list to a DataFrame
    avg_expression_df = pd.DataFrame(avg_expression_list, index=unique_groups, columns=var_names)

    return avg_expression_df

# Example usage:  sctl.tl.average_feature_expression()
'''
Neuron_subtype_split_avg_expression_df = sctl.tl.average_feature_expression(adata, groupby_key, use_raw=True, log1p=False, zscore=False)
df=Neuron_subtype_split_avg_expression_df[gene_list]
df = df.reindex(columns=gene_list)
display(df)

figsize=(7,10)
fig1, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize)
#df.plot.barh(stacked=False,ax=axes).legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), borderaxespad=0)

df.plot.barh(stacked=False,ax=axes)
# Customize legend and axes
handles, labels = axes.get_legend_handles_labels()
legend_mapping = {label: handle for label, handle in zip(labels, handles)}
axes.legend([legend_mapping[gene] for gene in gene_list], gene_list, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

axes.invert_yaxis()
# Add axis labels
axes.set_xlabel('CP10K')
axes.set_ylabel(groupby_key)  # Add meaningful y-axis label (optional)
plt.tight_layout()
plt.show()
'''