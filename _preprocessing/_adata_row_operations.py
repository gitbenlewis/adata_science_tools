# _adata_row_operations.py

# module at adata_science_tools/_preprocessing/_adata_row_operations.py

# module imports
import pandas as pd
import numpy as np
import anndata as ad


def compute_paired_mean_adata(
    adata,
    layer='RFU',
    pair_by_key='AnimalID_Tattoo',
    groupby_key='Treatment_unique',
    datapoint_1='drug78hr',
    datapoint_2='drug30hr',
    debug_mode=True
):
    """
    Compute the average of two selected groups in an AnnData object and return a new AnnData object.
    
    Parameters:
    - adata: AnnData object
    - layer: The layer to use (default: 'RFU'). If None, uses .X
    - pair_by_key: The key in .obs to pair the data
    - groupby_key: The key in .obs used to select groups
    - datapoint_1: First group to compute the mean
    - datapoint_2: Second group to compute the mean
    - debug_mode: If True, prints debug information
    
    Returns:
    - adata_mean_datapoint_12: AnnData object with averaged values
    - mean_datapoint_12_df: DataFrame of averaged values
    """
    _adata = adata.copy()
    
    X = _adata.layers[layer] if layer else _adata.X
    
    # Boolean indexing to select cells in each group
    datapoint_1_idx = _adata.obs[groupby_key].isin([datapoint_1])
    datapoint_2_idx = _adata.obs[groupby_key].isin([datapoint_2])
    
    # Select the data matrix for each group
    data_datapoint_1 = X[datapoint_1_idx, :].copy()
    data_datapoint_2 = X[datapoint_2_idx, :].copy()
    
    if debug_mode:
        display('data_datapoint_1:', data_datapoint_1.shape, data_datapoint_1[:11,:2])
        display('data_datapoint_2:', data_datapoint_2.shape, data_datapoint_2[:11,:2])
    
    # Ensure `pair_by_key` is categorical for proper sorting
    if not isinstance(_adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
        _adata.obs[pair_by_key] = _adata.obs[pair_by_key].astype('category')
    
    # Sort the data by `pair_by_key`
    data_datapoint_1_rel = data_datapoint_1[np.argsort(_adata.obs.loc[datapoint_1_idx, pair_by_key].cat.codes)]
    data_datapoint_2_rel = data_datapoint_2[np.argsort(_adata.obs.loc[datapoint_2_idx, pair_by_key].cat.codes)]
    
    if debug_mode:
        display('data_datapoint_1_rel:', data_datapoint_1_rel.shape, data_datapoint_1_rel[:11,:2])
        display('data_datapoint_2_rel:', data_datapoint_2_rel.shape, data_datapoint_2_rel[:11,:2])
    
    # Compute the mean
    mean_datapoint_12 = (data_datapoint_1_rel + data_datapoint_2_rel) / 2
    
    if debug_mode:
        display('mean_datapoint_12:', mean_datapoint_12.shape, mean_datapoint_12[:11,:2])
    
    # Convert to DataFrame
    mean_datapoint_12_df = pd.DataFrame(mean_datapoint_12, columns=_adata.var.index)
    mean_datapoint_12_df['obs_names1'] = _adata[datapoint_1_idx].obs_names.to_numpy()[
        np.argsort(_adata.obs.loc[datapoint_1_idx, pair_by_key].cat.codes)
    ]
    mean_datapoint_12_df['obs_names2'] = _adata[datapoint_2_idx].obs_names.to_numpy()[
        np.argsort(_adata.obs.loc[datapoint_2_idx, pair_by_key].cat.codes)
    ]
    mean_datapoint_12_df['obs_names'] = 'avg_' + mean_datapoint_12_df['obs_names1'] + '_' + mean_datapoint_12_df['obs_names2']
    mean_datapoint_12_df.set_index('obs_names', inplace=True)
    mean_datapoint_12_df.drop(columns=['obs_names1', 'obs_names2'], inplace=True)
    
    if debug_mode:
        display(mean_datapoint_12_df.head(5))
    
    # Create new AnnData object
    adata_mean_datapoint_12 = ad.AnnData(X=mean_datapoint_12_df)
    adata_mean_datapoint_12.var_names = _adata.var_names
    adata_mean_datapoint_12.var = _adata.var.copy()
    
    if debug_mode:
        display(adata_mean_datapoint_12[:11,:2])
    
    return adata_mean_datapoint_12, mean_datapoint_12_df


def compute_paired_difference_adata(
    adata,
    layer='RFU',
    pair_by_key='AnimalID_Tattoo',
    groupby_key='Treatment_unique',
    datapoint_1='drug78hr',
    datapoint_2='drug30hr',
    debug_mode=True
):
    """
    Compute the paired difference (datapoint_1 - datapoint_2) in an AnnData object and return a new AnnData object.
    
    Parameters:
    - adata: AnnData object
    - layer: The layer to use (default: 'RFU'). If None, uses .X
    - pair_by_key: The key in .obs to pair the data
    - groupby_key: The key in .obs used to select groups
    - datapoint_1: Group treated as minuend
    - datapoint_2: Group treated as subtrahend
    - debug_mode: If True, prints debug information
    
    Returns:
    - adata_diff_datapoint_12: AnnData object with difference values
    - diff_datapoint_12_df: DataFrame of difference values
    """
    _adata = adata.copy()
    
    X = _adata.layers[layer] if layer else _adata.X
    
    datapoint_1_idx = _adata.obs[groupby_key].isin([datapoint_1])
    datapoint_2_idx = _adata.obs[groupby_key].isin([datapoint_2])
    
    data_datapoint_1 = X[datapoint_1_idx, :].copy()
    data_datapoint_2 = X[datapoint_2_idx, :].copy()
    
    if debug_mode:
        display('data_datapoint_1:', data_datapoint_1.shape, data_datapoint_1[:11,:2])
        display('data_datapoint_2:', data_datapoint_2.shape, data_datapoint_2[:11,:2])
    
    if not isinstance(_adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
        _adata.obs[pair_by_key] = _adata.obs[pair_by_key].astype('category')
    
    data_datapoint_1_rel = data_datapoint_1[np.argsort(_adata.obs.loc[datapoint_1_idx, pair_by_key].cat.codes)]
    data_datapoint_2_rel = data_datapoint_2[np.argsort(_adata.obs.loc[datapoint_2_idx, pair_by_key].cat.codes)]
    
    if debug_mode:
        display('data_datapoint_1_rel:', data_datapoint_1_rel.shape, data_datapoint_1_rel[:11,:2])
        display('data_datapoint_2_rel:', data_datapoint_2_rel.shape, data_datapoint_2_rel[:11,:2])
    
    diff_datapoint_12 = data_datapoint_1_rel - data_datapoint_2_rel
    
    if debug_mode:
        display('diff_datapoint_12:', diff_datapoint_12.shape, diff_datapoint_12[:11,:2])
    
    diff_datapoint_12_df = pd.DataFrame(diff_datapoint_12, columns=_adata.var.index)
    diff_datapoint_12_df['obs_names1'] = _adata[datapoint_1_idx].obs_names.to_numpy()[
        np.argsort(_adata.obs.loc[datapoint_1_idx, pair_by_key].cat.codes)
    ]
    diff_datapoint_12_df['obs_names2'] = _adata[datapoint_2_idx].obs_names.to_numpy()[
        np.argsort(_adata.obs.loc[datapoint_2_idx, pair_by_key].cat.codes)
    ]
    diff_datapoint_12_df['obs_names'] = 'diff_' + diff_datapoint_12_df['obs_names1'] + '_' + diff_datapoint_12_df['obs_names2']
    diff_datapoint_12_df.set_index('obs_names', inplace=True)
    diff_datapoint_12_df.drop(columns=['obs_names1', 'obs_names2'], inplace=True)
    
    if debug_mode:
        display(diff_datapoint_12_df.head(5))
    
    adata_diff_datapoint_12 = ad.AnnData(X=diff_datapoint_12_df)
    adata_diff_datapoint_12.var_names = _adata.var_names
    adata_diff_datapoint_12.var = _adata.var.copy()
    
    if debug_mode:
        display(adata_diff_datapoint_12[:11,:2])
    
    return adata_diff_datapoint_12, diff_datapoint_12_df

'''_adata = adata.copy()
adata_mean_datapoint_12, mean_datapoint_12_df = compute_mean_adata(
    adata=_adata,
    layer='Volume_norm',
    pair_by_key='AnimalID_Tattoo',
    groupby_key='Treatment_unique',
    datapoint_1='drug78hr',
    datapoint_2='drug30hr',
    debug_mode=False
)
display(adata_mean_datapoint_12[:11,:2], adata_mean_datapoint_12.obs.head(5))
display(mean_datapoint_12_df.head(5))'''
