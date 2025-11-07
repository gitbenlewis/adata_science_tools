# functions for working with AnnData objects with somalogic data
# module level package imports
from anndata import AnnData
from typing import Dict, List, TextIO, Tuple, Union
import numpy as np


def read_adat_2_AnnData(path_or_buf: Union[str, TextIO]) -> AnnData:
    """Returns an adata object instead a somalogic adat file path from the filepath/name.
    Parameters
    ----------
    path_or_buf : Union[str, TextIO]
        Path or buffer that the file will be read from
    Examples
    --------
    >>> adata=read_adat_2_AnnData(example_data_file)
    Returns
    -------
    adata object instead of  : Adat object
    """
    # functions for working with AnnData objects with somalogic data
    import somadata
    import csv
    import json
    import logging
    import re
    import warnings
    from importlib.metadata import version
    from typing import Dict, List, TextIO, Tuple, Union
    from anndata import AnnData
    import anndata as ad
    import numpy as np
    import pandas as pd
    
    if type(path_or_buf) == str:
        with open(path_or_buf, 'r') as f:
            rfu_matrix, row_metadata, column_metadata, header_metadata = somadata.io.adat.file.parse_file(f)
    else:
        rfu_matrix, row_metadata, column_metadata, header_metadata = somadata.io.adat.file.parse_file(
            path_or_buf
        )
    rfu_matrix_array = np.array(rfu_matrix)

    adata = ad.AnnData(X=rfu_matrix_array)
    adata.obs = pd.DataFrame(row_metadata)
    adata.var = pd.DataFrame(column_metadata)
    adata.uns = header_metadata
    # make sure that all uns are strings to avoid issues with writing to h5ad files
    for key, value in adata.uns.items():
        if not isinstance(value, str):
            adata.uns[key] = str(value)
    return adata


'''
# if source code changed and above function added to somadata package then can use adata = somadata.read_adat_2_AnnData('./somadata/data/example_data.adat')
# need to load code in cell above
example_data_file='./somadata/data/example_data.adat'
adata=read_adat_2_AnnData(example_data_file)
print('loaded adata info \n',adata)

# set the obs and var names
adata.obs_names = adata.obs['SampleId']
adata.var_names = adata.var['SeqId']

# better if they are unique, check if unique
print('adata.obs_names are unique',adata.obs_names.is_unique)
print('adata.var_names are unique',adata.var_names.is_unique)
# show the non unique values in adata.obs_names and adata.var_names
print('duplicated adata.obs_names \n',adata.obs_names[adata.obs_names.duplicated()])
print('duplicated adata.var_names \n',adata.var_names[adata.var_names.duplicated()])

# make them unique
adata.obs_names_make_unique()
adata.var_names_make_unique()
# rename the index to SampleId_unique
adata.obs.index.rename('SampleId_unique', inplace=True)
print('adata.obs_names are unique',adata.obs_names.is_unique)
print('adata.var_names are unique',adata.var_names.is_unique)

adata
'''




def set_sampletype_obs_values(adata: AnnData,
                              donor_obs_column: str = 'SampleType',
                              donor_obs_col_values_to_paste: list[str] | None = None,
                              obs_columns_toFix: list[str] | None = None,
                              make_copy: bool = False
                              ):
    '''paste the value in the 'donor_obs_column' column to other obs columns 
    where the donor_obs_column value is in donor_obs_col_values_to_paste list
    Parameters
    ----------
    adata : AnnData object
        The AnnData object to modify.
    donor_obs_column : str, optional
        The name of the obs column to use as the donor column, by default 'SampleType'
    donor_obs_col_values_to_paste : list, optional
        The list of values in the donor_obs_column to paste to other columns, by default ['QC', 'Buffer', 'Calibrator']
    obs_columns_toFix : list, optional
        The list of obs columns to paste the donor values to, by default ['obs_col_1', 'obs_col_2', 'obs_col_3']
    make_copy : bool, optional
        Whether to make a copy of the adata object before modifying it, by default False
    Returns
    -------
    AnnData object
        The modified AnnData object.
    makes a copy if make_copy is True
    '''
# #) make copy if requested
    if make_copy:
        _adata = adata.copy()
    else:
        _adata = adata
# #) keep only columns_toFix are present in adata.obs
    filtered_obs_columns_toFix = [col for col in obs_columns_toFix if col in _adata.obs.columns]
    if obs_columns_toFix != filtered_obs_columns_toFix:
        print(f'Note: some obs columns to fix not present in adata.obs: {set(obs_columns_toFix) - set(filtered_obs_columns_toFix)}')

# #) mask select rows and paste donor values to obs columns to fix
    mask = _adata.obs[donor_obs_column].isin(donor_obs_col_values_to_paste).fillna(False)     # ; fillna(False) avoids NaN==True
    # Assign donor value into each target column for masked rows
    # Broadcasts the Series across the selected columns
    _adata.obs.loc[mask, filtered_obs_columns_toFix] = _adata.obs.loc[mask, donor_obs_column].values[:, None]



# example usage
#adata=adata_from_adat_somafile.copy()
#set_sampletype_obs_values(
#    adata,
#    donor_obs_column='SampleType',
#    donor_obs_col_values_to_paste=['QC', 'Buffer', 'Calibrator'],
#    obs_columns_toFix = ['AliquotingNotes', 'AssayNotes', 'TimePoint'],
#    make_copy=False
#)
#display(adata.obs.tail(20))




############ AnnData object version ############
def make_df_obs_adataX_soma(adata,layer=None,index=None,varcolumns=None,include_obs=True):
    """
    Parameters

    """
    import pandas as pd
    # functions for working with AnnData objects with somalogic data
    #import somadata
    #import csv
    #import json
    #import logging
    #import re
    #import warnings
    #from importlib.metadata import version
    #import csv
    from anndata import AnnData

    # Set up feature (variable) columns
    if varcolumns is None:
        varcolumns = adata.var_names
    elif isinstance(varcolumns, str):
        varcolumns = adata.var[varcolumns]
    elif isinstance(varcolumns, list):
        if len(varcolumns)==1:
            varcolumns = adata.var[varcolumns[0]]
        else:
            varcolumns = adata.var[varcolumns]
            varcolumns = pd.MultiIndex.from_arrays(varcolumns.values.T, names=varcolumns.columns)  
    # Set up the index
    index=adata.obs_names if index is None else adata.obs[index]
    if layer is None:
        df_adataX=pd.DataFrame(adata.X,columns=varcolumns,index =index  
                               )
    else:
        df_adataX=pd.DataFrame(adata.layers[layer],columns=varcolumns,index =index 
                               )
    if include_obs:
        df_obs=adata.obs
        df_obs_adataX= pd.concat([df_obs,df_adataX], axis=1)
        return df_obs_adataX
    return df_adataX



