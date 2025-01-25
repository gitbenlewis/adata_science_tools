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

############ AnnData object version ############
def make_df_obs_adataX(adata,layer=None,index=None,varcolumns=None,include_obs=True):
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