# functions for working with AnnData objects with somalogic data
# module level package imports
from anndata import AnnData
from typing import Dict, List, TextIO, Tuple, Union
import logging
import numpy as np
import pandas as pd
import pandas as pd


LOGGER = logging.getLogger(__name__)


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




def soma_fill_sampletype_obs_values(
    adata: AnnData,
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
    donor_obs_col_values_to_paste = donor_obs_col_values_to_paste or ['QC', 'Buffer', 'Calibrator']
    obs_columns_toFix = list(obs_columns_toFix or [])

    _adata = adata.copy() if make_copy else adata


    if donor_obs_column not in _adata.obs.columns:
        raise KeyError(f'donor_obs_column "{donor_obs_column}" is not present in adata.obs')

    if not obs_columns_toFix:
        print('Note: no obs columns were provided to update; returning the original AnnData.')
        return _adata

    filtered_obs_columns_toFix = [col for col in obs_columns_toFix if col in _adata.obs.columns]
    if obs_columns_toFix != filtered_obs_columns_toFix:
        missing = set(obs_columns_toFix) - set(filtered_obs_columns_toFix)
        print(f'Note: some obs columns to fix not present in adata.obs: {missing}')

    if not filtered_obs_columns_toFix:
        print('Note: none of the requested obs columns exist in adata.obs; nothing to update.')
        return _adata

    donor_series = _adata.obs[donor_obs_column]
    mask = donor_series.isin(donor_obs_col_values_to_paste).fillna(False)  # fillna avoids NaN==True

    if not mask.any():
        print('Note: no rows matched the provided donor_obs_col_values_to_paste; nothing to update.')
        return _adata

    # Broadcast donor values across each target column
    donor_values = donor_series.loc[mask].to_numpy()
    broadcast_values = np.tile(donor_values[:, None], (1, len(filtered_obs_columns_toFix)))
    _adata.obs.loc[mask, filtered_obs_columns_toFix] = broadcast_values

    return _adata



# example usage
#adata=adata_from_adat_somafile.copy()
#soma_fill_sampletype_obs_values(
#    adata,
#    donor_obs_column='SampleType',
#    donor_obs_col_values_to_paste=['QC', 'Buffer', 'Calibrator'],
#    obs_columns_toFix = ['AliquotingNotes', 'AssayNotes', 'TimePoint'],
#    make_copy=False
#)
#display(adata.obs.tail(20))


from anndata import AnnData
# make non user provided samples have unique 'SampleId' s
def soma_make_adata_index_unique_by_merge(
    adata: AnnData,
    donor_obs_column: str = 'Barcode2d',
    mask: pd.Series | None = None,
    duplicates_index_only: bool = True,
    ensure_global_unique: bool = False,
    make_copy: bool = False,
) -> AnnData:
    """Make adata.obs_names unique by merging with another obs column for masked rows.

    Args:
        adata (AnnData): Input AnnData object.
        donor_obs_column (str, optional): Column in adata.obs to merge with. Defaults to 'Barcode2d'.
        mask (pd.Series | None, optional): Boolean mask to select rows to modify. Defaults to None.
        duplicates_index_only (bool, optional): Whether to only modify rows with duplicate obs_names. Defaults to True.
        ensure_global_unique (bool, optional): Whether to ensure all obs_names are globally unique. Defaults to False.
        make_copy (bool, optional): Whether to make a copy of the AnnData object. Defaults to False.

    Returns:
        AnnData: Modified AnnData object with unique obs_names.
    """
    _adata = adata.copy() if make_copy else adata

    
    if mask is None:
        mask = pd.Series([True] * _adata.n_obs, index=_adata.obs_names)

    # Create new obs names by merging with donor_obs_column for masked rows

    if duplicates_index_only:
        dup_mask = pd.Index(_adata.obs_names).duplicated(keep=False)
        mask = mask & dup_mask

    # Base: current index as strings
    idx = pd.Series(_adata.obs_names.astype(str), index=_adata.obs_names)
    # New names for masked rows: "<old_index>_<donor_value>"
    donor = _adata.obs.loc[mask, donor_obs_column].astype(str)
    new_idx = idx.copy()
    new_idx.loc[mask] = idx.loc[mask] + "_" + donor

    _adata.obs_names = new_idx

    if ensure_global_unique:
        _adata.obs_names_make_unique()

    return _adata

# example usage
#adata=adtl.adata_from_adat_somafile.copy()
#set_sampletype_obs_values(
#    adata,
#    donor_obs_column='Barcode2d',
#    mask=adata.obs['SampleType'].isin(['QC', 'Buffer', 'Calibrator']),
#    duplicates_index_only=True,
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



import anndata as ad
import pandas as pd
import anndata as ad
from pathlib import Path
def somascan_adat_file_2_adata_h5ad_csv(
    save_raw_h5ad: bool = False,
    also_raw_save_csvs: bool = False,
    merge_external_metadata: bool = False,
    save_plus_metadata_h5ad: bool = False,
    also_plus_metadata_save_csvs: bool = False,
    output_dir: str | Path | None = None,
    dataset_alias: str | None = None,
    raw_somascan_adat_data_file: str | None = None,
    new_index_key_from_raw_obs_metadata: str | None = None,
    remove_mouse_QC_apatmers: bool = False,
    raw_output_filename: str | None = None,
    external_obs_metadata_2_merge_file: str | None = None,
    merge_key_in_external_obs_metadata: str | None = None,
    merge_key_in_raw_obs_metadata: str | None = None,
    column_in_metadata_to_set_as_index: str | None = None,
    external_var_metadata_2_merge_file: str | None = None,
    merge_key_in_external_var_metadata: str | None = None,
    merge_key_in_raw_var_metadata: str | None = None,
    columns_in_external_var_metadata_to_use: list | None = None,
    plus_metadata_file_name: str | None = None,
    remove_mouse_QC_apatmers_QC_samples: bool = True,
    logger: logging.Logger | None = None,
) -> ad.AnnData | tuple[ad.AnnData, ad.AnnData]:
    logger = logger or LOGGER

    def _save_dataset(_adata, output_path):
        """Save dataset helper function."""
        h5ad_path = Path(f"{output_path}.h5ad")
        obs_csv_path = Path(f"{output_path}.obs.csv")
        var_csv_path = Path(f"{output_path}.var.csv")
        X_csv_path = Path(f"{output_path}.X.csv")
        logger.info(f"Saving adata to {h5ad_path}")
        _adata.write_h5ad(h5ad_path)
        logger.info(f"Saving adata.obs to {obs_csv_path}")
        _adata.obs.to_csv(obs_csv_path)
        logger.info(f"Saving adata.var to {var_csv_path}")
        _adata.var.to_csv(var_csv_path)
        logger.info(f"Saving adata.X to {X_csv_path}")
        pd.DataFrame(_adata.X, index=_adata.obs_names, columns=_adata.var_names).to_csv(X_csv_path)
        output_dir = h5ad_path.parent
        logger.info(f"Saved dataset to directory: {output_dir}")
        basename = h5ad_path.stem
        logger.info(f"Saved dataset with base filename: {basename}")

    logger.info(f"save_raw_h5ad: {save_raw_h5ad}")
    logger.info(f"also_raw_save_csvs: {also_raw_save_csvs}")
    logger.info(f"merge_external_metadata: {merge_external_metadata}")
    logger.info(f"save_plus_metadata_h5ad: {save_plus_metadata_h5ad}")
    logger.info(f"also_plus_metadata_save_csvs: {also_plus_metadata_save_csvs}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"dataset_alias: {dataset_alias}")
    logger.info(f"raw_somascan_adat_data_file: {raw_somascan_adat_data_file}")
    logger.info(f"new_index_key_from_raw_obs_metadata: {new_index_key_from_raw_obs_metadata}")
    logger.info(f"remove_mouse_QC_apatmers: {remove_mouse_QC_apatmers}")
    logger.info(f"raw_output_filename: {raw_output_filename}")
    logger.info(f"external_obs_metadata_2_merge_file: {external_obs_metadata_2_merge_file}")
    logger.info(f"merge_key_in_external_obs_metadata: {merge_key_in_external_obs_metadata}")
    logger.info(f"merge_key_in_raw_obs_metadata: {merge_key_in_raw_obs_metadata}")
    logger.info(f"column_in_metadata_to_set_as_index: {column_in_metadata_to_set_as_index}")
    logger.info(f"external_var_metadata_2_merge_file: {external_var_metadata_2_merge_file}")
    logger.info(f"merge_key_in_external_var_metadata: {merge_key_in_external_var_metadata}")
    logger.info(f"merge_key_in_raw_var_metadata: {merge_key_in_raw_var_metadata}")
    logger.info(f"columns_in_external_var_metadata_to_use: {columns_in_external_var_metadata_to_use}")
    logger.info(f"plus_metadata_file_name: {plus_metadata_file_name}")
    logger.info(f"remove_mouse_QC_apatmers_QC_samples: {remove_mouse_QC_apatmers_QC_samples}")
    ## a) load the somascan adat file into anndata
    logger.info(f"Loading somascan adat file: {raw_somascan_adat_data_file}")
    adata_raw_from_adat=read_adat_2_AnnData(raw_somascan_adat_data_file)
    ### general qc here
    logger.info(f'loaded adata info \n{adata_raw_from_adat}')
    # better if they are unique, check if unique
    logger.info(f'adata.obs_names are unique {adata_raw_from_adat.obs_names.is_unique}')
    logger.info(f'adata.var_names are unique {adata_raw_from_adat.var_names.is_unique}')
    # show the non unique values in adata.obs_names and adata.var_names
    logger.info(f'duplicated adata.obs_names \n{adata_raw_from_adat.obs_names[adata_raw_from_adat.obs_names.duplicated()]}')
    logger.info(f'duplicated adata.var_names \n{adata_raw_from_adat.var_names[adata_raw_from_adat.var_names.duplicated()]}')
    logger.info(f' info \n')
    logger.info(f'adata_raw_from_adat.var.head(5)\n{adata_raw_from_adat.var.head(5)}')
    logger.info(f'adata_raw_from_adat.obs.head(5)\n{adata_raw_from_adat.obs.head(5)}')
    logger.info(f'adata_raw_from_adat.obs.tail(5)\n{adata_raw_from_adat.obs.tail(5)}')
    ## b ) make changes to adata object
    ############ change the obs names / obs index to the merge_key_in_raw_obs_metadata
    adata_raw_from_adat.obs_names = adata_raw_from_adat.obs[new_index_key_from_raw_obs_metadata]
    adata_raw_from_adat.obs_names.name = new_index_key_from_raw_obs_metadata+'_index'

    # make the obs index unique by mergeing with another column if needed
    adata_raw_from_adat=soma_make_adata_index_unique_by_merge(
    adata_raw_from_adat,
    donor_obs_column='Barcode2d',
    mask=adata_raw_from_adat.obs['SampleType'].isin(['QC', 'Buffer', 'Calibrator']),
    duplicates_index_only=False,
    #duplicates_index_only=True,
    ensure_global_unique=True,
    make_copy=False
    )
    soma_fill_sampletype_obs_values(
    adata_raw_from_adat,
    donor_obs_column='SampleType',
    donor_obs_col_values_to_paste=['QC', 'Buffer', 'Calibrator'],
    obs_columns_toFix = ['AliquotingNotes', 'AssayNotes', 'TimePoint'],
    make_copy=False
    )
    # copy new unieque index to a new column in obs
    adata_raw_from_adat.obs[new_index_key_from_raw_obs_metadata+'_unique']=adata_raw_from_adat.obs_names
    # maybe skip this step
    #adata_raw_from_adat.obs['Human_sample']=adata_raw_from_adat.obs['SampleId']
    #adata_raw_from_adat.obs['merge_key_in_external_obs_metadata']=adata_raw_from_adat.obs[merge_key_in_raw_obs_metadata]
    
    ############ change the var names / var index
    # make combo var column
    adata_raw_from_adat.var['SeqIdEntrezGeneSymbol']=adata_raw_from_adat.var['SeqId']+' | '+adata_raw_from_adat.var['EntrezGeneSymbol']
    adata_raw_from_adat.var['EntrezGeneSymbol_SeqId'] = adata_raw_from_adat.var['EntrezGeneSymbol'] + '_' + adata_raw_from_adat.var['SeqId']
    # ) select new index from the make combo  var columns
    adata_raw_from_adat.var_names = adata_raw_from_adat.var['EntrezGeneSymbol_SeqId']
    # Preserve the historical CSV header expected by downstream repos.
    adata_raw_from_adat.var_names.name = 'var_names'
    # remove_mouse_QC_apatmers
    if remove_mouse_QC_apatmers:
        logger.info(f"Removing mouse and QC aptamers from adata.var")
        initial_var_count = adata_raw_from_adat.n_vars
        adata_raw_from_adat=adata_raw_from_adat[:,
         (adata_raw_from_adat.var['Type'].str.contains('Protein')
          &(~adata_raw_from_adat.var['Organism'].str.contains('Mouse')))].copy()
        final_var_count = adata_raw_from_adat.n_vars
        logger.info(f"Removed {initial_var_count - final_var_count} aptamers. Remaining aptamers: {final_var_count}")
    logger.info(f'final adata_raw_from_adat info \n{adata_raw_from_adat}')
    logger.info(f'adata.obs_names are unique {adata_raw_from_adat.obs_names.is_unique}')
    logger.info(f'adata.var_names are unique {adata_raw_from_adat.var_names.is_unique}')
    # show the non unique values in adata.obs_names and adata.var_names
    logger.info(f'duplicated adata.obs_names \n{adata_raw_from_adat.obs_names[adata_raw_from_adat.obs_names.duplicated()]}')
    logger.info(f'duplicated adata.var_names \n{adata_raw_from_adat.var_names[adata_raw_from_adat.var_names.duplicated()]}')
    logger.info(f'adata_raw_from_adat.var.head(5)\n{adata_raw_from_adat.var.head(5)}')
    logger.info(f'adata_raw_from_adat.obs.head(5)\n{adata_raw_from_adat.obs.head(5)}')
    logger.info(f'adata_raw_from_adat.obs.tail(5)\n{adata_raw_from_adat.obs.tail(5)}')
    ##
    ##### make new layer named 'RFU' from X
    adata_raw_from_adat.layers['RFU']=adata_raw_from_adat.X.copy()
    ############
    # c ) save the raw adata
    # Output Preparation
    # make sure object type columns in adata.obs are converted to string
    for col in adata_raw_from_adat.obs.select_dtypes(include=['object']).columns:
        adata_raw_from_adat.obs[col] = adata_raw_from_adat.obs[col].astype(str)
    raw_h5ad_filename = (raw_output_filename + ".h5ad") if raw_output_filename else "raw_adat_2_adata.h5ad"
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        raw_h5ad_path = Path(output_dir) / raw_h5ad_filename
        raw_output_path = Path(output_dir) / raw_output_filename if raw_output_filename else raw_h5ad_path.with_suffix("")
    else:
        raw_h5ad_path = Path(raw_h5ad_filename).resolve()
        raw_output_path = raw_h5ad_path.with_suffix("")
    if save_raw_h5ad:
        if also_raw_save_csvs:
            _save_dataset(adata_raw_from_adat, raw_output_path)
        else:
            adata_raw_from_adat.write_h5ad(raw_h5ad_path)
            print(f"Saved AnnData to {raw_h5ad_path}")
    elif also_raw_save_csvs:
        obs_csv_path = raw_output_path.with_suffix(".obs.csv")
        var_csv_path = raw_output_path.with_suffix(".var.csv")
        X_csv_path = raw_output_path.with_suffix(".X.csv")
        adata_raw_from_adat.obs.to_csv(obs_csv_path)
        adata_raw_from_adat.var.to_csv(var_csv_path)
        pd.DataFrame(
            adata_raw_from_adat.X,
            index=adata_raw_from_adat.obs_names,
            columns=adata_raw_from_adat.var_names,
        ).to_csv(X_csv_path)
        print(f"Saved CSV outputs with prefix: {raw_output_path}")

    if merge_external_metadata:
        # load the external obs metadata if provided
        if external_obs_metadata_2_merge_file and merge_key_in_external_obs_metadata and merge_key_in_raw_obs_metadata:
            external_obs_metadata_df = pd.read_csv(external_obs_metadata_2_merge_file,
                                                   index_col=merge_key_in_external_obs_metadata
                                                   )
            logger.info(f"loaded external_obs_metadata_df: \n{external_obs_metadata_df.head(2)}")
            logger.info(f"external_obs_metadata_dfs columns : \n{external_obs_metadata_df.columns}")
            # merge
            logger.info(f"Merging adata.obs with external metadata on keys: \n"
                        f"adata.obs key: {merge_key_in_raw_obs_metadata} \n"
                        f"external metadata key: {merge_key_in_external_obs_metadata}")
            # make a copy of adata to avoid modifying the original
            logger.info(f"adata_raw_from_adat.obs before merge head(2): \n{adata_raw_from_adat.obs.head(2)}")
            adata=adata_raw_from_adat.copy()
            adata.obs = adata.obs.merge(
                external_obs_metadata_df,
            left_on=merge_key_in_raw_obs_metadata,#
            right_index=True,#
            #right_on=merge_key_in_external_obs_metadata,#
            #validate='one_to_one',
                how='left'
            )
            logger.info(f"adata.obs after merge head(2): \n{adata.obs.head(2)}")
            logger.info(f"adata.obs columns after merge: \n{adata.obs.columns}")
            # set the index to column_in_metadata_to_set_as_index
            if column_in_metadata_to_set_as_index:
                logger.info(f"Setting adata.obs index to column: {column_in_metadata_to_set_as_index}")
                adata.obs.set_index(column_in_metadata_to_set_as_index, inplace=True, drop=False)
            # rename index to obs_names
            adata.obs.index.name = 'obs_names'
            adata.obs_names = adata.obs.index.astype(str)# make sure obs_names are string
            logger.info(f"adata.obs after merge: \n{adata.obs.head(2)}")
            logger.info(f"adata.obs columns after merge: \n{adata.obs.columns}")
            # convert all columns of type object to string
            for col in adata.obs.select_dtypes(include=['object']).columns:
                adata.obs[col] = adata.obs[col].astype(str)
            logger.info(f"adata.obs dtypes after conversion: \n{adata.obs.dtypes}")
        # merge external var metadata if provided
        if external_var_metadata_2_merge_file and merge_key_in_external_var_metadata:
            logger.info(f"before any var merges adata.var.shape {adata.var.shape }")
            external_var_metadata_df = pd.read_csv(external_var_metadata_2_merge_file
            , index_col=merge_key_in_external_var_metadata
            )
            # select only specified columns if provided
            if columns_in_external_var_metadata_to_use:
                external_var_metadata_df = external_var_metadata_df[columns_in_external_var_metadata_to_use]
            logger.info(f"loaded external_var_metadata_df: \n{external_var_metadata_df.head(2)}")
            logger.info(f"external_var_metadata_dfs columns : \n{external_var_metadata_df.columns}")
            # drop any rows with duplicate values in merge_key_in_external_var_metadata
            external_var_metadata_df = external_var_metadata_df[~external_var_metadata_df.index.duplicated(keep='first')]
            # merge
            logger.info(f"Merging adata.var with external var metadata on keys: \n"
                        f"adata.index key : \n"
                        f"external var metadata key: {merge_key_in_external_var_metadata}")
            logger.info(f"adata_raw_from_adat.var before merge head(2): \n{adata.var.head(2)}")
            new_var = adata.var.merge(
                external_var_metadata_df,
            left_on=merge_key_in_external_var_metadata,#
            right_index=True,#
            #validate='one_to_one',
                how='left'
            )
            logger.info(f"new_var {new_var.head()}")
            logger.info(f"new_var.shape {new_var.shape}")
            adata.var = new_var.copy()
            # rename index to var_names
            adata.var.index.name = 'var_names'
            logger.info(f"after merge adata.var.shape {adata.var.shape }")
            logger.info(f"adata.var after merge head(2): \n{adata.var.head(2)}")
            logger.info(f"adata.var columns after merge: \n{adata.var.columns}")
            # convert all columns of type object to string
            for col in adata.var.select_dtypes(include=['object']).columns:
                adata.var[col] = adata.var[col].astype(str)
            logger.info(f"adata.var dtypes after conversion: \n{adata.var.dtypes}")
        ########### remove QC samples and QC aptamers if needed 
        if remove_mouse_QC_apatmers_QC_samples:
            logger.info(f"Removing mouse and QC aptamers from adata.var")
            initial_var_count = adata.n_vars
            adata=adata[:,
             (adata.var['Type'].str.contains('Protein')
              &(~adata.var['Organism'].str.contains('Mouse')))].copy()
            final_var_count = adata.n_vars
            logger.info(f"Removed {initial_var_count - final_var_count} aptamers. Remaining aptamers: {final_var_count}")
            logger.info(f"Removing QC samples from adata.obs")
            initial_obs_count = adata.n_obs
            adata=adata[~adata.obs['SampleType'].isin(['QC', 'Buffer', 'Calibrator'])].copy()
            final_obs_count = adata.n_obs
            logger.info(f"Removed {initial_obs_count - final_obs_count} samples. Remaining samples: {final_obs_count}")
        # save the updated adata
        plus_metadata_h5ad_filename = (plus_metadata_file_name + ".h5ad") if plus_metadata_file_name else "raw_adat_2_adata.h5ad"
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plus_metadata_h5ad_path = Path(output_dir) / plus_metadata_h5ad_filename
            plus_metadata_output_path = Path(output_dir) / plus_metadata_file_name
        else:
            plus_metadata_h5ad_path = Path(plus_metadata_h5ad_filename).resolve()
            plus_metadata_output_path = plus_metadata_h5ad_path.with_suffix("")
        _save_dataset(adata, plus_metadata_output_path)
        logger.info(f"Saved updated AnnData with external metadata to {plus_metadata_h5ad_path}")
        return adata , adata_raw_from_adat
    else:
        return adata_raw_from_adat
