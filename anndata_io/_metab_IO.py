# adata_science_tools/anndata_io/_metab_IO.py
# functions to read Metabolon Excel files into AnnData objects
# module level package imports
# updated 2026-02-04
import logging
from typing import Dict, List, TextIO, Tuple, Union
import numpy as np
import anndata as ad
import pandas as pd
import anndata as ad
from pathlib import Path

def metabolon_excel_2_adata_h5ad_csv(
    metabolon_excel_file:  str | None = None,
    excel_sheet_for_assay_data: str = "Batch-norm Imputed Data",
    excel_sheet_for_obs_metadata: str = "Sample Meta Data", 
    excel_sheet_for_var_metadata: str = "Chemical Annotation",
    index_col_for_var_metadata='CHEM_ID',
    excel_sheet_list_for_layers: list = ["Volume-normalized Data","Log Transformed Data", "Batch-normalized Data", "Batch-norm Imputed Data", "Peak Area Data"],
    output_dir: Path | None = None,
    save_h5ad: bool = True,
    output_filename: str | None ='dataset.metab',
    also_save_csvs: bool = True,
    logger: logging.Logger | None = None,
    ######## new parameters for mergeing external metadata to adata.obs can be added here ########
    merge_external_metadata: bool = False,
    save_plus_metadata_h5ad: bool = False,
    also_plus_metadata_save_csvs: bool = False,
    external_obs_metadata_2_merge_file: str | None = None,
    external_var_metadata_2_merge_file: str | None = None,
    merge_key_in_external_obs_metadata: str | None = None,
    merge_key_in_raw_obs_metadata: str | None = None,
    column_in_metadata_to_set_as_index: str | None = None,
    merge_key_in_external_var_metadata: str | None = None,
    merge_key_in_raw_var_metadata: str | None = None,
    columns_in_external_var_metadata_to_use: list | None = None,
    plus_metadata_file_name: str | None = 'dataset.plus_metadata',
) -> ad.AnnData: 
    """Parse metabolon Excel file, attach layers, and write h5ad/obs/var outputs.
    """
    # updated 2026-02-04
    def _save_dataset(_adata,output_path,logger=logger):
        """Save dataset helper function."""
        H5AD_PATH =Path(f"{output_path}.h5ad")
        OBS_CSV_PATH = Path(f"{output_path}.obs.csv")
        VAR_CSV_PATH = Path(f"{output_path}.var.csv")
        X_CSV_PATH = Path(f"{output_path}.X.csv")
        logger.info(f"Saving adata to {H5AD_PATH}")
        _adata.write_h5ad(H5AD_PATH)
        logger.info(f"Saving adata.obs to {OBS_CSV_PATH}")
        _adata.obs.to_csv(OBS_CSV_PATH)
        logger.info(f"Saving adata.var to {VAR_CSV_PATH}")
        _adata.var.to_csv(VAR_CSV_PATH)
        logger.info(f"Saving adata.X to {X_CSV_PATH}")
        pd.DataFrame(_adata.X, index=_adata.obs_names, columns=_adata.var_names).to_csv(X_CSV_PATH)
        # Save layers as CSVs
        for layer_name, layer_data in _adata.layers.items():
            safe_name = str(layer_name).replace("/", "_")
            layer_csv_path = Path(f"{output_path}.layer.{safe_name}.csv")
            logger.info(f"Saving adata.layers['{layer_name}'] to {layer_csv_path}")
            if hasattr(layer_data, "toarray"):
                layer_data = layer_data.toarray()
            pd.DataFrame(layer_data, index=_adata.obs_names, columns=_adata.var_names).to_csv(layer_csv_path)
        output_dir = H5AD_PATH.parent
        logger.info(f"Saved dataset to directory: {output_dir}")
        basename=H5AD_PATH.stem
        logger.info(f"Saved dataset with base filename: {basename}")
    # log input parameters
    if logger:
        logger.info(f"metabolon_excel_file: {metabolon_excel_file}")
        logger.info(f"excel_sheet_for_assay_data: {excel_sheet_for_assay_data}")
        logger.info(f"excel_sheet_for_obs_metadata: {excel_sheet_for_obs_metadata}")
        logger.info(f"excel_sheet_for_var_metadata: {excel_sheet_for_var_metadata}")
        logger.info(f"index_col_for_var_metadata: {index_col_for_var_metadata}")
        logger.info(f"excel_sheet_list_for_layers: {excel_sheet_list_for_layers}")
        logger.info(f"output_dir: {output_dir}")
        logger.info(f"save_h5ad: {save_h5ad}")
        logger.info(f"output_filename: {output_filename}")
        logger.info(f"also_save_csvs: {also_save_csvs}")
        logger.info(f"merge_external_metadata: {merge_external_metadata}")
        logger.info(f"save_plus_metadata_h5ad: {save_plus_metadata_h5ad}")
        logger.info(f"also_plus_metadata_save_csvs: {also_plus_metadata_save_csvs}")
        logger.info(f"external_obs_metadata_2_merge_file: {external_obs_metadata_2_merge_file}")
        logger.info(f"external_var_metadata_2_merge_file: {external_var_metadata_2_merge_file}")
        logger.info(f"merge_key_in_external_obs_metadata: {merge_key_in_external_obs_metadata}")
        logger.info(f"merge_key_in_raw_obs_metadata: {merge_key_in_raw_obs_metadata}")
        logger.info(f"column_in_metadata_to_set_as_index: {column_in_metadata_to_set_as_index}")
        logger.info(f"merge_key_in_external_var_metadata: {merge_key_in_external_var_metadata}")
        logger.info(f"merge_key_in_raw_var_metadata: {merge_key_in_raw_var_metadata}")
        logger.info(f"columns_in_external_var_metadata_to_use: {columns_in_external_var_metadata_to_use}")
        logger.info(f"plus_metadata_file_name: {plus_metadata_file_name}")
    
    
    # First do the raw excel file parse) Parse metabolon Excel file, attach layers, and write h5ad/obs/var outputs. -------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    metabolon_path = Path(metabolon_excel_file)
    if not metabolon_path.exists():
        raise FileNotFoundError(f"Metabolon file not found: {metabolon_path}")

    # 1. Load data from excel
    assay_data = pd.read_excel(metabolon_path, sheet_name=excel_sheet_for_assay_data, index_col=0)
    ###  excel_obs_metadata
    excel_obs_metadata = pd.read_excel(metabolon_path, sheet_name=excel_sheet_for_obs_metadata, 
                                      # index_col=0
                                       )
    # set the index to the 0 column
    excel_obs_metadata.set_index(excel_obs_metadata.columns[0], inplace=True, drop=False)
    # rename index to obs_names
    excel_obs_metadata.index.name = 'obs_names'
    ###  excel_var_metadata
    excel_var_metadata = pd.read_excel(metabolon_path, sheet_name=excel_sheet_for_var_metadata, #index_col=index_col_for_var_metadata
                                       )
    # set the index to the specified column
    excel_var_metadata.set_index(index_col_for_var_metadata, inplace=True, drop=False)
    # rename index to var_names
    excel_var_metadata.index.name = 'var_names'
    # 2. Standardize index types and stripping
    assay_data.index = assay_data.index.astype(str)
    assay_data.columns = assay_data.columns.astype(str).str.strip()
    excel_obs_metadata.index = excel_obs_metadata.index.astype(str)
    excel_var_metadata.index = excel_var_metadata.index.astype(str).str.strip()

    # 3. Build adata
    adata_raw = ad.AnnData(
        X=assay_data.values,
        obs=excel_obs_metadata.loc[assay_data.index],
        var=excel_var_metadata.loc[assay_data.columns]
    )
    # set obs and var index names
    adata_raw.obs.index.name = 'obs_names'
    adata_raw.var.index.name = 'var_names'
    # add an sample order column to adata.obs
    adata_raw.obs['metab_data_table_order'] = range(1, adata_raw.n_obs + 1)

    # 4. Save layers
    for sheet_name in excel_sheet_list_for_layers:
        try:
            layer_df = pd.read_excel(metabolon_path, sheet_name=sheet_name, index_col=0)
            layer_df.index = layer_df.index.astype(str)
            layer_df.columns = layer_df.columns.astype(str).str.strip()
            
            layer_key = sheet_name.lower().replace(" ", "_").replace("-", "_")
            # Align layer data with adata structure
            adata_raw.layers[layer_key] = layer_df.loc[adata_raw.obs.index, adata_raw.var.index].values
            print(f"Added layer '{layer_key}' from sheet '{sheet_name}'.")
        except Exception as exc:
            print(f"Skipping sheet '{sheet_name}': {exc}")

    # make sure object type columns in adata.obs are converted to string
    for col in adata_raw.obs.select_dtypes(include=['object']).columns:
        adata_raw.obs[col] = adata_raw.obs[col].astype(str)
    # 5. Output Preparation
    #h5ad_filename = (output_filename + ".h5ad") if output_filename else "dataset.h5ad"
    #if output_dir:
    #    output_dir.mkdir(parents=True, exist_ok=True)
    #    h5ad_path = output_dir / h5ad_filename

    # 6. Safety: Clear index names to prevent ValueError on write_h5ad
    #adata.obs.index.name = 'obs_names'
    #adata.var.index.name = 'var_names'

    #if save_h5ad:
    #    adata_raw.write_h5ad(h5ad_path)
    #    print(f"Saved AnnData to {h5ad_path}")
    #if also_save_csvs:
    #    obs_csv_path = output_dir / f"{output_filename}.obs.csv"
    #    var_csv_path = output_dir / f"{output_filename}.var.csv"
    #    X_csv_path = output_dir / f"{output_filename}.X.csv"
    #    adata_raw.obs.to_csv(obs_csv_path)
    #    adata_raw.var.to_csv(var_csv_path)
    #    # Convert X back to DF to save as CSV
    #    pd.DataFrame(adata_raw.X, index=adata_raw.obs_names, columns=adata_raw.var_names).to_csv(X_csv_path)
    #    print(f"Saved CSV outputs with prefix: {output_filename}")
    if save_h5ad:
        logger.info(f"Saving raw adata to {output_dir / output_filename}")
        _save_dataset(adata_raw, output_dir / output_filename, logger=logger)
    # --------------------------------------------------------------------------------------------------

    # second part: merge external metadata to adata.obs can be added here -------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    if not merge_external_metadata:
        adata = adata_raw
        return adata
    else:
        # 1) load the external obs metadata if provided
        if external_obs_metadata_2_merge_file and merge_key_in_external_obs_metadata and merge_key_in_raw_obs_metadata:
            external_obs_metadata_df = pd.read_csv(external_obs_metadata_2_merge_file,
                                                  # index_col=merge_key_in_external_obs_metadata
                                                   )
            logger.info(f"loaded external_obs_metadata_df: \n{external_obs_metadata_df.head(2)}")
            logger.info(f"external_obs_metadata_dfs columns : \n{external_obs_metadata_df.columns}")
            # merge
            logger.info(f"Merging adata.obs with external metadata on keys: \n"
                        f"adata.obs key: {merge_key_in_raw_obs_metadata} \n"
                        f"external metadata key: {merge_key_in_external_obs_metadata}")
            # make a copy of adata to avoid modifying the original
            logger.info(f"adata_raw.obs before merge head(2): \n{adata_raw.obs.head(2)}")
            adata = adata_raw.copy()
            adata.obs = adata.obs.merge(
                external_obs_metadata_df,
                #left_on=merge_key_in_raw_obs_metadata,
                #right_index=True,
                left_on=merge_key_in_raw_obs_metadata,
                right_on=merge_key_in_external_obs_metadata,
                validate='one_to_one',
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
        # 2) merge external var metadata if provided
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
            validate='one_to_one',
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
        if save_plus_metadata_h5ad:
            logger.info(f"Saving plus metadata adata to {output_dir / plus_metadata_file_name}")
            # 3) save the updated adata with plus metadata
            _save_dataset(adata, output_dir / plus_metadata_file_name, logger=logger)
        return adata