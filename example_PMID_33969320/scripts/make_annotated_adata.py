#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_PMID_33969320/scripts/make_annotated_adata.py

import sys
import os
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import logging
import yaml

 # CFG Configuration
####################################
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIG_YAML_PATH = REPO_ROOT / "config" / "config.yaml"
with REPO_CONFIG_YAML_PATH.open() as f:
    CFG = yaml.safe_load(f)

# out and log path 
OUTPUT_DIR = Path(CFG["make_annotated_adata_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------- Configuration ----------------------------------------------------

#### start #### log file setup
# ---------- logging setup ----------
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_BASE_NAME = Path(__file__).stem
LOG_FILENAME = f"{SCRIPT_BASE_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"
RESULTS_LOG_FILE = LOG_DIR / LOG_FILENAME
SCRIPT_LOG_DIR = Path(__file__).resolve().parent / "logs"
SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_LOG_FILE = SCRIPT_LOG_DIR / LOG_FILENAME
ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) for h in ROOT_LOGGER.handlers):
    for log_path in (RESULTS_LOG_FILE, SCRIPT_LOG_FILE):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        fh.setLevel(logging.INFO)
        ROOT_LOGGER.addHandler(fh)
if not any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in ROOT_LOGGER.handlers
):
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    sh.setLevel(logging.INFO)
    ROOT_LOGGER.addHandler(sh)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Logging to %s", RESULTS_LOG_FILE)
LOGGER.info("Logging to %s", SCRIPT_LOG_FILE)
logging.captureWarnings(True)
logging.getLogger("py.warnings").propagate = True
# dataclass G() ---------------------------------------------------------------------
# save all the global variable in a dataclass G
from dataclasses import dataclass
@dataclass
class G():
    '''Class to hold global variables'''
    WRITE_DIR='/home/ubuntu/write/'
    ACTIO_REPO_PARENT_DIR='/home/ubuntu/projects/ActioBio/'
    GITBENLEWIS_REPO_PARENT_DIR='/home/ubuntu/projects/gitbenlewis/'
    SCRIPTS_DIR='../scripts/'
    CONFIG_DIR='../config/'
    RESULTS_DIR='../results/'
    WRITE_CACHE=False
    ##### input files   
    
    # control variables to change
    SAVE_OUTPUT=True
    SAVE_OUTPUT_FIGURES=True
# ------------- dataclass G()  --------------------------------------------------------

########## import custom code libraries ################################################
import sys
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
# Use the current working directory instead of __file__
#REPO_ROOT = Path(os.getcwd()).resolve().parent.parent
print(f"REPO_ROOT set to: {str(REPO_ROOT)}")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from code_library import adata_science_tools as adtl
print(f"Using adtl from {adtl.__file__}")
#from code_library import gseapy_pre_rank_wrap as gprw
#print(f"Using gprw from {gprw.__file__}")
#from code_library import gseapy_dot_plots as gdp
#print(f"Using gdp from {gdp.__file__}")

########################################################## import custom code libraries ################################################



def add_opposite_direction_flag(
    df,
    new_col_name,
    ref_cols,
    reversed_ref_cols,
):
    """
    Add a boolean column that's True when all ref_cols are positive
    and all reversed_ref_cols are negative OR vice versa.
    """
    ref_pos = (df[ref_cols] > 0).all(axis=1)
    ref_neg = (df[ref_cols] < 0).all(axis=1)
    rev_pos = (df[reversed_ref_cols] > 0).all(axis=1)
    rev_neg = (df[reversed_ref_cols] < 0).all(axis=1)

    df[new_col_name] = (ref_pos & rev_neg) | (ref_neg & rev_pos)
    return df

import anndata


def merge_feature_results_to_var(
    adata: anndata.AnnData,
    feature_results: str | Path | pd.DataFrame,
    columns_to_include: list[str] | None = None,
    values_suffix: str = "_values",
    coerce_values_to_str: bool = True,
    feature_results_index_col: str = "var_names", # should match adata.var.index
    always_add_suffix_result_columns: bool = False,
    merge_suffixes: tuple[str, str] = ("", "_feature_results"),
    how: str = "left", # left is default to keep all genes in adata.var
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Merge feature_results (DataFrame or CSV path) into adata.var."""
    if feature_results is None:
        raise ValueError("feature_results is required.")
    log = logger or logging.getLogger(__name__)
    if isinstance(feature_results, (str, Path)):
        index_col_arg = feature_results_index_col or 0
        feature_results_df = pd.read_csv(feature_results, index_col=index_col_arg
                                         )
    elif isinstance(feature_results, pd.DataFrame):
        feature_results_df = feature_results
 
    else:
        raise TypeError("feature_results must be a pandas.DataFrame or a CSV path.")
    # log the shape of the DataFrame, the column names and the index name
    log.info("Feature results DataFrame shape: %s", feature_results_df.shape)
    log.info("Feature results DataFrame columns: %s", feature_results_df.columns.tolist())
    log.info("Feature results DataFrame index name: %s", feature_results_df.index.name)
    log.info("adata.var index name: %s", adata.var.index.name)

    unnamed_cols = [c for c in feature_results_df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        feature_results_df = feature_results_df.drop(columns=unnamed_cols)
        log.warning("Dropping unnamed columns from feature results: %s", unnamed_cols)

    if columns_to_include is not None:
        cols_present = [c for c in columns_to_include if c in feature_results_df.columns]
        if feature_results_index_col in feature_results_df.columns and feature_results_index_col not in cols_present:
            cols_present = [feature_results_index_col] + cols_present
        missing = [c for c in columns_to_include if c not in feature_results_df.columns]
        if missing:
            log.warning("feature_results: missing columns skipped: %s", missing)
        feature_results_df = feature_results_df.loc[:, cols_present].copy()
    else:
        feature_results_df = feature_results_df.copy()

    if feature_results_index_col in feature_results_df.columns:
        feature_results_df = feature_results_df.set_index(feature_results_index_col, drop=True)
    elif feature_results_df.index.name == feature_results_index_col:
        log.info( f" feature_results_df index name: {feature_results_df.index.name} matches requested '{feature_results_index_col}'")
    else:
        log.warning(f"merge_feature_results_to_var: '{feature_results_index_col}' not found in columns; using existing index.")
        log.info( f" feature_results_df index name: {feature_results_df.index.name}")

    if coerce_values_to_str:
        value_cols = [c for c in feature_results_df.columns if c.endswith(values_suffix)]
        if value_cols:
            feature_results_df[value_cols] = feature_results_df[value_cols].astype(str)

    if always_add_suffix_result_columns:
        log.info( f" Adding suffix '{merge_suffixes[1]}' to feature_results result columns.")
        new_column_names = {
            col: f"{col}{merge_suffixes[1]}" for col in feature_results_df.columns if not col.endswith(merge_suffixes[1])
        }
        log.info( f" Renaming feature_results result columns: {new_column_names}")
        feature_results_df = feature_results_df.rename(columns=new_column_names)
    # set both indexes to string to avoid merge issues
    adata.var.index = adata.var.index.map(str)
    feature_results_df.index = feature_results_df.index.map(str)
    log.info("Merging feature results into adata.var using how='%s' with suffixes=%s", how, merge_suffixes)
    log.info("adata.var before merge shape: %s", adata.var.shape)
    merged_var = adata.var.merge(feature_results_df, left_index=True, right_index=True, how=how, suffixes=merge_suffixes)
    adata.var = merged_var
    log.info("adata.var after merge shape: %s", adata.var.shape)
    log.info(adata.var.head(2))
    return merged_var


# -- SCRIPT globals-------------------------------------------------------------------
INPUT_ADATA_RUN_PARAMS = CFG['make_annotated_adata_params']['annotate_adata_runs']['input']
ADD_PCS_RESULTS_TO_ADATA = CFG['make_annotated_adata_params']['annotate_adata_runs']['input']['PCA_analysis_params']['add_pca_results_to_adata'] or False
ADD_OPPOSITE_DIRECTION_FLAG_CALLS_DICT = INPUT_ADATA_RUN_PARAMS.get('add_opposite_direction_flag_calls_dict',None)
ADATA_H5AD_PATH = Path(INPUT_ADATA_RUN_PARAMS['adata_input_h5ad_path'])
ANNOTATED_ADATA_OUTPUT_H5AD_PATH =  Path(INPUT_ADATA_RUN_PARAMS['annotated_adata_output_h5ad_path'])

DICTIONARY_OF_DIRS_TO_MERGE=INPUT_ADATA_RUN_PARAMS['dirs_with_tables_2_merge_2_adata_var']


#PCA_PARAMS = CFG['pca_params']
#PCA_PARAMS_RUN_INPUT = CFG['pca_params']['pca_runs']['input']
#SAMPLES_X_PCA_TABLE_PATH= Path(PCA_PARAMS_RUN_INPUT['samples_x_pca_table_path'])
#PCA_LOADINGS_TABLE_PATH= Path(PCA_PARAMS_RUN_INPUT['pca_loadings_table_path'])
#VARIANCE_EXPLAINED_RATIO_TABLE_PATH= Path(PCA_PARAMS_RUN_INPUT['variance_explained_ratio_table_path'])
#OBSM_KEY_FOR_PCS = PCA_PARAMS_RUN_INPUT.get('obsm_key_for_pcs', 'X_pca_custom')


PARSE_ANNOTATED_ADATA = CFG['make_annotated_adata_params']['annotate_adata_runs']['input'].get('parse_annotate_adata', False)

# ----------------------------------------SCRIPT globals-------------------------------------------


if __name__ == "__main__":
    LOGGER.info("Loading adata from %s", ADATA_H5AD_PATH)# ADATA_H5AD_PATH # ADATA_ANN_H5AD_PATH
    adata = anndata.read_h5ad(ADATA_H5AD_PATH)
    LOGGER.info(f"adata loaded  {adata}")
    LOGGER.info(f"adata loaded with {adata.n_obs} cells and {adata.n_vars} genes.")
    LOGGER.info(f"adata loaded adata.var with columns {adata.var.columns} genes.")
    LOGGER.info(f"adata.var index name: {adata.var.index.name}")

    if ADD_PCS_RESULTS_TO_ADATA:
        import anndata as ad
        # load the PCA table
        pca_df = pd.read_csv(SAMPLES_X_PCA_TABLE_PATH, index_col=0)
        # load loadings table
        loadings_df = pd.read_csv(PCA_LOADINGS_TABLE_PATH, index_col=0)
        # load variance explained ratio table
        variance_explained_df = pd.read_csv(VARIANCE_EXPLAINED_RATIO_TABLE_PATH)
        LOGGER.info("Adding PCA results to adata from PCA tables")

        pca_columns = [col for col in pca_df.columns if col.startswith('PC')]
        pcs_only = pca_df[pca_columns]
        pcs_only = pcs_only.reindex(adata.obs_names)
        adata.obsm[OBSM_KEY_FOR_PCS] = pcs_only.to_numpy()
        adata.uns[f"{OBSM_KEY_FOR_PCS}_variance_ratio"] = variance_explained_df["Explained Variance"].astype(float).values
        loadings_aligned = loadings_df.reindex(adata.var_names)
        adata.varm[f"{OBSM_KEY_FOR_PCS}_loadings"] = loadings_aligned.to_numpy()
        print(f'Added PCA results to adata.obsm["{OBSM_KEY_FOR_PCS}"], adata.uns["{OBSM_KEY_FOR_PCS}_variance_ratio"], adata.varm["{OBSM_KEY_FOR_PCS}_loadings"]')
        print(adata)
        #adata.write_h5ad(ADATA_ANN_H5AD_PATH)
        #print(f"Added PCA tables to adata and saved to {ADATA_ANN_H5AD_PATH}")

    #LOGGER.info("START ######## Merging DA/deseq2 test results into adata.var")

    ADATA_VAR_INDEX = adata.var.index.name
    LOGGER.info(f"adata.var index name used: {ADATA_VAR_INDEX}")
    for dir, dir_values in DICTIONARY_OF_DIRS_TO_MERGE.items():
        LOGGER.info(f"var table dir key: {dir}  with info: \n {dir_values}")
        LOGGER.info(f'merge control set to: {dir_values["merge"]}')
        if dir_values["merge"]:
            # zip through the filenames and merge each one
            zip_filenames_and_tags=zip(dir_values['var_table_filenames_list'],dir_values['suffix_for_adata_var_columns_list'])
            for table_filename, suffix in zip_filenames_and_tags:
                table_path = Path(dir_values['var_results_dir']) / Path(table_filename)
                comparison_tag = suffix
                LOGGER.info(f" Merging var result table for suffix/comparison_tag: {comparison_tag} from\n{table_path}")
                merged_var = merge_feature_results_to_var(
                    adata,
                    table_path,
                feature_results_index_col=dir_values.get('feature_results_index_col', str(ADATA_VAR_INDEX)),
                always_add_suffix_result_columns=dir_values['always_add_suffix_result_columns'] or False,
                merge_suffixes=("", f"_{comparison_tag}"),
                how="left",
                logger=LOGGER,)
                LOGGER.info(f"adata after merging var results table: {adata}")

            LOGGER.info(f"adata after merging DA/deseq2 results  {adata}")
        elif not dir_values["merge"]:
            LOGGER.info(f"Skipping merging var tables from dir key: {dir} as 'merge' is set to False")
        else:
            LOGGER.warning(f"Invalid 'merge' value for dir key: {dir}. Expected True or False, got {dir_values['merge']}")

    if ADD_OPPOSITE_DIRECTION_FLAG_CALLS_DICT:
        LOGGER.info("Adding opposite direction flag column as per config.")
        df_var = adata.var.copy()
        for call_key, call_values in ADD_OPPOSITE_DIRECTION_FLAG_CALLS_DICT.items():
            LOGGER.info(f"call_key: {call_key} with values: {call_values}")
            df_var=add_opposite_direction_flag(
                df_var,
                new_col_name=call_values['new_col_name'],
                ref_cols=call_values['ref_cols'],
                reversed_ref_cols=call_values['reversed_ref_cols'],
            )
        adata.var = df_var
    LOGGER.info(f"adata.var columns after all merging and flag additions: {adata.var.columns}")

    # save the annotated adata
    LOGGER.info(f"Saving annotated adata to {ANNOTATED_ADATA_OUTPUT_H5AD_PATH}")
    # convert any columns with object dtype to string to avoid issues with saving
    for col in adata.var.select_dtypes(include=['object']).columns:
        adata.var[col] = adata.var[col].astype(str)
    # ensure the output directory exists
    ANNOTATED_ADATA_OUTPUT_H5AD_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(ANNOTATED_ADATA_OUTPUT_H5AD_PATH)
    LOGGER.info(f"make_annotated_adata.py All done!")


    #if PARSE_ANNOTATED_ADATA:
        #from make_parse_datasets import parse_datasets_somascan
        #PARSE_DATASET_CALLS_DICT= CFG["parse_datasets_params"]["parse_dataset_calls"]
        #PARSE_DATASET_PARAMS_INPUT=PARSE_DATASET_CALLS_DICT['annotated'] 



