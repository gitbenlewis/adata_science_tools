#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/general_dataset_template_private/scripts/make_merged_tables.py
# updated: 2026-03-04
####################################
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
OUTPUT_DIR = Path(CFG["merge_tables_params"]["repo_results_dir"])
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
print(f"Using adata_science_tools / adtl from {adtl.__file__}")
try:    
    from code_library import run_GSEApy_wrapper as rgw
    print(f"Using run_GSEApy_wrapper / rgw from {rgw.__file__}")
except ImportError as e:
    print(f"run_GSEApy_wrapper not available: {e}")
try:    
    from code_library import RNAseq_analysis as rnaseq
    print(f"Using RNAseq_analysis / rnaseq from {rnaseq.__file__}")
except ImportError as e:
    print(f"RNAseq_analysis not available: {e}")
########################################################## import custom code libraries ################################################

# 
#### paths
MERGE_TABLES_PARAMS=CFG['merge_tables_params']
MERGE_TABLES_RUNS_PARAMS=MERGE_TABLES_PARAMS['merge_table_runs']
MERGE_TABLES_DEFAULTS = MERGE_TABLES_PARAMS.get("defaults_params", {})

# ------------- parameters end --------------------------------------------------------


def reorder_columns_front(df: pd.DataFrame, preferred_cols: list[str]) -> pd.DataFrame:
    """Move preferred_cols to the front, preserving their order and the remaining column order."""
    existing_preferred = [c for c in preferred_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_preferred]
    return df.loc[:, existing_preferred + remaining]



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

def merge_diff_test_tables(merge_run_key,**kwargs):
    """Merge diff test tables from multiple runs."""
    LOGGER.info("Starting merge run: %s", merge_run_key)
    merge_run_params=MERGE_TABLES_RUNS_PARAMS.get(merge_run_key,{})
    LOGGER.info("merge_run_params: %s", merge_run_params)
    run_control=merge_run_params.get('run', True)
    save_path=merge_run_params.get('save_path','./no_path_merged_table.csv')
    csv_path_list=merge_run_params.get('csv_path_list',[])
    suffix_list=merge_run_params.get('suffix_list',[])
    shared_column_key=merge_run_params.get('shared_column_key','')
    zipped_csv_and_suffix_list=zip(csv_path_list,suffix_list)
    front_columns=merge_run_params.get('front_columns',[])
    add_opposite_direction_flag_calls_dict=merge_run_params.get('add_opposite_direction_flag_calls_dict',None)
    LOGGER.info(f'add_opposite_direction_flag_calls_dict: {add_opposite_direction_flag_calls_dict}')
    LOGGER.info("zipped_csv_and_suffix_list: %s", zipped_csv_and_suffix_list)
    if run_control:
        LOGGER.info("Merging csv files from diff_test runs.")
        # load first csv in list 
        #df_1=pd.read_csv(csv_path_list[0],index_col=shared_column_key)
        #df_merged=pd.DataFrame()
        LOGGER.info("Loading first csv from path: %s", csv_path_list[0])
        df_merged=pd.read_csv(csv_path_list[0],
                              #index_col=shared_column_key
                              )
        zipped_list = list(zipped_csv_and_suffix_list)
        if len(zipped_list) > 1:
            LOGGER.info("Merging in second csv from path: %s", zipped_list[1][0])
            LOGGER.info("Using suffixes: %s, %s", zipped_list[0][1], zipped_list[1][1])
            df_2_merge = pd.read_csv(zipped_list[1][0])
            df_merged = df_merged.merge(
                df_2_merge,
                on=shared_column_key,
                suffixes=(zipped_list[0][1], zipped_list[1][1]),
                how='outer',
                validate='one_to_one'
            )
            start_idx = 2
        else:
            start_idx = 1
        for csv_path, suffix in zipped_list[start_idx:]:
            LOGGER.info("Merging in csv from path: %s", csv_path)
            LOGGER.info("Using suffix: %s", suffix)
            df_2_merge=pd.read_csv(csv_path,
                                   #index_col=shared_column_key
                                   )
            LOGGER.info("Merging df_merged shape: %s with df_2_merge shape: %s", df_merged.shape, df_2_merge.shape)
            df_merged=df_merged.merge(df_2_merge,
                                      #left_index=True, 
                                      #right_index=True,
                                        on=shared_column_key,
                                      suffixes=('', suffix),
                                      how='outer',
                                      validate='one_to_one')
            LOGGER.info(f"Post-merge df_merged shape: {df_merged.shape}")
            LOGGER.info(f"Post-merge df_merged columns: {df_merged.columns}")
        if add_opposite_direction_flag_calls_dict:
            LOGGER.info("Adding opposite direction flag column as per config.")
            for call_key, call_values in add_opposite_direction_flag_calls_dict.items():
                LOGGER.info(f"call_key: {call_key} with values: {call_values}")
                df_merged=add_opposite_direction_flag(
                    df_merged,
                    new_col_name=call_values['new_col_name'],
                    ref_cols=call_values['ref_cols'],
                    reversed_ref_cols=call_values['reversed_ref_cols'],
                )
        LOGGER.info(f"Columns before reordering: {df_merged.columns}")
        if front_columns:
            LOGGER.info("Reordering columns to move front_columns to front: %s", front_columns)
            df_merged=reorder_columns_front(df_merged, front_columns)
        LOGGER.info(f"Columns after reordering: {df_merged.columns}")
        LOGGER.info("Merged df shape: %s", df_merged.shape)
        LOGGER.info("Saving merged df to path: %s", save_path)
        df_merged.to_csv(save_path, index=False)
 

if __name__ == "__main__":

    LOGGER.info("Starting make_merged_tables.py script.")
    for merge_run_key in MERGE_TABLES_RUNS_PARAMS.keys():
        LOGGER.info("Running merge_table_runs key: %s", merge_run_key)
        merge_diff_test_tables(merge_run_key)


