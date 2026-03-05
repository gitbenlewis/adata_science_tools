#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/general_dataset_template_private/scripts/make_model_fit_tables.py
# updated: 2026-03-04
####################################
import sys
import os
from pathlib import Path
import pandas as pd
import anndata
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
OUTPUT_DIR = Path(CFG["model_fit_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
####################################



#### start #### log file setup
# ---------- logging setup ----------
import logging
from datetime import datetime
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_BASE_NAME = Path(__file__).stem
LOG_FILENAME = f"{SCRIPT_BASE_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"
RESULTS_LOG_FILE  = LOG_DIR / LOG_FILENAME
SCRIPT_LOG_DIR = Path(__file__).resolve().parent / "logs"
SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_LOG_FILE = SCRIPT_LOG_DIR / LOG_FILENAME
ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) for h in ROOT_LOGGER.handlers): # add file handler once
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
# parameters---------------------------------------------------------------------
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
# ------------- parameters --------------------------------------------------------



if __name__ == "__main__":
    LOGGER.info("Starting make_model_fit_tables.py script.")
    # model_fit()_runs ---------------------------------------------------------------------
    # OLS model fit runs-----------
    # SET DEFAULTS for OLS model_fit() RUNS
    LOGGER.info("Starting OLS model_fit() runs as per config.")
    LAYER=CFG['model_fit_params']['default_params']['layer']
    ADD_ADATA_VAR_COLUMN_KEY_LIST=CFG['model_fit_params']['default_params']['add_adata_var_column_key_list']
    DICTIONARY_OF_OLS_MODEL_FIT_RUNS=CFG['model_fit_params']['smf_ols_model_fit_runs']
    SAVE_TABLE=CFG['model_fit_params']['default_params']['save_table']
    FEATURE_COLUMNS=CFG['model_fit_params']['default_params']['feature_columns']
    # log defaults set above
    LOGGER.info(f"Defaults set for OLS model_fit() runs: ")
    LOGGER.info(f"layer: {LAYER} ")
    LOGGER.info(f"add_adata_var_column_key_list: {ADD_ADATA_VAR_COLUMN_KEY_LIST} ")
    LOGGER.info(f"save_table: {SAVE_TABLE} ")
    LOGGER.info(f"feature_columns: {FEATURE_COLUMNS} ")

    ## start loop through DICTIONARY_OF_OLS_MODEL_FIT_RUNS
    for run_key, run_values in DICTIONARY_OF_OLS_MODEL_FIT_RUNS.items():
        LOGGER.info(f"run_key: {run_key}  with info: \n {run_values}")
        LOGGER.info(f'run control set to: {run_values["run"]}')
        if run_values["run"]:
            LOGGER.info(f"Proceeding with OLS model_fit() for {run_key} as per config.")
            LOGGER.info(f"Loading adata from path: {run_values['adata_path']}")
            adata=anndata.read_h5ad(Path(run_values['adata_path']) )
            LOGGER.info(f"laoded adata: \n{adata}")
            LOGGER.info(adata.var.head(2))
            LOGGER.info(adata.obs.head(2)) # display the first 2 rows of the
            ############ run model_fit() ############
            ols_summary_df = adtl.fit_smf_ols_models_and_summarize_adata(
            adata,
            layer=run_values['layer'],# LAYER,
            feature_columns=FEATURE_COLUMNS,
            predictors=run_values['predictors'],#predictors,
            model_name=run_values['model_name'],#model_name,
            add_adata_var_column_key_list=run_values['add_adata_var_column_key_list'],#ADD_ADATA_VAR_COLUMN_KEY_LIST,
            save_table=SAVE_TABLE,
            save_path=run_values['save_path'],
            save_result_to_adata_uns_as_dict=run_values['save_result_to_adata_uns_as_dict'],
            )

            LOGGER.info(ols_summary_df.shape) # print the shape of the results dataframe
            LOGGER.info(ols_summary_df.head(10)) # display the first 10 rows of the results dataframe
        else:
            LOGGER.info(f"Skipping OLS model_fit() for {run_key} as per config.")
    # ----------- OLS model fit runs-----------

    # lmem model fit runs-----------
    # SET DEFAULTS for lmem model_fit() RUNS
    LOGGER.info("Starting lmem model_fit() runs as per config.")
    LAYER=CFG['model_fit_params']['default_params']['layer']
    ADD_ADATA_VAR_COLUMN_KEY_LIST=CFG['model_fit_params']['default_params']['add_adata_var_column_key_list']
    DICTIONARY_OF_LMEM_MODEL_FIT_RUNS=CFG['model_fit_params'].get('smf_lmem_model_fit_runs') or {}
    SAVE_TABLE=CFG['model_fit_params']['default_params']['save_table']
    FEATURE_COLUMNS=CFG['model_fit_params']['default_params']['feature_columns']
    # log defaults set above
    # log defaults set above
    LOGGER.info(f"Defaults set for lmem model_fit() runs: ")
    LOGGER.info(f"layer: {LAYER} ")
    LOGGER.info(f"add_adata_var_column_key_list: {ADD_ADATA_VAR_COLUMN_KEY_LIST} ")
    LOGGER.info(f"save_table: {SAVE_TABLE} ")
    LOGGER.info(f"feature_columns: {FEATURE_COLUMNS} ")
    ## start loop through DICTIONARY_OF_LMEM_MODEL_FIT_RUNS
    for run_key, run_values in DICTIONARY_OF_LMEM_MODEL_FIT_RUNS.items():
        LOGGER.info(f"run_key: {run_key}  with info: \n {run_values}")
        LOGGER.info(f'run control set to: {run_values["run"]}')
        if run_values["run"]:
            LOGGER.info(f"Proceeding with lmem model_fit() for {run_key} as per config.")
            LOGGER.info(f"Loading adata from path: {run_values['adata_path']}")
            adata=anndata.read_h5ad(Path(run_values['adata_path']) )
            LOGGER.info(f"laoded adata: \n{adata}")
            LOGGER.info(adata.var.head(2))
            LOGGER.info(adata.obs.head(2)) # display the first 2 rows of the
            ############ run model_fit() ############
            lmem_summary_df = adtl.fit_smf_mixedlm_models_and_summarize_adata(
            adata,
            layer=run_values['layer'],# LAYER,
            feature_columns=FEATURE_COLUMNS,
            predictors=run_values['predictors'],#predictors,
            model_name=run_values['model_name'],#model_name,
            group=run_values['group'],
            add_adata_var_column_key_list=run_values['add_adata_var_column_key_list'],#ADD_ADATA_VAR_COLUMN_KEY_LIST,
            save_table=SAVE_TABLE,
            save_path=run_values['save_path'],
            save_result_to_adata_uns_as_dict=run_values['save_result_to_adata_uns_as_dict'],
            )

            LOGGER.info(lmem_summary_df.shape) # print the shape of the results dataframe
            LOGGER.info(lmem_summary_df.head(10)) # display the first 10 rows of the results dataframe
        else:
            LOGGER.info(f"Skipping lmem model_fit() for {run_key} as per config.")
    # ----------- lmem model fit runs-----------

    # ------------- model_fit()_runs  --------------------------------------------------------
