#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_PMID_33969320/scripts/make_diff_test_tables.py


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
OUTPUT_DIR = Path(CFG["make_diff_test_tables_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
####################################

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_LOG_FILE),
        logging.FileHandler(SCRIPT_LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Logging to %s", RESULTS_LOG_FILE)
LOGGER.info("Logging to %s", SCRIPT_LOG_FILE)
logging.captureWarnings(True)
logging.getLogger("py.warnings").propagate = True
# ---------- logging setup ----------

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
print(f"Using adtl from {adtl.__file__}")
#from code_library import gseapy_pre_rank_wrap as gprw
#print(f"Using gprw from {gprw.__file__}")
#from code_library import gseapy_dot_plots as gdp
#print(f"Using gdp from {gdp.__file__}")

########################################################## import custom code libraries ################################################
# ------------- parameters --------------------------------------------------------

from code_library import adata_science_tools as adtl
print("adtl: %s", adtl.__file__)
LOGGER.info("adtl %s:", adtl.__file__)


if __name__ == "__main__":
    LOGGER.info("Starting make_diff_test_tables.py script.")
    print('test hello')
# diff_test()_runs ---------------------------------------------------------------------

# SET DEFAULTS FOR diff_test() RUNS
from collections import ChainMap
DICTIONARY_OF_DIFF_TEST_RUNS=CFG['make_diff_test_tables_params']['diff_test_runs']
DEFAULT_PARAMS=CFG['make_diff_test_tables_params']['default_params']
for run_key, run_values in DICTIONARY_OF_DIFF_TEST_RUNS.items():
    if run_key == "COVID_over_NOT_D0":
        LOGGER.info(f"###################################################################################################")
        LOGGER.info(f"Starting diff_test() run for {run_key} with info: \n {run_values}")
        LOGGER.info(f'run control set to: {run_values["run"]}')
        chain_run_values=ChainMap(run_values, DEFAULT_PARAMS)
        if chain_run_values["run"]:
            LOGGER.info(f"Proceeding with diff_test() for {run_key} as per config.")
            LOGGER.info(f"Loading adata from path: {chain_run_values['adata_path']}")
            adata=anndata.read_h5ad(Path(chain_run_values['adata_path']) )
            LOGGER.info(f"laoded adata: \n{adata}")
            LOGGER.info(adata.var.head(2))
            LOGGER.info(adata.obs.head(2)) # display the first 2 rows of the 
            ############ run diff_test() ############
            _diff_test_df=adtl.diff_test(
            adata[adata.obs['Day']=='0', :], # subset adata to only include Day 0 samples for this run
            layer=chain_run_values.get('layer', None),
            groupby_key=chain_run_values.get('groupby_key','Treatment'),# 'Treatment',
            groupby_key_target_values=chain_run_values['groupby_key_target_values'],
            groupby_key_ref_values=chain_run_values['groupby_key_ref_values'],
            comparison_col_tag=chain_run_values['comparison_col_tag'],
            tests=chain_run_values['tests'],
            pair_by_key=chain_run_values.get('pair_by_key', None),
            add_values2results= chain_run_values.get('add_values2results', None),
            sortby=chain_run_values.get('sortby', None),
            ascending=chain_run_values.get('ascending', True),
            add_adata_var_column_key_list=chain_run_values.get('add_adata_var_column_key_list', None),
            save_table=G.SAVE_OUTPUT,
            logger=LOGGER,
            save_path=chain_run_values['save_path'],
            #save_result_to_adata_uns_as_dict=True,
            )
            LOGGER.info(_diff_test_df.shape) # print the shape of the results dataframe
            LOGGER.info(_diff_test_df.head(10)) # display the first 10 rows of the results dataframe
        else:
            LOGGER.info(f"Skipping diff_test() for {run_key} as per config.")

    else:
        LOGGER.info(f"###################################################################################################")
        LOGGER.info(f"run_key: {run_key}  with info: \n {run_values}")
        LOGGER.info(f'run control set to: {run_values["run"]}')
        chain_run_values=ChainMap(run_values, DEFAULT_PARAMS)
        if chain_run_values["run"]:
            LOGGER.info(f"Proceeding with diff_test() for {run_key} as per config.")
            LOGGER.info(f"Loading adata from path: {chain_run_values['adata_path']}")
            adata=anndata.read_h5ad(Path(chain_run_values['adata_path']) )
            LOGGER.info(f"laoded adata: \n{adata}")
            LOGGER.info(adata.var.head(2))
            LOGGER.info(adata.obs.head(2)) # display the first 2 rows of the 
            ############ run diff_test() ############
            _diff_test_df=adtl.diff_test(
            adata,
            layer=chain_run_values.get('layer', None),
            groupby_key=chain_run_values.get('groupby_key','Treatment'),# 'Treatment',
            groupby_key_target_values=chain_run_values['groupby_key_target_values'],
            groupby_key_ref_values=chain_run_values['groupby_key_ref_values'],
            comparison_col_tag=chain_run_values['comparison_col_tag'],
            tests=chain_run_values['tests'],
            pair_by_key=chain_run_values.get('pair_by_key', None),
            add_values2results= chain_run_values.get('add_values2results', None),
            sortby=chain_run_values.get('sortby', None),
            ascending=chain_run_values.get('ascending', True),
            add_adata_var_column_key_list=chain_run_values.get('add_adata_var_column_key_list', None),
            save_table=G.SAVE_OUTPUT,
            logger=LOGGER,
            save_path=chain_run_values['save_path'],
            #save_result_to_adata_uns_as_dict=True,
            )
            LOGGER.info(_diff_test_df.shape) # print the shape of the results dataframe
            LOGGER.info(_diff_test_df.head(10)) # display the first 10 rows of the results dataframe

        else:
            LOGGER.info(f"Skipping diff_test() for {run_key} as per config.")


# ------------- diff_test()_runs  --------------------------------------------------------
