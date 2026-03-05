#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/general_dataset_template_private/scripts/make_diff_test_tables.py
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


from collections import ChainMap


if __name__ == "__main__":
    LOGGER.info("Starting make_diff_test_tables.py script.")

    # diff_test()_runs ---------------------------------------------------------------------
    # SET DEFAULTS FOR diff_test() RUNS
    DICTIONARY_OF_DIFF_TEST_RUNS = CFG["make_diff_test_tables_params"]["diff_test_runs"]
    DEFAULT_PARAMS = CFG["make_diff_test_tables_params"]["default_params"]

    for run_key, run_values in DICTIONARY_OF_DIFF_TEST_RUNS.items():
        LOGGER.info("###################################################################################################")
        LOGGER.info("run_key: %s  with info: \n %s", run_key, run_values)
        LOGGER.info("run control set to: %s", run_values["run"])
        chain_run_values = ChainMap(run_values, DEFAULT_PARAMS)

        if chain_run_values["run"]:
            LOGGER.info("Proceeding with diff_test() for %s as per config.", run_key)
            LOGGER.info("Loading adata from path: %s", chain_run_values["adata_path"])
            adata = anndata.read_h5ad(Path(chain_run_values["adata_path"]))
            LOGGER.info("laoded adata: \n%s", adata)
            LOGGER.info(adata.var.head(2))
            LOGGER.info(adata.obs.head(2))  # display the first 2 rows

            _diff_test_df = adtl.diff_test(
                adata,
                layer=chain_run_values.get("layer", None),
                groupby_key=chain_run_values.get("groupby_key", "Treatment"),
                groupby_key_target_values=chain_run_values["groupby_key_target_values"],
                groupby_key_ref_values=chain_run_values["groupby_key_ref_values"],
                comparison_col_tag=chain_run_values["comparison_col_tag"],
                tests=chain_run_values["tests"],
                pair_by_key=chain_run_values.get("pair_by_key", None),
                add_values2results=chain_run_values.get("add_values2results", None),
                sortby=chain_run_values.get("sortby", None),
                ascending=chain_run_values.get("ascending", True),
                add_adata_var_column_key_list=chain_run_values.get("add_adata_var_column_key_list", None),
                save_table=G.SAVE_OUTPUT,
                logger=LOGGER,
                save_path=chain_run_values["save_path"],
                save_result_to_adata_uns_as_dict=chain_run_values.get("save_result_to_adata_uns_as_dict", False),
            )
            LOGGER.info(_diff_test_df.shape)  # print shape of results dataframe
            LOGGER.info(_diff_test_df.head(10))  # display first 10 rows of results dataframe
        else:
            LOGGER.info("Skipping diff_test() for %s as per config.", run_key)


# ------------- diff_test()_runs  --------------------------------------------------------
