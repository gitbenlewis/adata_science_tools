#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/general_dataset_template_private/scripts/make_gseapy_dotplots.py
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
OUTPUT_DIR = Path(CFG["GSEApy_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
####################################

# ----------------- Configuration ----------------------------------------------------

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
print(f"Using adtl from {adtl.__file__}")
from code_library import run_GSEApy_wrapper as rgw
print(f"Using run_GSEApy_wrapper as rgw from {rgw.__file__}")


########################################################## import custom code libraries ################################################
# ------------- parameters --------------------------------------------------------


# Configuration ---------------------------------------------------------------------
#### paths
#inputs
GSEAPY_DOTPLOT_PARAMS= CFG['GSEApy_params']['gseapy_dotplot_params']
GSEAPY_RUN_DIR_LIST= GSEAPY_DOTPLOT_PARAMS['gseapy_run_dir_list']
TOP_TERMS_2_PLOT= GSEAPY_DOTPLOT_PARAMS.get('top_terms_2_plot', 20)
FIGSIZE= GSEAPY_DOTPLOT_PARAMS.get('figsize', (6,8))
GSEA_REPO_RESULTS_DIR= Path(CFG['GSEApy_params']['repo_results_dir'])

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    for gseapy_run_results_dir in GSEAPY_RUN_DIR_LIST:
        GSEA_RESULTS_DIR = Path(gseapy_run_results_dir)
        DOTPLOTS_DIR = GSEA_RESULTS_DIR / "dotplots"
        DOTPLOTS_DIR.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(GSEA_RESULTS_DIR.glob("*.csv"))
        if not csv_files:
            LOGGER.warning("No .csv files found in %s", GSEA_RESULTS_DIR)
            continue

        for csv_path in csv_files:
            if csv_path.name.endswith(".rnk_df.csv"):
                LOGGER.info("Skipping rank file %s", csv_path)
                continue
            output_path = DOTPLOTS_DIR / f"{csv_path.stem}.dotplot.png"
            LOGGER.info("Plotting dotplot for %s -> %s", csv_path, output_path)
            try:
                FIGSIZE=FIGSIZE or (6,8)
                rgw.dotplot_from_table(
                    csv_path,
                    title=f'{os.path.basename(csv_path).split(".", 1)[0]}\n {os.path.basename(csv_path).split(".", 1)[1].rsplit(".", 1)[0] or ""}',
                    top_term=TOP_TERMS_2_PLOT,
                    ofname=str(output_path),
                    figsize=FIGSIZE,
                )
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping %s due to missing/invalid columns: %s", csv_path, exc)
            except Exception as exc:
                LOGGER.warning("Skipping %s due to unexpected error: %s", csv_path, exc)
            finally:
                plt.close("all")

