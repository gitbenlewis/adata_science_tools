#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/ActioBio/actio_general_dataset_template_Ben/scripts/make_parse_datasets.py
# updated: 2026-03-04
# ####################################
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
OUTPUT_DIR = Path(CFG["make_parse_datasets_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
###################################


 
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

# paths Configuration ---------------------------------------------------------------------
PARSE_DATASETS_CFG = CFG["make_parse_datasets_params"]
PARSE_DATASET_CALLS_DICT= CFG["make_parse_datasets_params"]["parse_dataset_calls"]
# paths Configuration ---------------------------------------------------------------------
 
import anndata as ad
import pandas as pd
import anndata as ad
from pathlib import Path


def _save_dataset(_adata: ad.AnnData, output_path: Path) -> None:
    """Save a parsed dataset as h5ad plus CSV exports."""
    h5ad_path = Path(f"{output_path}.h5ad")
    obs_csv_path = Path(f"{output_path}.obs.csv")
    var_csv_path = Path(f"{output_path}.var.csv")
    x_csv_path = Path(f"{output_path}.X.csv")
    LOGGER.info("Saving adata to %s", h5ad_path)
    _adata.write_h5ad(h5ad_path)
    LOGGER.info("Saving adata.obs to %s", obs_csv_path)
    _adata.obs.to_csv(obs_csv_path)
    LOGGER.info("Saving adata.var to %s", var_csv_path)
    _adata.var.to_csv(var_csv_path)
    LOGGER.info("Saving adata.X to %s", x_csv_path)
    x_data = _adata.X.toarray() if hasattr(_adata.X, "toarray") else _adata.X
    pd.DataFrame(x_data, index=_adata.obs_names, columns=_adata.var_names).to_csv(x_csv_path)
    for layer_name, layer_data in _adata.layers.items():
        safe_name = str(layer_name).replace("/", "_")
        layer_csv_path = Path(f"{output_path}.layer.{safe_name}.csv")
        LOGGER.info("Saving adata.layers['%s'] to %s", layer_name, layer_csv_path)
        layer_values = layer_data.toarray() if hasattr(layer_data, "toarray") else layer_data
        pd.DataFrame(layer_values, index=_adata.obs_names, columns=_adata.var_names).to_csv(layer_csv_path)
    LOGGER.info("Saved dataset to directory: %s", h5ad_path.parent)
    LOGGER.info("Saved dataset with base filename: %s", h5ad_path.stem)


def parse_datasets_filter_obs_boolean_column(parse_dataset_params):
    """Parse configured dataset subsets from a single input AnnData."""
    LOGGER.info("Starting make_parse_datasets.py script.")
    dataset_input_h5ad_path = Path(parse_dataset_params["dataset_input_h5ad_path"])
    adata = ad.read_h5ad(dataset_input_h5ad_path)
    LOGGER.info("laoded adata for parse_datasets.py: \n%s", adata)

    datasets_cfg = parse_dataset_params.get("datasets") or {}
    if not datasets_cfg:
        LOGGER.warning("No datasets configured under parse_dataset_params['datasets']; nothing to do.")
        return

    for dataset_key, dataset_cfg in datasets_cfg.items():
        output_dir = Path(dataset_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = str(dataset_cfg["filename"])
        output_path = output_dir / output_filename

        filter_obs_boolean_column = dataset_cfg.get("filter_obs_boolean_column")
        filter_obs_column_key = dataset_cfg.get("filter_obs_column_key")
        filter_obs_column_values_list = dataset_cfg.get("filter_obs_column_values_list")

        if filter_obs_boolean_column:
            adata_subset = adata[adata.obs[filter_obs_boolean_column] == True, :].copy()
            LOGGER.info(
                "Applied boolean obs filter for '%s': %s == True",
                dataset_key,
                filter_obs_boolean_column,
            )
            LOGGER.info("%s value counts:", filter_obs_boolean_column)
            LOGGER.info(adata_subset.obs[filter_obs_boolean_column].value_counts(dropna=False))
        elif filter_obs_column_key and filter_obs_column_values_list:
            adata_subset = adata[
                adata.obs[filter_obs_column_key].isin(filter_obs_column_values_list),
                :,
            ].copy()
            LOGGER.info(
                "Applied obs isin filter for '%s': %s in %s",
                dataset_key,
                filter_obs_column_key,
                filter_obs_column_values_list,
            )
            LOGGER.info("%s value counts:", filter_obs_column_key)
            LOGGER.info(adata_subset.obs[filter_obs_column_key].value_counts(dropna=False))
        else:
            adata_subset = adata.copy()
            LOGGER.info(
                "No obs filter configured for '%s'; using full dataset.",
                dataset_key,
            )
        if "Treatment" in adata_subset.obs.columns:
            LOGGER.info("Treatment value counts:")
            LOGGER.info(adata_subset.obs["Treatment"].value_counts(dropna=False))

        for col in adata_subset.obs.select_dtypes(["category"]).columns:
            adata_subset.obs[col] = adata_subset.obs[col].cat.remove_unused_categories()

        LOGGER.info("Saving %s dataset to %s", dataset_key, output_path)
        _save_dataset(adata_subset, output_path)




if __name__ == "__main__":
    PARSE_DATASET_PARAMS_INPUT=PARSE_DATASET_CALLS_DICT['input'] 
    parse_datasets_filter_obs_boolean_column(PARSE_DATASET_PARAMS_INPUT)
    
