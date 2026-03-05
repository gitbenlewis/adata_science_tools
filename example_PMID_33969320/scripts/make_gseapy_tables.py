#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/general_dataset_template_private/scripts/make_gseapy_tables.py
# updated: 2026-03-04


####################################
import sys
import os
from pathlib import Path
from typing import Mapping, Sequence, Union, Optional
import pandas as pd
import gseapy as gp
import numpy as np
import glob
import logging
from datetime import datetime
import gseapy.utils as gutils
import gseapy.base as gbase
import gseapy.biomart as gbiomart
import gseapy.enrichr as genrich
 
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
OUTPUT_DIR = Path(CFG["GSEApy_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
####################################

GSEA_REPO_RESULTS_DIR= Path(CFG['GSEApy_params']['repo_results_dir'])
os.makedirs(GSEA_REPO_RESULTS_DIR, exist_ok=True)

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
root = logging.getLogger()
root.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) for h in root.handlers): # add file handler once
    for log_path in (RESULTS_LOG_FILE, SCRIPT_LOG_FILE):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        fh.setLevel(logging.INFO)
        root.addHandler(fh)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Logging to %s", RESULTS_LOG_FILE)
LOGGER.info("Logging to %s", SCRIPT_LOG_FILE)
# #) make gseapy loggers use our handlers
# file_handler = next(h for h in root.handlers if isinstance(h, logging.FileHandler))
file_handlers = []
_seen_log_paths = set()
for handler in root.handlers:
    if isinstance(handler, logging.FileHandler):
        log_path = getattr(handler, "baseFilename", None)
        if log_path in _seen_log_paths:
            continue
        _seen_log_paths.add(log_path)
        file_handlers.append(handler)
def log_init_with_file(name, log_level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    for handler in file_handlers:
        logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.propagate = True
    return logger
gutils.log_init = log_init_with_file
gbase.log_init = log_init_with_file
gbiomart.log_init = log_init_with_file
genrich.log_init = log_init_with_file
# -----END----- logging setup ---------- 


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




#### GSEApy DEFAULT PARAMS ##########################################################
'''
GSEA_FEATURE_NAME_COLUMN_NAME= CFG['GSEApy_params']['default_params']['feature_name_column_name']#'gene_name'
GSEA_PERMUTATION_NUM=  CFG['GSEApy_params']['default_params']['gsea_permutation_num'] # 10000
GSEA_MIN_SIZE = CFG['GSEApy_params']['default_params']['min_size'] # 10
GSEA_MAX_SIZE = CFG['GSEApy_params']['default_params']['max_size'] # 3000
GSEA_RESULTS_TABLE_MIN_GENES_MATCHED = CFG['GSEApy_params']['default_params']['results_table_min_genes_matched']# 0
GSEA_THREADS = CFG['GSEApy_params']['default_params']['threads']# 30
GSEA_SEED = CFG['GSEApy_params']['default_params']['seed']# 6  
GSEA_RUN_FILENAME_TAG= CFG['GSEApy_params']['default_params']['GSEA_run_filename_tag']#'GSEA'
GSEA_RANK_METRIC_CALC_FLAVOR= CFG['GSEApy_params']['default_params']['rank_metric_calc_flavor']#'signed_neg_log10_pvalue'
GSEA_L2FC_COLUMN_NAME= CFG['GSEApy_params']['default_params']['l2fc_column_name']#'log2FoldChange'
GSEA_PVALUE_COLUMN_NAME = CFG['GSEApy_params']['default_params']['pvalue_column_name']#'pvalue'
GENE_TERM_COLLECTIONS_2_USE= CFG['GSEApy_params']['default_params']['gene_term_collections_2_use']
'''

CFG_GSEA_PARAMS= CFG['GSEApy_params']
CFG_GSEA_PARAMS_DEFAULTS= CFG_GSEA_PARAMS['default_params']


####### GMT prep #########################
GENE_TERM_COLLECTIONS_2_USE= CFG_GSEA_PARAMS_DEFAULTS['gene_term_collections_2_use']
SET_OF_GMT_FILE_PATHS = []
LIST_OF_TERM_COLLECTION_NAMES= []
## loop over gmt file dirs
for gmt_file_dir in CFG['GSEApy_params']['default_params']['gene_term_collections_gmt_file_directory_list']:
    gmt_dir_path = Path(CFG['GSEApy_params']['default_params']['gene_term_collections_gmt_file_directory_list'][gmt_file_dir])
    LOGGER.info(f"Checking GMT file dir path: {gmt_dir_path}, exists: {gmt_dir_path.exists()}")
    gmt_file_paths= glob.glob(str(gmt_dir_path / '*.gmt'))
    LOGGER.info(f"Found {len(gmt_file_paths)} GMT files in {gmt_dir_path}: {gmt_file_paths}")
    kept_gmt_file_paths= [gmt_file_path for gmt_file_path in gmt_file_paths
                                if os.path.basename(gmt_file_path).split('.gmt')[0] in GENE_TERM_COLLECTIONS_2_USE]
    LOGGER.info(f"Keeping {len(kept_gmt_file_paths)} GMT files for {gmt_file_dir} term collections: {kept_gmt_file_paths}")
    SET_OF_GMT_FILE_PATHS += kept_gmt_file_paths
    ## #) now make LIST_OF_TERM_COLLECTION_NAMES
    kept_term_collection_names= [
        os.path.basename(gmt_file_path).split('.gmt')[0] for gmt_file_path in kept_gmt_file_paths ]
    kept_term_collection_names=[name.replace('.symbols','') for name in kept_term_collection_names] 
    LOGGER.info(f"Keeping {len(kept_term_collection_names)} term collection names for {gmt_file_dir} term collections: {kept_term_collection_names}")
    LIST_OF_TERM_COLLECTION_NAMES += kept_term_collection_names
####### end GMT prep #########################



####### end GMT prep #########################

if CFG['GSEApy_params']['make_gseapy_tables_run_control']['run_4_all_nf_differentialabundance_runs']:
    pass
if CFG['GSEApy_params']['make_gseapy_tables_run_control']['run_4_selected_nf_differentialabundance_runs']:
    pass
if CFG['GSEApy_params']['make_gseapy_tables_run_control']['run_defined_gseapy_runs']:
    gseapy_runs=  CFG['GSEApy_params']['defined_gseapy_runs']
else:
    gseapy_runs= {}

from collections import ChainMap
if __name__ == "__main__":

    GSEA_REPO_RESULTS_DIR= Path(CFG['GSEApy_params']['repo_results_dir'])
    os.makedirs(GSEA_REPO_RESULTS_DIR, exist_ok=True)

    for run_key, run_values in gseapy_runs.items():
        # 
        gseapy_run_name=run_values['gseapy_run_name']
        da_run_name=run_values['DA_run_name']
        data_csv_path=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"data_csv_path": None})['data_csv_path']
        file_base_name = os.path.basename(data_csv_path).replace('.csv','')
        full_output_dir = os.path.join(GSEA_REPO_RESULTS_DIR,da_run_name,
                                        gseapy_run_name)
        # 
        os.makedirs(full_output_dir, exist_ok=True)
        LOGGER.info(f"Processing file: {data_csv_path} with base name: {file_base_name}, and output dir: {full_output_dir}")
        rgw.run_gseapy_prerank_multiple_term_collections(
            data_csv_path=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"data_csv_path": None})['data_csv_path'],
            data_csv_path_sepstr=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"data_csv_path_sepstr": ","})['data_csv_path_sepstr'],
            feature_name_column_name=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"feature_name_column_name": "var_names"})['feature_name_column_name'],#
            list_of_term_collections=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"list_of_term_collections": SET_OF_GMT_FILE_PATHS})['list_of_term_collections'],#SET_OF_GMT_FILE_PATHS,
            list_of_term_collection_names=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"list_of_term_collection_names": LIST_OF_TERM_COLLECTION_NAMES})['list_of_term_collection_names'],#LIST_OF_TERM_COLLECTION_NAMES,
            rank_metric_calc_flavor=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"rank_metric_calc_flavor": "signed_neg_log10_pvalue"})['rank_metric_calc_flavor'],
            l2fc_column_name= ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"l2fc_column_name": "log2FoldChange"})['l2fc_column_name'],#run_values['l2fc_column_name'],
            pvalue_column_name=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"pvalue_column_name": "pvalue"})['pvalue_column_name'],#run_values['pvalue_column_name'],
            existing_rank_metric_column_name=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"existing_rank_metric_column_name": None})['existing_rank_metric_column_name'],#run_values.get('existing_rank_metric_column_name', None),
            permutation_num=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"permutation_num": 1000})['permutation_num'],#GSEA_PERMUTATION_NUM,
            min_size=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"min_size": 10})['min_size'],#GSEA_MIN_SIZE,
            max_size=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"max_size": 3000})['max_size'],#GSEA_MAX_SIZE,
            results_table_min_genes_matched=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"results_table_min_genes_matched": 1})['results_table_min_genes_matched'],#GSEA_RESULTS_TABLE_MIN_GENES_MATCHED,
            threads=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"threads": 10})['threads'],#GSEA_THREADS,
            seed=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"seed": 6})['seed'],#GSEA_SEED,
            file_base_name=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"file_base_name": file_base_name})['file_base_name'],#file_base_name,
            results_dir= ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"results_dir": full_output_dir})['results_dir'],#full_output_dir,
            GSEA_run_filename_tag=ChainMap(run_values,CFG_GSEA_PARAMS_DEFAULTS,{"GSEA_run_filename_tag": None})['GSEA_run_filename_tag'],#GSEA_RUN_FILENAME_TAG,
            )
        


    LOGGER.info(f"make_gsea_tables.py All done!")
