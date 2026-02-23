#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_PMID_33969320/scripts/make_volcano_plots.py

import sys
import os
from pathlib import Path
import pandas as pd
import anndata
from dataclasses import dataclass
from datetime import datetime
import logging
import yaml

# config
####################################
from pathlib import Path
import yaml
REPO_CONFIG_YAML_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with REPO_CONFIG_YAML_PATH.open() as f:
    CFG = yaml.safe_load(f)
# out and log path 
VOLCANO_CFG = CFG.get("volcano_plot_params", {})
OUTPUT_DIR = Path(
    VOLCANO_CFG.get(
        "repo_results_dir",
        Path(__file__).resolve().parent.parent / "results" / "figures" / "volcano_plots",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 

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

#### paths and config dictionaries
ANNOTATED_ADATA_OUTPUT_H5AD_PATH= Path(CFG['make_annotated_adata_params']['annotate_adata_runs']['input']['annotated_adata_output_h5ad_path'])
VOLCANO_PLOT_PARAMS = CFG.get('volcano_plot_params', {})
VOLCANO_PLOT_RUNS_PARAMS = VOLCANO_PLOT_PARAMS.get('volcano_plot_runs') or {}
FIGURES_OUTPUT_DIR= Path(VOLCANO_PLOT_PARAMS.get('repo_results_dir', OUTPUT_DIR))
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOLCANO_PLOT_DEFAULTS = VOLCANO_PLOT_PARAMS.get("defaults_params", {})

# Configuration ---------------------------------------------------------------------
#### paths
#inputs
metadata_table_path = CFG.get('dataset_level_config', {}).get('sample_sheet')
METADATA_TABLE = Path(metadata_table_path) if metadata_table_path else None
adata_h5ad_path = CFG.get('post_process_nfcore_rnaseq_parameters', {}).get('h5ad_out_path')
ADATA_H5AD_PATH = Path(adata_h5ad_path) if adata_h5ad_path else None
adata_ann_h5ad_path = CFG.get('post_process_nfcore_rnaseq_parameters', {}).get('annotated_h5ad_out_path')
ADATA_ANN_H5AD_PATH = Path(adata_ann_h5ad_path) if adata_ann_h5ad_path else None

# get from config later
DA_RESULTS_DIR=Path('/home/ubuntu/projects/ActioBio/RNAseq_repos/RNAseq_repo_TRPV4_mouse_bladder_Vizzard/results/differentialabundance/DA_filtered/')

# ----------------- Configuration ----------------------------------------------------
# ------------- parameters end --------------------------------------------------------




if __name__ == "__main__":
    from collections import ChainMap
    import anndata
    LOGGER.info("Starting make_volcano_plots.py script.")
    LOGGER.info("Loading adata from %s", ANNOTATED_ADATA_OUTPUT_H5AD_PATH)# ADATA_H5AD_PATH # ADATA_ANN_H5AD_PATH
    adata = anndata.read_h5ad(ANNOTATED_ADATA_OUTPUT_H5AD_PATH)
    LOGGER.info(f"adata loaded  {adata}")
    LOGGER.info(f"adata loaded with {adata.n_obs} cells and {adata.n_vars} genes.")
    LOGGER.info(f"adata loaded adata.var with columns {adata.var.columns} genes.")


    # -------------  VOLCANO PLOTS RUNS--------------------------------------------------------
    LOGGER.info(f"plotting volcano plots now...")
    
    for plot_key, plot_params in VOLCANO_PLOT_RUNS_PARAMS.items():
        chained_params = ChainMap(plot_params, VOLCANO_PLOT_DEFAULTS)
        if chained_params.get("run", False):
            LOGGER.info(f" run name {plot_key}")
            plot_params = VOLCANO_PLOT_RUNS_PARAMS[plot_key]
            ####
            if table_csv_path := chained_params.get("table_csv_path", None):
                LOGGER.info(f"Loading table from path: {table_csv_path}")
                var_filtered_df = pd.read_csv(table_csv_path, index_col=0)
                LOGGER.info(f"Table loaded with shape {var_filtered_df.shape} and columns {var_filtered_df.columns.values.tolist()}")
                LOGGER.info(f"Filtering table for non-null values in pvalue column: {chained_params['pvalue_col']}")
                LOGGER.info(f"PRE filter table shape: {var_filtered_df.shape}")
                var_filtered_df = var_filtered_df[~var_filtered_df[chained_params["pvalue_col"]].isnull()].copy()
                LOGGER.info(f"POST filter table shape: {var_filtered_df.shape}")
            elif chained_params.get("adata_h5ad_path", None):
                LOGGER.info(f"No table_csv_path provided for {plot_key}, loading adata and filtering var for pvalue column.")
                adata_path=chained_params.get("adata_h5ad_path", ANNOTATED_ADATA_OUTPUT_H5AD_PATH)
                LOGGER.info(f"Loading adata from path: {adata_path}")
                adata = anndata.read_h5ad(adata_path)
                LOGGER.info(f"adata loaded  {adata}")
                LOGGER.info(f"adata.var] columns: {adata.var.columns.values.tolist()}")
                LOGGER.info(f"PRE filter adata.var.shape: {adata.var.shape}")
                adata_filtered = adata[:,~(adata.var[plot_params["pvalue_col"]].isnull())].copy()
                LOGGER.info(f"POST filter adata.var.shape: {adata_filtered.var.shape}")
                var_filtered_df=adata_filtered.var.copy()
                LOGGER.info(f"var_filtered_df shape: {var_filtered_df.shape}")
                LOGGER.info(f"var_filtered_df head:\n{var_filtered_df.head(2)}")
            else:
                LOGGER.error(f"No valid input path provided for {plot_key}. Skipping this plot.")
                continue
            #####
            # print number of genes with pvalue < pvalue_threshold
            pvalue_threshold=chained_params.get("pvalue_threshold", 0.2)
            n_significant_genes=(var_filtered_df[chained_params["pvalue_col"]] < pvalue_threshold).sum()
            LOGGER.info(f"Number of genes with {chained_params['pvalue_col']} < {pvalue_threshold}: {n_significant_genes}")
            # print number of genes with abs(log2FoldChange) > log2FoldChange_threshold
            log2fc_threshold=chained_params.get("log2FoldChange_threshold", 0.5)
            n_fc_genes=(var_filtered_df[chained_params["l2fc_col"]].abs() > log2fc_threshold).sum()
            LOGGER.info(f"Number of genes with abs({chained_params['l2fc_col']}) > {log2fc_threshold}: {n_fc_genes}")
            # print number of genes with both pvalue and log2FoldChange thresholds
            n_both_genes=((var_filtered_df[chained_params["pvalue_col"]] < pvalue_threshold) & (var_filtered_df[chained_params["l2fc_col"]].abs() > log2fc_threshold)).sum()
            LOGGER.info(f"Number of genes with both {chained_params['pvalue_col']} < {pvalue_threshold} and abs({chained_params['l2fc_col']}) > {log2fc_threshold}: {n_both_genes}")
            # make volcano plot
            LOGGER.info(f"Making volcano plot for run_key: {plot_key}")
            # logg plot save path from chained_params["file_name"]
            LOGGER.info(f"Volcano plot will be saved to:\n {chained_params['file_name']}")
            if chained_params.get("comparison_column_key", None) and chained_params.get("comparisons_to_keep", None):
                LOGGER.info(f"Filtering var_filtered_df for comparisons in column {chained_params['comparison_column_key']} to keep only {chained_params['comparisons_to_keep']}")
                LOGGER.info(f"Before filtering, var_filtered_df shape: {var_filtered_df.shape}")
                var_filtered_df = var_filtered_df[var_filtered_df[chained_params["comparison_column_key"]].isin(chained_params["comparisons_to_keep"])].copy()
                LOGGER.info(f"After filtering, var_filtered_df shape: {var_filtered_df.shape}")
            adtl.volcano_plot_generic(
                var_filtered_df,
                l2fc_col=chained_params.get("l2fc_col",VOLCANO_PLOT_DEFAULTS.get("l2fc_col","log2FoldChange")),
                set_xlabel=chained_params.get("set_xlabel",VOLCANO_PLOT_DEFAULTS.get("set_xlabel","log2FoldChange")),
                xlimit=chained_params.get("xlimit",VOLCANO_PLOT_DEFAULTS.get("xlimit",4)),
                pvalue_col=chained_params.get("pvalue_col",VOLCANO_PLOT_DEFAULTS.get("pvalue_col","pvalue")),
                set_ylabel=chained_params.get("set_ylabel",VOLCANO_PLOT_DEFAULTS.get("set_ylabel","-log10(pvalue)")),
                ylimit=chained_params.get("ylimit",VOLCANO_PLOT_DEFAULTS.get("ylimit",5)),
                title_text=chained_params.get("title_text",VOLCANO_PLOT_DEFAULTS.get("title_text","Volcano Plot")),
                comparison_label=chained_params["comparison_label"],
                title_fontsize=chained_params.get("title_fontsize",VOLCANO_PLOT_DEFAULTS.get("title_fontsize", 20)),
                axis_label_and_tick_fontsize=chained_params.get("axis_label_and_tick_fontsize",VOLCANO_PLOT_DEFAULTS.get("axis_label_and_tick_fontsize", 16)),
                legend_fontsize=chained_params.get("legend_fontsize",VOLCANO_PLOT_DEFAULTS.get("legend_fontsize", 14)),
                hue_column=chained_params.get("hue_column",VOLCANO_PLOT_DEFAULTS.get("hue_column",None)),
                log2FoldChange_threshold=chained_params.get("log2FoldChange_threshold",VOLCANO_PLOT_DEFAULTS.get("log2FoldChange_threshold", 0.5)),
                pvalue_threshold=chained_params.get("pvalue_threshold",VOLCANO_PLOT_DEFAULTS.get("pvalue_threshold", 0.2)),
                figsize=chained_params.get("figsize",VOLCANO_PLOT_DEFAULTS.get("figsize", (15, 10))),
                legend_bbox_to_anchor=chained_params.get("legend_bbox_to_anchor",VOLCANO_PLOT_DEFAULTS.get("legend_bbox_to_anchor", (1.15, 1))),
                label_top_features=chained_params.get("label_top_features",VOLCANO_PLOT_DEFAULTS.get("label_top_features", True)),
                only_label_hue_dots=chained_params.get("only_label_hue_dots",VOLCANO_PLOT_DEFAULTS.get("only_label_hue_dots", True)),
                feature_label_col=chained_params.get("feature_label_col",VOLCANO_PLOT_DEFAULTS.get("feature_label_col", "gene_name")),
                n_top_features=chained_params.get("n_top_features",VOLCANO_PLOT_DEFAULTS.get("n_top_features", 25)),
                label_top_features_fontsize=chained_params.get("label_top_features_fontsize",VOLCANO_PLOT_DEFAULTS.get("label_top_features_fontsize", 16)),
                label_features_char_limit=chained_params.get("label_features_char_limit",VOLCANO_PLOT_DEFAULTS.get("label_features_char_limit", None)),
                dot_size_shrink_factor=chained_params.get("dot_size_shrink_factor",VOLCANO_PLOT_DEFAULTS.get("dot_size_shrink_factor", 10)),
                savefig=G.SAVE_OUTPUT_FIGURES,
                file_name=chained_params["file_name"],
            )
            # ------------- r

    LOGGER.info(f"make_volcano_plots.py All done!")
