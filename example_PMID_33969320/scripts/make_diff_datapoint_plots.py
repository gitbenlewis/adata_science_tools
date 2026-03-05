#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/general_dataset_template_private/scripts/make_diff_datapoint_plots.py
# updated: 2026-03-04 
import sys
import os
from pathlib import Path
import pandas as pd
from collections import ChainMap
from dataclasses import dataclass
from datetime import datetime
import logging
import yaml
import matplotlib.pyplot as plt
# CFG Configuration
####################################
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIG_YAML_PATH = REPO_ROOT / "config" / "config.yaml"
with REPO_CONFIG_YAML_PATH.open() as f:
    CFG = yaml.safe_load(f)

# out and log path 
OUTPUT_DIR = Path(CFG["diff_datapoint_plots_params"]["repo_results_dir"])
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
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
# ---------- logging setup ----------
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
ANNOTATED_ADATA_OUTPUT_H5AD_PATH= Path(CFG['make_annotated_adata_params']['annotate_adata_runs']['input']['annotated_adata_output_h5ad_path'])
DIFF_DATAPOINT_PLOTS_PARAMS=CFG['diff_datapoint_plots_params']
FIGURES_OUTPUT_DIR= Path(DIFF_DATAPOINT_PLOTS_PARAMS['repo_results_dir'])
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BARH_L2FC_DOTPLOT_COLUMN_CALLS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_l2fc_dotplot_column_calls') or {}
BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_l2fc_dotplot_column_calls_defaults') or {}

BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_dotplot_dotplot_column_calls') or {}
BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_dotplot_dotplot_column_calls_defaults') or {}

BARH_DOTPLOT_DOTPLOT_DOTPLOT_COLUMN_CALLS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_dotplot_dotplot_dotplot_column_calls') or {}
BARH_DOTPLOT_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_dotplot_dotplot_dotplot_column_calls_defaults') or {}

BARH_4X_DOTPLOT_COLUMN_CALLS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_4X_dotplot_column_calls') or {}
BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS = DIFF_DATAPOINT_PLOTS_PARAMS.get('barh_4X_dotplot_column_calls_defaults') or {}
# ------------- parameters end --------------------------------------------------------




if __name__ == "__main__":
    import anndata
    LOGGER.info("Starting make_diff_datapoint_plots.py script.")
    ##################### diff datapoint plots ##########################
    # -------------  diff datapoint plots RUNS--------------------------------------------------------

    for plot_key, plot_params in BARH_4X_DOTPLOT_COLUMN_CALLS.items():
        chained_plot_params = ChainMap(plot_params, BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS)
        if not chained_plot_params.get("run", False):
            LOGGER.info("Skipping barh_4X_dotplot_column '%s' because run=false", plot_key)
            continue
        plot_params = chained_plot_params
        LOGGER.info("starting barh_4X_dotplot_column calls: ")
        LOGGER.info("--------------------------------------------------")
        LOGGER.info(f"Starting diff datapoint plot for plot_key: {plot_key}")
        LOGGER.info(f"saving {plot_key} diff datapoint plot to file: {plot_params['file_name']}")
        adata_path=plot_params.get("adata_h5ad_path", )
        lessthan_filterby_col=plot_params.get("lessthan_filterby_col", None)
        lessthan_filterby_threshold=plot_params.get("lessthan_filterby_threshold", None)
        sortby_col=plot_params.get("sortby_col", None)
        ascending=plot_params.get("ascending", True)
        top_n_features=plot_params.get("top_n_features", 40)
        LOGGER.info(f"Loading adata from path: {adata_path}")
        adata = anndata.read_h5ad(adata_path)
        LOGGER.info(f"adata loaded  {adata}")
        LOGGER.info(f"adata.var] columns: {adata.var.columns.values.tolist()}")
        LOGGER.info(f" filtering adata.var by non-null values in column: {plot_params['dotplot_l2fc_vars_col_label']}")
        LOGGER.info(f"PRE filter adata.var.shape: {adata.var.shape}")
        adata_filtered = adata[:,~(adata.var[plot_params["dotplot_l2fc_vars_col_label"]].isnull())].copy()
        LOGGER.info(f"POST filter adata.var.shape: {adata_filtered.var.shape}")
        if not plot_params.get("feature_list", None):
            LOGGER.info(f"No feature_list provided in plot_params, will select top {top_n_features} features by {sortby_col}")
            var_mask_col=plot_params.get("var_mask_col", None)
            if var_mask_col:
                LOGGER.info(f"Filtering adata_filtered.var by True values in column: {var_mask_col}")
                adata_filtered = adata_filtered[:,adata_filtered.var[var_mask_col]].copy()
                LOGGER.info(f"POST mask filter adata.var.shape: {adata_filtered.var.shape}")
            if lessthan_filterby_col and lessthan_filterby_threshold is not None:
                LOGGER.info(f"Filtering adata_filtered.var by {lessthan_filterby_col} <= {lessthan_filterby_threshold}")
                adata_filtered = adata_filtered[:,adata_filtered.var[lessthan_filterby_col] <= lessthan_filterby_threshold].copy()
                LOGGER.info(f"POST filterby filter adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_startswith_str_to_filter_features", False):
                for prefix in plot_params.get("list_of_startswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes starting with '{prefix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.startswith(prefix)].copy()
                    LOGGER.info(f"POST remove {prefix} genes adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_endswith_str_to_filter_features", False):
                for suffix in plot_params.get("list_of_endswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes ending with '{suffix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.endswith(suffix)].copy()
                    LOGGER.info(f"POST remove {suffix} genes adata.var.shape: {adata_filtered.var.shape}")
            var_filtered_df=adata_filtered.var.copy()
            LOGGER.info(f"var_filtered_df shape: {var_filtered_df.shape}")
            LOGGER.info(f"{var_filtered_df.head(2)}")
            LOGGER.info(f"Sorting by column: {sortby_col}\nascending: {ascending}\ntop_n_features: {top_n_features}")
            var_filtered_df.sort_values(by=[sortby_col], ascending=[ascending], inplace=True)
            top_n_var_names=var_filtered_df.head(top_n_features).index.tolist()
            LOGGER.info(f"top_n_var_names: {top_n_var_names}")
            feature_list=top_n_var_names
        else:
            feature_list=plot_params.get("feature_list", None)
        LOGGER.info(f"Final feature_list length: {len(feature_list)}\nfeatures: {feature_list}")
        # re-order categorical variable if specified
        comparison_col=plot_params.get("comparison_col", None)
        desired_order=plot_params.get("comparison_col_order", None)
        if comparison_col and desired_order:
            LOGGER.info(f"Re-ordering categorical variable '{comparison_col}' to have order: {desired_order}")
            LOGGER.info(f"Before re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
            adata.obs[comparison_col] = pd.Categorical(adata.obs[comparison_col], categories=desired_order, ordered=True)
            LOGGER.info(f"After re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")

        adtl.barh_4X_dotplot_column(
            # shared parameters
            adata=adata,
            layer=plot_params.get('layer', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('layer', None)) or None,
            feature_list=feature_list,
            feature_label_vars_col=plot_params.get('feature_label_vars_col', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_vars_col')) or None,
            feature_label_char_limit=plot_params.get('feature_label_char_limit', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_char_limit')) or None,
            feature_label_x=plot_params.get('feature_label_x', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_x')) or -0.02,
            figsize=plot_params.get('figsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('figsize', (22, 25))) or (22, 25),
            fig_title=plot_params.get('fig_title', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title', 'fig_title\n')) or None,
            fig_title_y=plot_params.get('fig_title_y', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_y', .96)) or .99,
            subfig_title_y=plot_params.get('subfig_title_y', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_y', 0.91)) or 0.94,
            fig_title_fontsize=plot_params.get('fig_title_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_fontsize', 30)) or 30,
            subfig_title_fontsize=plot_params.get('subfig_title_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_fontsize', 24)) or 24,
            feature_label_fontsize=plot_params.get('feature_label_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_fontsize', 24)) or 24,
            tick_label_fontsize=plot_params.get('tick_label_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tick_label_fontsize', 16)) or 16,
            legend_fontsize=plot_params.get('legend_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('legend_fontsize', 20)) or 20,
            row_hspace=plot_params.get('row_hspace', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('row_hspace', 0.4)) or None,
            col_wspace=plot_params.get('col_wspace', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('col_wspace', -0.1)) or -0.1,
            bar_dotplot_width_ratios=plot_params.get(
                'bar_dotplot_width_ratios',
                BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('bar_dotplot_width_ratios', [1.5, 1.0, 1.0, 1.0, 1.0])
            ) or [1.5, 1.0, 1.0, 1.0, 1.0],
            tight_layout_rect_arg=plot_params.get('tight_layout_rect_arg', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tight_layout_rect_arg')) or [0, 0, 1, 1],  # [left, bottom, right, top]
            use_tight_layout=plot_params.get('use_tight_layout', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('use_tight_layout', False)) or False,
            savefig=G.SAVE_OUTPUT_FIGURES,
            file_name=plot_params.get('file_name', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('file_name', 'barh_4X_dotplot.png')) or 'barh_4X_dotplot.png',
            # barh specific parameters
            comparison_col=plot_params.get('comparison_col', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_col', 'Treatment')) or 'Treatment',
            barh_remove_yticklabels=plot_params.get('barh_remove_yticklabels', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_remove_yticklabels', True)) or True,
            comparison_order=plot_params.get('comparison_order', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_order', None)) or None,
            hue_palette_color_list=plot_params.get('hue_palette_color_list', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('hue_palette_color_list', None)) or None,
            barh_figure_plot_title=plot_params.get('barh_figure_plot_title', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_figure_plot_title', 'barh_figure_plot_title')) or 'Expression (TPM)',
            barh_subplot_xlabel=plot_params.get('barh_subplot_xlabel', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_subplot_xlabel', 'barh_subplot_xlabel')) or 'Expression (TPM)',
            barh_sharex=plot_params.get('barh_sharex', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_sharex', False)) or False,
            barh_set_xaxis_lims=plot_params.get('barh_set_xaxis_lims', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_set_xaxis_lims', None)) or None,
            barh_legend=plot_params.get('barh_legend', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend', True)) or True,
            barh_legend_bbox_to_anchor=plot_params.get('barh_legend_bbox_to_anchor', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend_bbox_to_anchor', (0.5, .01))) or (0.5, .01),
            # dotplot1 parameters
            dotplot_figure_plot_title=plot_params.get('dotplot_figure_plot_title', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_figure_plot_title', 'log2fc')) or 'log2fc',
            dotplot_pval_vars_col_label=plot_params.get('dotplot_pval_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_pval_vars_col_label', 'pvalue')) or 'pvalue',
            dotplot_l2fc_vars_col_label=plot_params.get('dotplot_l2fc_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_l2fc_vars_col_label', 'log2FoldChange')) or 'log2FoldChange',
            dotplot_subplot_xlabel=plot_params.get('dotplot_subplot_xlabel', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_subplot_xlabel', 'log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            pval_label=plot_params.get('pval_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pval_label', 'p-value')) or 'p-value',
            pvalue_cutoff_ring=plot_params.get('pvalue_cutoff_ring', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pvalue_cutoff_ring', 0.1)) or 0.1,
            sizes=plot_params.get('sizes', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('sizes', (20, 2000))) or (20, 2000),
            dotplot_sharex=plot_params.get('dotplot_sharex', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_sharex', True)) or True,
            dotplot_set_xaxis_lims=plot_params.get('dotplot_set_xaxis_lims', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_set_xaxis_lims', None)) or None,
            dotplot_legend=plot_params.get('dotplot_legend', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend', True)) or True,
            dotplot_legend_bins=plot_params.get('dotplot_legend_bins', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bins', 3)) or 3,
            dotplot_legend_bbox_to_anchor=plot_params.get('dotplot_legend_bbox_to_anchor', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bbox_to_anchor', (0.5, .02))) or (0.5, .02),
            dotplot_annotate=plot_params.get('dotplot_annotate', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate', True)) or True,
            dotplot_annotate_xy=plot_params.get('dotplot_annotate_xy', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_xy', (0.8, 1.2))) or (0.8, 1.2),
            dotplot_annotate_labels=plot_params.get('dotplot_annotate_labels', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot_annotate_fontsize=plot_params.get('dotplot_annotate_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_fontsize', None)) or None,
            # dotplot2 parameters
            dotplot2_figure_plot_title=plot_params.get('dotplot2_figure_plot_title', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_figure_plot_title', 'log2fc (2)')) or 'log2fc (2)',
            dotplot2_pval_vars_col_label=plot_params.get('dotplot2_pval_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_vars_col_label', 'pvalue_alt')) or 'pvalue_alt',
            dotplot2_l2fc_vars_col_label=plot_params.get('dotplot2_l2fc_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_l2fc_vars_col_label', 'log2FoldChange_alt')) or 'log2FoldChange_alt',
            dotplot2_subplot_xlabel=plot_params.get('dotplot2_subplot_xlabel', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_subplot_xlabel', 'log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            dotplot2_pval_label=plot_params.get('dotplot2_pval_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_label', 'p-value')) or 'p-value',
            dotplot2_pvalue_cutoff_ring=plot_params.get('dotplot2_pvalue_cutoff_ring', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pvalue_cutoff_ring', 0.1)) or 0.1,
            dotplot2_sizes=plot_params.get('dotplot2_sizes', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sizes', (20, 2000))) or (20, 2000),
            dotplot2_sharex=plot_params.get('dotplot2_sharex', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sharex', True)) or True,
            dotplot2_set_xaxis_lims=plot_params.get('dotplot2_set_xaxis_lims', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_set_xaxis_lims', None)) or None,
            dotplot2_legend=plot_params.get('dotplot2_legend', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend', True)) or True,
            dotplot2_legend_bins=plot_params.get('dotplot2_legend_bins', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bins', 3)) or 3,
            dotplot2_legend_bbox_to_anchor=plot_params.get('dotplot2_legend_bbox_to_anchor', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bbox_to_anchor', (0.5, -.01))) or (0.5, -.01),
            dotplot2_annotate=plot_params.get('dotplot2_annotate', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate', True)) or True,
            dotplot2_annotate_xy=plot_params.get('dotplot2_annotate_xy', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_xy', (0.8, 1.2))) or (0.8, 1.2),
            dotplot2_annotate_labels=plot_params.get('dotplot2_annotate_labels', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot2_annotate_fontsize=plot_params.get('dotplot2_annotate_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_fontsize', None)) or None,
            # dotplot3 parameters
            dotplot3_figure_plot_title=plot_params.get('dotplot3_figure_plot_title', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_figure_plot_title', 'log2fc (3)')) or 'log2fc (3)',
            dotplot3_pval_vars_col_label=plot_params.get('dotplot3_pval_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_pval_vars_col_label', 'pvalue_alt2')) or 'pvalue_alt2',
            dotplot3_l2fc_vars_col_label=plot_params.get('dotplot3_l2fc_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_l2fc_vars_col_label', 'log2FoldChange_alt2')) or 'log2FoldChange_alt2',
            dotplot3_subplot_xlabel=plot_params.get('dotplot3_subplot_xlabel', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_subplot_xlabel', 'log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            dotplot3_pval_label=plot_params.get('dotplot3_pval_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_pval_label', 'p-value')) or 'p-value',
            dotplot3_pvalue_cutoff_ring=plot_params.get('dotplot3_pvalue_cutoff_ring', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_pvalue_cutoff_ring', 0.1)) or 0.1,
            dotplot3_sizes=plot_params.get('dotplot3_sizes', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_sizes', (20, 2000))) or (20, 2000),
            dotplot3_sharex=plot_params.get('dotplot3_sharex', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_sharex', True)) or True,
            dotplot3_set_xaxis_lims=plot_params.get('dotplot3_set_xaxis_lims', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_set_xaxis_lims', None)) or None,
            dotplot3_legend=plot_params.get('dotplot3_legend', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_legend', True)) or True,
            dotplot3_legend_bins=plot_params.get('dotplot3_legend_bins', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_legend_bins', 3)) or 3,
            dotplot3_legend_bbox_to_anchor=plot_params.get('dotplot3_legend_bbox_to_anchor', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_legend_bbox_to_anchor', (0.5, -.01))) or (0.5, -.01),
            dotplot3_annotate=plot_params.get('dotplot3_annotate', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_annotate', True)) or True,
            dotplot3_annotate_xy=plot_params.get('dotplot3_annotate_xy', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_annotate_xy', (0.8, 1.2))) or (0.8, 1.2),
            dotplot3_annotate_labels=plot_params.get('dotplot3_annotate_labels', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot3_annotate_fontsize=plot_params.get('dotplot3_annotate_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot3_annotate_fontsize', None)) or None,
            # dotplot4 parameters
            dotplot4_figure_plot_title=plot_params.get('dotplot4_figure_plot_title', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_figure_plot_title', 'log2fc (4)')) or 'log2fc (4)',
            dotplot4_pval_vars_col_label=plot_params.get('dotplot4_pval_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_pval_vars_col_label', 'pvalue_alt3')) or 'pvalue_alt3',
            dotplot4_l2fc_vars_col_label=plot_params.get('dotplot4_l2fc_vars_col_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_l2fc_vars_col_label', 'log2FoldChange_alt3')) or 'log2FoldChange_alt3',
            dotplot4_subplot_xlabel=plot_params.get('dotplot4_subplot_xlabel', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_subplot_xlabel', 'log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            dotplot4_pval_label=plot_params.get('dotplot4_pval_label', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_pval_label', 'p-value')) or 'p-value',
            dotplot4_pvalue_cutoff_ring=plot_params.get('dotplot4_pvalue_cutoff_ring', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_pvalue_cutoff_ring', 0.1)) or 0.1,
            dotplot4_sizes=plot_params.get('dotplot4_sizes', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_sizes', (20, 2000))) or (20, 2000),
            dotplot4_sharex=plot_params.get('dotplot4_sharex', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_sharex', True)) or True,
            dotplot4_set_xaxis_lims=plot_params.get('dotplot4_set_xaxis_lims', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_set_xaxis_lims', None)) or None,
            dotplot4_legend=plot_params.get('dotplot4_legend', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_legend', True)) or True,
            dotplot4_legend_bins=plot_params.get('dotplot4_legend_bins', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_legend_bins', 3)) or 3,
            dotplot4_legend_bbox_to_anchor=plot_params.get('dotplot4_legend_bbox_to_anchor', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_legend_bbox_to_anchor', (0.5, -.01))) or (0.5, -.01),
            dotplot4_annotate=plot_params.get('dotplot4_annotate', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_annotate', True)) or True,
            dotplot4_annotate_xy=plot_params.get('dotplot4_annotate_xy', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_annotate_xy', (0.8, 1.2))) or (0.8, 1.2),
            dotplot4_annotate_labels=plot_params.get('dotplot4_annotate_labels', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot4_annotate_fontsize=plot_params.get('dotplot4_annotate_fontsize', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot4_annotate_fontsize', None)) or None,
            use_single_dotplot_colormap=plot_params.get('use_single_dotplot_colormap', BARH_4X_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('use_single_dotplot_colormap', False)),
        )
        plt.close("all")
    ##################################################################################
    for plot_key, plot_params in BARH_DOTPLOT_DOTPLOT_DOTPLOT_COLUMN_CALLS.items():
        chained_plot_params = ChainMap(plot_params, BARH_DOTPLOT_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS)
        if not chained_plot_params.get("run", False):
            LOGGER.info("Skipping barh_dotplot_dotplot_dotplot_column '%s' because run=false", plot_key)
            continue
        plot_params = chained_plot_params
        LOGGER.info("starting barh_dotplot_dotplot_dotplot_column calls: ")
        LOGGER.info("--------------------------------------------------")
        LOGGER.info(f"Starting diff datapoint plot for plot_key: {plot_key}")
        LOGGER.info(f"saving {plot_key} diff datapoint plot to file: {plot_params['file_name']}")
        adata_path=plot_params.get("adata_h5ad_path", )
        lessthan_filterby_col=plot_params.get("lessthan_filterby_col", None)
        lessthan_filterby_threshold=plot_params.get("lessthan_filterby_threshold", None)
        sortby_col=plot_params.get("sortby_col", None)
        ascending=plot_params.get("ascending", True)
        top_n_features=plot_params.get("top_n_features", 40)

        LOGGER.info(f"Loading adata from path: {adata_path}")
        adata = anndata.read_h5ad(adata_path)
        LOGGER.info(f"adata loaded  {adata}")
        LOGGER.info(f"adata.var] columns: {adata.var.columns.values.tolist()}")
        LOGGER.info(f" filtering adata.var by non-null values in column: {plot_params['dotplot_l2fc_vars_col_label']}")
        LOGGER.info(f"PRE filter adata.var.shape: {adata.var.shape}")
        adata_filtered = adata[:,~(adata.var[plot_params["dotplot_l2fc_vars_col_label"]].isnull())].copy()
        LOGGER.info(f"POST filter adata.var.shape: {adata_filtered.var.shape}")
        if not plot_params.get("feature_list", None):
            LOGGER.info(f"No feature_list provided in plot_params, will select top {top_n_features} features by {sortby_col}")
            var_mask_col=plot_params.get("var_mask_col", None)
            if var_mask_col:
                LOGGER.info(f"Filtering adata_filtered.var by True values in column: {var_mask_col}")
                adata_filtered = adata_filtered[:,adata_filtered.var[var_mask_col]].copy()
                LOGGER.info(f"POST mask filter adata.var.shape: {adata_filtered.var.shape}")
            if lessthan_filterby_col and lessthan_filterby_threshold is not None:
                LOGGER.info(f"Filtering adata_filtered.var by {lessthan_filterby_col} <= {lessthan_filterby_threshold}")
                adata_filtered = adata_filtered[:,adata_filtered.var[lessthan_filterby_col] <= lessthan_filterby_threshold].copy()
                LOGGER.info(f"POST filterby filter adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_startswith_str_to_filter_features", False):
                for prefix in plot_params.get("list_of_startswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes starting with '{prefix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.startswith(prefix)].copy()
                    LOGGER.info(f"POST remove {prefix} genes adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_endswith_str_to_filter_features", False):
                for suffix in plot_params.get("list_of_endswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes ending with '{suffix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.endswith(suffix)].copy()
                    LOGGER.info(f"POST remove {suffix} genes adata.var.shape: {adata_filtered.var.shape}")
            var_filtered_df=adata_filtered.var.copy()
            LOGGER.info(f"var_filtered_df shape: {var_filtered_df.shape}")
            LOGGER.info(f"{var_filtered_df.head(2)}")
            LOGGER.info(f"Sorting by column: {sortby_col}\nascending: {ascending}\ntop_n_features: {top_n_features}")
            var_filtered_df.sort_values(by=[sortby_col], ascending=[ascending], inplace=True)
            top_n_var_names=var_filtered_df.head(top_n_features).index.tolist()
            LOGGER.info(f"top_n_var_names: {top_n_var_names}")
            feature_list=top_n_var_names
        else:
            feature_list=plot_params.get("feature_list", None)
        LOGGER.info(f"Final feature_list length: {len(feature_list)}\nfeatures: {feature_list}")
        # re-order categorical variable if specified
        comparison_col=plot_params.get("comparison_col", None)
        desired_order=plot_params.get("comparison_col_order", None)
        if comparison_col and desired_order:
            LOGGER.info(f"Re-ordering categorical variable '{comparison_col}' to have order: {desired_order}")
            LOGGER.info(f"Before re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
            adata.obs[comparison_col] = pd.Categorical(adata.obs[comparison_col], categories=desired_order, ordered=True)
            LOGGER.info(f"After re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
        adtl.barh_dotplot_dotplot_dotplot_column(
            # shared parameters
            adata=adata,
            layer=plot_params.get('layer', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('layer',None)) or None,
            feature_list=feature_list,
            feature_label_vars_col=plot_params.get('feature_label_vars_col', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_vars_col')) or None,
            feature_label_char_limit=plot_params.get('feature_label_char_limit', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_char_limit')) or None,
            feature_label_x=plot_params.get('feature_label_x', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_x')) or -0.02,
            figsize=plot_params.get('figsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('figsize', (20, 25))) or (20, 25),
            fig_title=plot_params.get('fig_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title','fig_title\n')) or None,
            fig_title_y=plot_params.get('fig_title_y', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_y',.96)) or .99,
            subfig_title_y=plot_params.get('subfig_title_y', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_y',0.91)) or 0.94,
            fig_title_fontsize=plot_params.get('fig_title_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_fontsize',30)) or 30,
            subfig_title_fontsize=plot_params.get('subfig_title_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_fontsize',24)) or 24,
            feature_label_fontsize=plot_params.get('feature_label_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_fontsize',24)) or 24,
            tick_label_fontsize=plot_params.get('tick_label_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tick_label_fontsize',16)) or 16,
            legend_fontsize=plot_params.get('legend_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('legend_fontsize',20)) or 20,
            row_hspace=plot_params.get('row_hspace', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('row_hspace',0.4)) or None,
            col_wspace=plot_params.get('col_wspace', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('col_wspace',-0.1)) or -0.1,
            bar_dotplot_width_ratios= [1.5, 1.0, 1.0, 1.0],
            #bar_dotplot_width_ratios=plot_params.get('bar_dotplot_width_ratios', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('bar_dotplot_width_ratios',[1.5, 1.0, 1.0])) or [1.5, 1.0, 1.0],
            tight_layout_rect_arg=plot_params.get('tight_layout_rect_arg', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tight_layout_rect_arg')) or [0, 0, 1, 1], # [left, bottom, right, top]
            use_tight_layout=plot_params.get('use_tight_layout', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('use_tight_layout', False)) or False,
            savefig=G.SAVE_OUTPUT_FIGURES,
            file_name=plot_params.get('file_name', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('file_name','barh_dotplot_dotplot.png')) or 'barh_dotplot_dotplot.png',
            # barh specific parameters
            comparison_col=plot_params.get('comparison_col', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_col','Treatment')) or 'Treatment',
            barh_remove_yticklabels=plot_params.get('barh_remove_yticklabels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_remove_yticklabels',True)) or True,
            comparison_order=plot_params.get('comparison_order', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_order',None)) or None,
            hue_palette_color_list=plot_params.get('hue_palette_color_list', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('hue_palette_color_list', None)) or None,
            barh_figure_plot_title=plot_params.get('barh_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_figure_plot_title','barh_figure_plot_title')) or 'Expression (TPM)',
            barh_subplot_xlabel=plot_params.get('barh_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_subplot_xlabel','barh_subplot_xlabel')) or 'Expression (TPM)',
            barh_sharex=plot_params.get('barh_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_sharex',False)) or False,
            barh_set_xaxis_lims=plot_params.get('barh_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_set_xaxis_lims',None)) or None,
            barh_legend=plot_params.get('barh_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend',True)) or True,
            barh_legend_bbox_to_anchor=plot_params.get('barh_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend_bbox_to_anchor',(0.5, .01))) or (0.5, .01),
            # dotplot1 parameters (match barh_l2fc_dotplot_column)
            dotplot_figure_plot_title=plot_params.get('dotplot_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_figure_plot_title','log2fc')) or 'log2fc',
            dotplot_pval_vars_col_label=plot_params.get('dotplot_pval_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_pval_vars_col_label','pvalue')) or 'pvalue',
            dotplot_l2fc_vars_col_label=plot_params.get('dotplot_l2fc_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_l2fc_vars_col_label','log2FoldChange')) or 'log2FoldChange',
            dotplot_subplot_xlabel=plot_params.get('dotplot_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_subplot_xlabel','log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            pval_label=plot_params.get('pval_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pval_label','p-value')) or 'p-value',
            pvalue_cutoff_ring=plot_params.get('pvalue_cutoff_ring', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pvalue_cutoff_ring',0.1)) or 0.1,
            sizes=plot_params.get('sizes', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('sizes',(20, 2000))) or (20, 2000),
            dotplot_sharex=plot_params.get('dotplot_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_sharex',True)) or True,
            dotplot_set_xaxis_lims=plot_params.get('dotplot_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_set_xaxis_lims',None)) or None,
            dotplot_legend=plot_params.get('dotplot_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend',True)) or True,
            dotplot_legend_bins=plot_params.get('dotplot_legend_bins', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bins',3)) or 3,
            dotplot_legend_bbox_to_anchor=plot_params.get('dotplot_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bbox_to_anchor',(0.5, .02))) or (0.5, .02),
            dotplot_annotate=plot_params.get('dotplot_annotate', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate',True)) or True,
            dotplot_annotate_xy=plot_params.get('dotplot_annotate_xy', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_xy',(0.8, 1.2))) or (0.8, 1.2),
            dotplot_annotate_labels=plot_params.get('dotplot_annotate_labels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot_annotate_fontsize=plot_params.get('dotplot_annotate_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_fontsize',None)) or None,
            # dotplot2 parameters (alt)
            dotplot2_figure_plot_title=plot_params.get('dotplot2_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_figure_plot_title','log2fc (2)')) or 'log2fc (2)',
            dotplot2_pval_vars_col_label=plot_params.get('dotplot2_pval_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_vars_col_label','pvalue_alt')) or 'pvalue_alt',
            dotplot2_l2fc_vars_col_label=plot_params.get('dotplot2_l2fc_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_l2fc_vars_col_label','log2FoldChange_alt')) or 'log2FoldChange_alt',
            dotplot2_subplot_xlabel=plot_params.get('dotplot2_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_subplot_xlabel','log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            dotplot2_pval_label=plot_params.get('dotplot2_pval_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_label','p-value')) or 'p-value',
            dotplot2_pvalue_cutoff_ring=plot_params.get('dotplot2_pvalue_cutoff_ring', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pvalue_cutoff_ring',0.1)) or 0.1,
            dotplot2_sizes=plot_params.get('dotplot2_sizes', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sizes',(20, 2000))) or (20, 2000),
            dotplot2_sharex=plot_params.get('dotplot2_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sharex',True)) or True,
            dotplot2_set_xaxis_lims=plot_params.get('dotplot2_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_set_xaxis_lims',None)) or None,
            dotplot2_legend=plot_params.get('dotplot2_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend',True)) or True,
            dotplot2_legend_bins=plot_params.get('dotplot2_legend_bins', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bins',3)) or 3,
            dotplot2_legend_bbox_to_anchor=plot_params.get('dotplot2_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bbox_to_anchor',(0.5, -.01))) or (0.5, -.01),
            dotplot2_annotate=plot_params.get('dotplot2_annotate', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate',True)) or True,
            dotplot2_annotate_xy=plot_params.get('dotplot2_annotate_xy', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_xy',(0.8, 1.2))) or (0.8, 1.2),
            dotplot2_annotate_labels=plot_params.get('dotplot2_annotate_labels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot2_annotate_fontsize=plot_params.get('dotplot2_annotate_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_fontsize',None)) or None,
            # dotplot3 parameters (alt)
            dotplot3_figure_plot_title=plot_params.get('dotplot3_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_figure_plot_title','log2fc (2)')) or 'log2fc (2)',
            dotplot3_pval_vars_col_label=plot_params.get('dotplot3_pval_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_vars_col_label','pvalue_alt')) or 'pvalue_alt',
            dotplot3_l2fc_vars_col_label=plot_params.get('dotplot3_l2fc_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_l2fc_vars_col_label','log2FoldChange_alt')) or 'log2FoldChange_alt',
            dotplot3_subplot_xlabel=plot_params.get('dotplot3_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_subplot_xlabel','log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            dotplot3_pval_label=plot_params.get('dotplot3_pval_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_label','p-value')) or 'p-value',
            dotplot3_pvalue_cutoff_ring=plot_params.get('dotplot3_pvalue_cutoff_ring', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pvalue_cutoff_ring',0.1)) or 0.1,
            dotplot3_sizes=plot_params.get('dotplot3_sizes', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sizes',(20, 2000))) or (20, 2000),
            dotplot3_sharex=plot_params.get('dotplot3_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sharex',True)) or True,
            dotplot3_set_xaxis_lims=plot_params.get('dotplot3_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_set_xaxis_lims',None)) or None,
            dotplot3_legend=plot_params.get('dotplot3_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend',True)) or True,
            dotplot3_legend_bins=plot_params.get('dotplot3_legend_bins', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bins',3)) or 3,
            dotplot3_legend_bbox_to_anchor=plot_params.get('dotplot3_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bbox_to_anchor',(0.5, -.01))) or (0.5, -.01),
            dotplot3_annotate=plot_params.get('dotplot3_annotate', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate',True)) or True,
            dotplot3_annotate_xy=plot_params.get('dotplot3_annotate_xy', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_xy',(0.8, 1.2))) or (0.8, 1.2),
            dotplot3_annotate_labels=plot_params.get('dotplot3_annotate_labels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot3_annotate_fontsize=plot_params.get('dotplot3_annotate_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_fontsize',None)) or None,
            )
        plt.close("all")
##################################################### barh_l2fc_dotplot_column_calls ##########################

    # barh_dotplot_dotplot_column_calls RUNS--------------------------------------------------------
    # barh_dotplot_dotplot_column_calls ##########################
    LOGGER.info(f" starting adtl.barh_dotplot_dotplot_column_calls(...) calls for diff datapoint plots")
    
  
    for plot_key, plot_params in BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS.items():
        chained_plot_params = ChainMap(plot_params, BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS)
        if not chained_plot_params.get("run", False):
            LOGGER.info("Skipping barh_dotplot_dotplot_column '%s' because run=false", plot_key)
            continue
        plot_params = chained_plot_params
        LOGGER.info("starting barh_dotplot_dotplot_column calls: ")
        LOGGER.info("--------------------------------------------------")
        LOGGER.info(f"Starting diff datapoint plot for plot_key: {plot_key}")
        # ------------- run_key=
        #run_key='Treatment_Drug_CYP_over_Vehicle_CYP'
        LOGGER.info(f" run name {plot_key}")
        adata_path=plot_params.get("adata_h5ad_path", ANNOTATED_ADATA_OUTPUT_H5AD_PATH)
        sortby_col=plot_params.get("sortby_col", None)
        ascending=plot_params.get("ascending", True)
        top_n_features=plot_params.get("top_n_features", 15)
        lessthan_filterby_col=plot_params.get("lessthan_filterby_col", None)
        lessthan_filterby_threshold=plot_params.get("lessthan_filterby_threshold", None)
        LOGGER.info(f"Loading adata from path: {adata_path}")
        adata = anndata.read_h5ad(adata_path)
        LOGGER.info(f"adata loaded  {adata}")
        LOGGER.info(f"adata.var] columns: {adata.var.columns.values.tolist()}")
        LOGGER.info(f" filtering adata.var by non-null values in column: {plot_params.get('dotplot_l2fc_vars_col_label', '')}")
        LOGGER.info(f"PRE filter adata.var.shape: {adata.var.shape}")
        adata_filtered = adata[:,~(adata.var[plot_params.get("dotplot_l2fc_vars_col_label", "")].isnull())].copy()
        LOGGER.info(f"POST filter adata.var.shape: {adata_filtered.var.shape}")
        if not plot_params.get("feature_list", None):
            LOGGER.info(f"No feature_list provided in plot_params, will select top {top_n_features} features by {sortby_col}")
            var_mask_col=plot_params.get("var_mask_col", None)
            if var_mask_col:
                LOGGER.info(f"Filtering adata_filtered.var by True values in column: {var_mask_col}")
                adata_filtered = adata_filtered[:,adata_filtered.var[var_mask_col]].copy()
                LOGGER.info(f"POST mask filter adata.var.shape: {adata_filtered.var.shape}")
            if lessthan_filterby_col and lessthan_filterby_threshold is not None:
                LOGGER.info(f"Filtering adata_filtered.var by {lessthan_filterby_col} <= {lessthan_filterby_threshold}")
                adata_filtered = adata_filtered[:,adata_filtered.var[lessthan_filterby_col] <= lessthan_filterby_threshold].copy()
                LOGGER.info(f"POST filterby filter adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_startswith_str_to_filter_features", False):
                for prefix in plot_params.get("list_of_startswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes starting with '{prefix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.startswith(prefix)].copy()
                    LOGGER.info(f"POST remove {prefix} genes adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_endswith_str_to_filter_features", False):
                for suffix in plot_params.get("list_of_endswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes ending with '{suffix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.endswith(suffix)].copy()
                    LOGGER.info(f"POST remove {suffix} genes adata.var.shape: {adata_filtered.var.shape}")
            var_filtered_df=adata_filtered.var.copy()
            LOGGER.info(f"var_filtered_df shape: {var_filtered_df.shape}")
            LOGGER.info(f"{var_filtered_df.head(2)}")
            LOGGER.info(f"Sorting by column: {sortby_col}\nascending: {ascending}\ntop_n_features: {top_n_features}")
            var_filtered_df.sort_values(by=[sortby_col], ascending=[ascending], inplace=True)
            top_n_var_names=var_filtered_df.head(top_n_features).index.tolist()
            LOGGER.info(f"top_n_var_names: {top_n_var_names}")
            feature_list=top_n_var_names
        else:
            feature_list=plot_params.get("feature_list", None)
        # re-order categorical variable if specified
        comparison_col=plot_params.get("comparison_col", None)
        desired_order=plot_params.get("comparison_col_order", None)
        if comparison_col and desired_order:
            LOGGER.info(f"Re-ordering categorical variable '{comparison_col}' to have order: {desired_order}")
            LOGGER.info(f"Before re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
            adata.obs[comparison_col] = pd.Categorical(adata.obs[comparison_col], categories=desired_order, ordered=True)
            LOGGER.info(f"After re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
        LOGGER.info(f"saving {plot_key} diff datapoint plot to file: {plot_params['file_name']}")
        adtl.barh_dotplot_dotplot_column(
            # shared parameters
            adata=adata,
            layer=plot_params.get('layer', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('layer',None)) or None,
            feature_list=feature_list,
            feature_label_vars_col=plot_params.get('feature_label_vars_col', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_vars_col')) or None,
            feature_label_char_limit=plot_params.get('feature_label_char_limit', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_char_limit')) or None,
            feature_label_x=plot_params.get('feature_label_x', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_x')) or -0.02,
            figsize=plot_params.get('figsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('figsize', (20, 25))) or (20, 25),
            fig_title=plot_params.get('fig_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title','fig_title\n')) or None,
            fig_title_y=plot_params.get('fig_title_y', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_y',.96)) or .99,
            subfig_title_y=plot_params.get('subfig_title_y', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_y',0.91)) or 0.94,
            fig_title_fontsize=plot_params.get('fig_title_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_fontsize',30)) or 30,
            subfig_title_fontsize=plot_params.get('subfig_title_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_fontsize',24)) or 24,
            feature_label_fontsize=plot_params.get('feature_label_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_fontsize',24)) or 24,
            tick_label_fontsize=plot_params.get('tick_label_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tick_label_fontsize',16)) or 16,
            legend_fontsize=plot_params.get('legend_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('legend_fontsize',20)) or 20,
            row_hspace=plot_params.get('row_hspace', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('row_hspace',0.4)) or None,
            col_wspace=plot_params.get('col_wspace', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('col_wspace',-0.1)) or -0.1,
            bar_dotplot_width_ratios=plot_params.get('bar_dotplot_width_ratios', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('bar_dotplot_width_ratios',[1.5, 1.0, 1.0])) or [1.5, 1.0, 1.0],
            tight_layout_rect_arg=plot_params.get('tight_layout_rect_arg', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tight_layout_rect_arg')) or [0, 0, 1, 1], # [left, bottom, right, top]
            use_tight_layout=plot_params.get('use_tight_layout', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('use_tight_layout', False)) or False,
            savefig=G.SAVE_OUTPUT_FIGURES,
            file_name=plot_params.get('file_name', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('file_name','barh_dotplot_dotplot.png')) or 'barh_dotplot_dotplot.png',
            # barh specific parameters
            comparison_col=plot_params.get('comparison_col', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_col','Treatment')) or 'Treatment',
            barh_remove_yticklabels=plot_params.get('barh_remove_yticklabels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_remove_yticklabels',True)) or True,
            comparison_order=plot_params.get('comparison_order', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_order',None)) or None,
            hue_palette_color_list=plot_params.get('hue_palette_color_list', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('hue_palette_color_list', None)) or None,
            barh_figure_plot_title=plot_params.get('barh_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_figure_plot_title','barh_figure_plot_title')) or 'Expression (TPM)',
            barh_subplot_xlabel=plot_params.get('barh_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_subplot_xlabel','barh_subplot_xlabel')) or 'Expression (TPM)',
            barh_sharex=plot_params.get('barh_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_sharex',False)) or False,
            barh_set_xaxis_lims=plot_params.get('barh_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_set_xaxis_lims',None)) or None,
            barh_legend=plot_params.get('barh_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend',True)) or True,
            barh_legend_bbox_to_anchor=plot_params.get('barh_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend_bbox_to_anchor',(0.5, .01))) or (0.5, .01),
            # dotplot1 parameters (match barh_l2fc_dotplot_column)
            dotplot_figure_plot_title=plot_params.get('dotplot_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_figure_plot_title','log2fc')) or 'log2fc',
            dotplot_pval_vars_col_label=plot_params.get('dotplot_pval_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_pval_vars_col_label','pvalue')) or 'pvalue',
            dotplot_l2fc_vars_col_label=plot_params.get('dotplot_l2fc_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_l2fc_vars_col_label','log2FoldChange')) or 'log2FoldChange',
            dotplot_subplot_xlabel=plot_params.get('dotplot_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_subplot_xlabel','log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            pval_label=plot_params.get('pval_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pval_label','p-value')) or 'p-value',
            pvalue_cutoff_ring=plot_params.get('pvalue_cutoff_ring', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pvalue_cutoff_ring',0.1)) or 0.1,
            sizes=plot_params.get('sizes', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('sizes',(20, 2000))) or (20, 2000),
            dotplot_sharex=plot_params.get('dotplot_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_sharex',True)) or True,
            dotplot_set_xaxis_lims=plot_params.get('dotplot_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_set_xaxis_lims',None)) or None,
            dotplot_legend=plot_params.get('dotplot_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend',True)) or True,
            dotplot_legend_bins=plot_params.get('dotplot_legend_bins', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bins',3)) or 3,
            dotplot_legend_bbox_to_anchor=plot_params.get('dotplot_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bbox_to_anchor',(0.5, .02))) or (0.5, .02),
            dotplot_annotate=plot_params.get('dotplot_annotate', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate',True)) or True,
            dotplot_annotate_xy=plot_params.get('dotplot_annotate_xy', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_xy',(0.8, 1.2))) or (0.8, 1.2),
            dotplot_annotate_labels=plot_params.get('dotplot_annotate_labels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot_annotate_fontsize=plot_params.get('dotplot_annotate_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_fontsize',None)) or None,
            # dotplot2 parameters (alt)
            dotplot2_figure_plot_title=plot_params.get('dotplot2_figure_plot_title', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_figure_plot_title','log2fc (2)')) or 'log2fc (2)',
            dotplot2_pval_vars_col_label=plot_params.get('dotplot2_pval_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_vars_col_label','pvalue_alt')) or 'pvalue_alt',
            dotplot2_l2fc_vars_col_label=plot_params.get('dotplot2_l2fc_vars_col_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_l2fc_vars_col_label','log2FoldChange_alt')) or 'log2FoldChange_alt',
            dotplot2_subplot_xlabel=plot_params.get('dotplot2_subplot_xlabel', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_subplot_xlabel','log2fc ((target)/(ref))')) or 'log2fc ((target)/(ref))',
            dotplot2_pval_label=plot_params.get('dotplot2_pval_label', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pval_label','p-value')) or 'p-value',
            dotplot2_pvalue_cutoff_ring=plot_params.get('dotplot2_pvalue_cutoff_ring', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_pvalue_cutoff_ring',0.1)) or 0.1,
            dotplot2_sizes=plot_params.get('dotplot2_sizes', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sizes',(20, 2000))) or (20, 2000),
            dotplot2_sharex=plot_params.get('dotplot2_sharex', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_sharex',True)) or True,
            dotplot2_set_xaxis_lims=plot_params.get('dotplot2_set_xaxis_lims', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_set_xaxis_lims',None)) or None,
            dotplot2_legend=plot_params.get('dotplot2_legend', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend',True)) or True,
            dotplot2_legend_bins=plot_params.get('dotplot2_legend_bins', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bins',3)) or 3,
            dotplot2_legend_bbox_to_anchor=plot_params.get('dotplot2_legend_bbox_to_anchor', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_legend_bbox_to_anchor',(0.5, -.01))) or (0.5, -.01),
            dotplot2_annotate=plot_params.get('dotplot2_annotate', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate',True)) or True,
            dotplot2_annotate_xy=plot_params.get('dotplot2_annotate_xy', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_xy',(0.8, 1.2))) or (0.8, 1.2),
            dotplot2_annotate_labels=plot_params.get('dotplot2_annotate_labels', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_labels', ('l2fc: ', 'p:'))) or ('l2fc: ', 'p:'),
            dotplot2_annotate_fontsize=plot_params.get('dotplot2_annotate_fontsize', BARH_DOTPLOT_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot2_annotate_fontsize',None)) or None,
            )
        plt.close("all")
    ##################################################### barh_dotplot_dotplot_column_calls ##########################





    # barh_l2fc_dotplot_column_calls RUNS--------------------------------------------------------
    # barh_l2fc_dotplot_column_calls ##########################
    LOGGER.info(f" starting adtl.barh_l2fc_dotplot_column(...) calls for diff datapoint plots")
    
    for plot_key, plot_params in BARH_L2FC_DOTPLOT_COLUMN_CALLS.items():
        chained_plot_params = ChainMap(plot_params, BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS)
        if not chained_plot_params.get("run", False):
            LOGGER.info("Skipping barh_l2fc_dotplot_column '%s' because run=false", plot_key)
            continue
        plot_params = chained_plot_params
        # ------------- run_key=
        #run_key='Treatment_Drug_CYP_over_Vehicle_CYP'
        LOGGER.info(f" run name {plot_key}")
        adata_path=plot_params.get("adata_h5ad_path", ANNOTATED_ADATA_OUTPUT_H5AD_PATH)
        lessthan_filterby_col=plot_params.get("lessthan_filterby_col", None)
        lessthan_filterby_threshold=plot_params.get("lessthan_filterby_threshold", None)
        sortby_col=plot_params.get("sortby_col", None)
        ascending=plot_params.get("ascending", True)
        top_n_features=plot_params.get("top_n_features", 15)
        LOGGER.info(f"Loading adata from path: {adata_path}")
        adata = anndata.read_h5ad(adata_path)
        LOGGER.info(f"adata loaded  {adata}")
        LOGGER.info(f"adata.var] columns: {adata.var.columns.values.tolist()}")
        LOGGER.info(f" filtering adata.var by non-null values in column: {plot_params.get('dotplot_l2fc_vars_col_label', '')}")
        LOGGER.info(f"PRE filter adata.var.shape: {adata.var.shape}")
        adata_filtered = adata[:,~(adata.var[plot_params.get("dotplot_l2fc_vars_col_label")].isnull())].copy()
        LOGGER.info(f"POST filter adata.var.shape: {adata_filtered.var.shape}")

        if not plot_params.get("feature_list", None):
            LOGGER.info(f"No feature_list provided in plot_params, will select top {top_n_features} features by {sortby_col}")
            var_mask_col=plot_params.get("var_mask_col", None)
            if var_mask_col:
                LOGGER.info(f"Filtering adata_filtered.var by True values in column: {var_mask_col}")
                adata_filtered = adata_filtered[:,adata_filtered.var[var_mask_col]].copy()
                LOGGER.info(f"POST mask filter adata.var.shape: {adata_filtered.var.shape}")
            if lessthan_filterby_col and lessthan_filterby_threshold is not None:
                LOGGER.info(f"Filtering adata_filtered.var by {lessthan_filterby_col} <= {lessthan_filterby_threshold}")
                adata_filtered = adata_filtered[:,adata_filtered.var[lessthan_filterby_col] <= lessthan_filterby_threshold].copy()
                LOGGER.info(f"POST filterby filter adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_startswith_str_to_filter_features", False):
                for prefix in plot_params.get("list_of_startswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes starting with '{prefix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.startswith(prefix)].copy()
                    LOGGER.info(f"POST remove {prefix} genes adata.var.shape: {adata_filtered.var.shape}")
            if plot_params.get("list_of_endswith_str_to_filter_features", False):
                for suffix in plot_params.get("list_of_endswith_str_to_filter_features", []):
                    LOGGER.info(f"Removing genes ending with '{suffix}' from adata_filtered.var")
                    adata_filtered = adata_filtered[:,~adata_filtered.var.index.str.endswith(suffix)].copy()
                    LOGGER.info(f"POST remove {suffix} genes adata.var.shape: {adata_filtered.var.shape}")
            var_filtered_df=adata_filtered.var.copy()
            LOGGER.info(f"var_filtered_df shape: {var_filtered_df.shape}")
            LOGGER.info(f"{var_filtered_df.head(2)}")
            LOGGER.info(f"Sorting by column: {sortby_col}\nascending: {ascending}\ntop_n_features: {top_n_features}")
            var_filtered_df.sort_values(by=[sortby_col], ascending=[ascending], inplace=True)
            top_n_var_names=var_filtered_df.head(top_n_features).index.tolist()
            LOGGER.info(f"top_n_var_names: {top_n_var_names}")
            feature_list=top_n_var_names
        else:
            feature_list=plot_params.get("feature_list", None)
        # re-order categorical variable if specified
        comparison_col=plot_params.get("comparison_col", None)
        desired_order=plot_params.get("comparison_col_order", None)
        if comparison_col and desired_order:
            LOGGER.info(f"Re-ordering categorical variable '{comparison_col}' to have order: {desired_order}")
            LOGGER.info(f"Before re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
            adata.obs[comparison_col] = pd.Categorical(adata.obs[comparison_col], categories=desired_order, ordered=True)
            LOGGER.info(f"After re-ordering, categories: {adata.obs[comparison_col].cat.categories.tolist()}")
        LOGGER.info(f"saving {plot_key} diff datapoint plot to file: {plot_params['file_name']}")
        adtl.barh_l2fc_dotplot_column(
                adata, # type: ignore
                layer=plot_params.get('layer', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('layer')) or None,
                feature_list=feature_list,
                feature_label_vars_col=plot_params.get('feature_label_vars_col', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_vars_col')) or None,
                feature_label_char_limit=plot_params.get('feature_label_char_limit', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_char_limit',30)) or None,#30,
                feature_label_x=plot_params.get('feature_label_x', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_x',-0.02)) or None,#-0.02,
                figsize=plot_params.get('figsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('figsize',(15, 25))) or None,#(15, 25),
                fig_title=plot_params.get('fig_title', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title','fig_title\n')) or None,
                fig_title_y=plot_params.get('fig_title_y', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_y',0.96)) or None,#0.96,
                subfig_title_y=plot_params.get('subfig_title_y', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_y',0.91)) or None,#0.91,
                fig_title_fontsize=plot_params.get('fig_title_fontsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('fig_title_fontsize',30)) or None,#30,
                subfig_title_fontsize=plot_params.get('subfig_title_fontsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('subfig_title_fontsize',24)) or None,#24,
                feature_label_fontsize=plot_params.get('feature_label_fontsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('feature_label_fontsize',24)) or None,#24,
                tick_label_fontsize=plot_params.get('tick_label_fontsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('tick_label_fontsize',16)) or None,#16,
                legend_fontsize=plot_params.get('legend_fontsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('legend_fontsize',24)) or None,#24,
                row_hspace=plot_params.get('row_hspace', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('row_hspace',0.4)) or None,#0.4,
                col_wspace=plot_params.get('col_wspace', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('col_wspace',-0.2)) or None,#-0.2,
                bar2dotplot_width_ratios=plot_params.get('bar2dotplot_width_ratios', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('bar2dotplot_width_ratios',[1.5, 1.5])) or None,# [1.5, 1.5],
                tight_layout_rect_arg=plot_params.get('tight_layout_rect_arg', [0, 0.03, 1, 1]) or None,
                use_tight_layout=plot_params.get('use_tight_layout', False) or None,
                savefig=G.SAVE_OUTPUT_FIGURES,
                file_name=plot_params.get('file_name', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('file_name')) or None,
                comparison_col=plot_params.get('comparison_col', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_col')) or None,#'Timepoint',
                comparison_order=plot_params.get('comparison_order', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('comparison_order',None)) or None,
                hue_palette_color_list=plot_params.get('hue_palette_color_list', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('hue_palette_color_list', None)) or None,
                barh_remove_yticklabels=plot_params.get('barh_remove_yticklabels', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_remove_yticklabels',True)) or None,#True,
                barh_figure_plot_title=plot_params.get('barh_figure_plot_title', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_figure_plot_title','barh_figure_plot_title')) or None,#'Metabolon Batch-norm imputed Values',
                barh_subplot_xlabel=plot_params.get('barh_subplot_xlabel', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_subplot_xlabel','barh_subplot_xlabel')) or None,#'Metabolon Batch-norm Values',
                barh_sharex=plot_params.get('barh_sharex', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_sharex',False)) or None,#False,
                barh_legend=plot_params.get('barh_legend', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend',True)) or None,#True,
                barh_legend_bbox_to_anchor=plot_params.get('barh_legend_bbox_to_anchor', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('barh_legend_bbox_to_anchor',(0.5, 0.01))) or None,#(0.5, 0.01),
                dotplot_figure_plot_title=plot_params.get('dotplot_figure_plot_title', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_figure_plot_title')) or None,#'log2FoldChange',
                dotplot_pval_vars_col_label=plot_params.get('dotplot_pval_vars_col_label', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_pval_vars_col_label')) or None,#'',
                dotplot_l2fc_vars_col_label=plot_params.get('dotplot_l2fc_vars_col_label', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_l2fc_vars_col_label')) or None,#'',
                dotplot_subplot_xlabel=plot_params.get('dotplot_subplot_xlabel', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_subplot_xlabel')) or None,#'
                pval_label=plot_params.get('pval_label', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pval_label')) or None,#"RAW paired pvalue:",
                l2fc_label='not used',
                pvalue_cutoff_ring=plot_params.get('pvalue_cutoff_ring', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('pvalue_cutoff_ring',0.1)) or None,#0.1,
                sizes=plot_params.get('sizes', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('sizes',(20, 2000))) or None,#(20, 2000),
                dotplot_sharex=plot_params.get('dotplot_sharex', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_sharex',True)) or None,#True,
                dotplot_set_xaxis_lims=plot_params.get('dotplot_set_xaxis_lims', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_set_xaxis_lims',None)) or None,
                dotplot_legend=plot_params.get('dotplot_legend', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend',True)) or None,#True,
                dotplot_legend_bins=plot_params.get('dotplot_legend_bins', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bins',3)) or None,#3,
                dotplot_legend_bbox_to_anchor=plot_params.get('dotplot_legend_bbox_to_anchor', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_legend_bbox_to_anchor',(0.5, 0.01))) or None,#(0.5, 0.01),
                dotplot_annotate=plot_params.get('dotplot_annotate', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate',True)) or None,#True,
                dotplot_annotate_xy=plot_params.get('dotplot_annotate_xy', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_xy',(0.8, 1.2))) or None,#(0.8, 1.2),
                dotplot_annotate_fontsize=plot_params.get('dotplot_annotate_fontsize', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_fontsize')) or None,#None,
                dotplot_annotate_labels=plot_params.get('dotplot_annotate_labels', BARH_L2FC_DOTPLOT_COLUMN_CALLS_DEFAULTS.get('dotplot_annotate_labels',('l2fc: ', 'RAW pvalue: '))) or None,#('l2fc: ', 'RAW pvalue: ')
            )
        plt.close("all")
    ##################################################### barh_l2fc_dotplot_column_calls ##########################

    LOGGER.info("Finished make_diff_datapoint_plots.py script.")
