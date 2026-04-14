#!/usr/bin/env python3
"""Run correlation dotplots for the simulated Age covariate example."""
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py
####################################
import sys
from collections import ChainMap
from pathlib import Path
import anndata
from dataclasses import dataclass
from datetime import datetime
import logging
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

 # CFG Configuration
####################################
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIG_YAML_PATH = REPO_ROOT / "config" / "config.yaml"
with REPO_CONFIG_YAML_PATH.open() as f:
    CFG = yaml.safe_load(f)

# out and log path
OUTPUT_DIR = Path(CFG["plot_dotplot_simulate_1_var_covar_age_params"]["repo_results_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
####################################


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
@dataclass
class G():
    """Class to hold global variables."""
    WRITE_DIR = '/home/ubuntu/write/'
    GITBENLEWIS_REPO_PARENT_DIR = '/home/ubuntu/projects/gitbenlewis/'
    SCRIPTS_DIR = '../scripts/'
    CONFIG_DIR = '../config/'
    RESULTS_DIR = '../results/'
    WRITE_CACHE = False
    SAVE_OUTPUT = True
    SAVE_OUTPUT_FIGURES = True
# ------------- dataclass G()  --------------------------------------------------------


########## import custom code libraries ################################################
print(f"REPO_ROOT set to: {str(REPO_ROOT)}")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
try:
    from code_library import adata_science_tools as adtl
    print(f"Using adata_science_tools / adtl from {adtl.__file__}")
except ImportError as e:
    print(f"code_library adata_science_tools not available: {e}")
    LOCAL_REPO_PARENT = REPO_ROOT.parent.parent
    if str(LOCAL_REPO_PARENT) not in sys.path:
        sys.path.append(str(LOCAL_REPO_PARENT))
    import adata_science_tools as adtl
    print(f"Using local adata_science_tools / adtl from {adtl.__file__}")
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


if __name__ == "__main__":
    LOGGER.info("Starting plot_dotplot_simulate_1_var_covar_age.py script.")

    SIM_CFG = CFG["simulate_1_var_covar_age_params"]
    PLOT_CFG = CFG["plot_dotplot_simulate_1_var_covar_age_params"]
    DEFAULT_PARAMS = PLOT_CFG.get("default_params") or {}
    PLOT_RUNS = PLOT_CFG.get("plot_dotplot_simulate_1_var_covar_age__runs") or {}
    SIM_OUTPUT_DIR = Path(SIM_CFG["repo_results_dir"])

    if not PLOT_RUNS:
        LOGGER.warning(
            "No plot runs configured under plot_dotplot_simulate_1_var_covar_age__runs; nothing to do."
        )

    for run_key, run_values in PLOT_RUNS.items():
        LOGGER.info("###################################################################################################")
        LOGGER.info("run_key: %s  with info: \n %s", run_key, run_values)
        chained_params = ChainMap(run_values, DEFAULT_PARAMS, {"run": False})
        LOGGER.info("run control set to: %s", chained_params["run"])

        if not chained_params["run"]:
            LOGGER.info("Skipping plot for %s as per config.", run_key)
            continue

        adata_h5ad_path = Path(
            chained_params.get(
                "adata_h5ad_path",
                SIM_OUTPUT_DIR / run_key / f"{run_key}.h5ad",
            )
        )
        if not adata_h5ad_path.exists():
            raise FileNotFoundError(
                f"Run '{run_key}' expected AnnData input at {adata_h5ad_path}, but it does not exist."
            )

        plot_path = Path(chained_params.get("file_name", OUTPUT_DIR / run_key / run_key))
        if plot_path.suffix == "":
            plot_path = plot_path.with_suffix(".png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        palette = chained_params.get(
            "palette",
            ['#AA4499', '#332288', '#1f77b4', "#661100", "#117733", '#ff7f0e'],
        )
        subset_palette = chained_params.get(
            "subset_palette",
            ['#AA4499', '#332288', '#1f77b4', "#661100", "#117733", '#ff7f0e'],
        )
        if not isinstance(palette, (list, tuple)):
            raise ValueError(f"Run '{run_key}' must provide a list-like palette, got {palette}.")
        if not isinstance(subset_palette, (list, tuple)):
            raise ValueError(
                f"Run '{run_key}' must provide a list-like subset_palette, got {subset_palette}."
            )

        column_key_x = chained_params.get("column_key_x", "Age")
        column_key_y = chained_params.get("column_key_y", "simulated_feature")
        hue = chained_params.get("hue", "case_control")
        subset_key = chained_params.get("subset_key", "case_control")
        axes_title = chained_params.get("axes_title", f"{column_key_y} vs {column_key_x}")

        LOGGER.info("Resolved plot params for %s: %s", run_key, dict(chained_params))
        adata = anndata.read_h5ad(adata_h5ad_path)
        LOGGER.info("Loaded adata from %s", adata_h5ad_path)
        LOGGER.info("adata: %s", adata)
        LOGGER.info("adata.obs columns: %s", list(adata.obs.columns))
        LOGGER.info("adata.var columns: %s", list(adata.var.columns))
        LOGGER.info("adata.var_names: %s", list(adata.var_names))

        if column_key_x not in adata.obs.columns:
            raise ValueError(
                f"Run '{run_key}' requires column_key_x '{column_key_x}' in adata.obs, found {list(adata.obs.columns)}."
            )
        if hue not in adata.obs.columns:
            raise ValueError(
                f"Run '{run_key}' requires hue '{hue}' in adata.obs, found {list(adata.obs.columns)}."
            )
        if subset_key not in adata.obs.columns:
            raise ValueError(
                f"Run '{run_key}' requires subset_key '{subset_key}' in adata.obs, found {list(adata.obs.columns)}."
            )
        if column_key_y not in adata.var_names and column_key_y not in adata.obs.columns:
            raise ValueError(
                f"Run '{run_key}' requires column_key_y '{column_key_y}' in adata.var_names or adata.obs columns."
            )

        fig = None
        try:
            fig, axes, fit, corr_value, corr_pvalue = adtl.corr_dotplot_dev(
                adata=adata,
                layer=chained_params.get("layer"),
                column_key_x=column_key_x,
                column_key_y=column_key_y,
                hue=hue,
                show=bool(chained_params.get("show", False)),
                axes_lines=bool(chained_params.get("axes_lines", False)),
                figsize=tuple(chained_params.get("figsize", (12, 8))),
                dropna=bool(chained_params.get("dropna", True)),
                show_y_intercept=bool(chained_params.get("show_y_intercept", False)),
                palette=palette,
                subset_palette=subset_palette,
                dot_size=float(chained_params.get("dot_size", 200)),
                title_fontsize=int(chained_params.get("title_fontsize", 16)),
                axis_label_fontsize=int(chained_params.get("axis_label_fontsize", 12)),
                legend_fontsize=int(chained_params.get("legend_fontsize", 12)),
                axes_title=axes_title,
                subset_key=subset_key,
                show_all_obs_fit=bool(chained_params.get("show_all_obs_fit", True)),
                show_stats_text=bool(chained_params.get("show_stats_text", False)),
                show_x_marginal_hist=bool(chained_params.get("show_x_marginal_hist", True)),
                show_y_marginal_hist=bool(chained_params.get("show_y_marginal_hist", True)),
                x_marginal_hist_bins=chained_params.get("x_marginal_hist_bins", 10),
                y_marginal_hist_bins=chained_params.get("y_marginal_hist_bins", 10),
                x_marginal_hist_height_ratio=float(
                    chained_params.get("x_marginal_hist_height_ratio", 0.3)
                ),
                y_marginal_hist_width_ratio=float(
                    chained_params.get("y_marginal_hist_width_ratio", 0.3)
                ),
            )
            LOGGER.info("corr_value for %s: %s", run_key, corr_value)
            LOGGER.info("corr_pvalue for %s: %s", run_key, corr_pvalue)
            LOGGER.info("axes returned for %s: %s", run_key, list(axes.keys()))
            LOGGER.info("fit returned for %s: %s", run_key, fit)

            if bool(chained_params.get("save_plot", True)):
                fig.savefig(
                    plot_path,
                    dpi=int(chained_params.get("dpi", 300)),
                    bbox_inches=chained_params.get("bbox_inches", "tight"),
                )
                LOGGER.info("Saved plot for %s to %s", run_key, plot_path)
        finally:
            if fig is not None:
                plt.close(fig)
