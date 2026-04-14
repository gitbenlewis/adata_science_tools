#!/usr/bin/env python3
"""Run covariate expectation correction workflows configured for this repo."""
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_simulated_data/scripts/simulate_1_var_covar_age.py
####################################
import sys
from collections import ChainMap
from pathlib import Path
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
OUTPUT_DIR = Path(CFG["simulate_1_var_covar_age_params"]["repo_results_dir"])
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
    from code_library.adata_science_tools._simulate_data import sim_covar_dependent_dataset
    print(f"Using adata_science_tools / adtl from {adtl.__file__}")
except ImportError as e:
    print(f"code_library adata_science_tools not available: {e}")
    LOCAL_REPO_PARENT = REPO_ROOT.parent.parent
    if str(LOCAL_REPO_PARENT) not in sys.path:
        sys.path.append(str(LOCAL_REPO_PARENT))
    import adata_science_tools as adtl
    from adata_science_tools._simulate_data import sim_covar_dependent_dataset
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
    LOGGER.info("Starting simulate_1_var_covar_age.py script.")

    SIM_CFG = CFG["simulate_1_var_covar_age_params"]
    DEFAULT_PARAMS = SIM_CFG.get("default_params") or {}
    SIM_RUNS = SIM_CFG.get("simulate_1_var_covar_age__runs") or {}

    if not SIM_RUNS:
        LOGGER.warning("No simulation runs configured under simulate_1_var_covar_age__runs; nothing to do.")

    for run_key, run_values in SIM_RUNS.items():
        LOGGER.info("###################################################################################################")
        LOGGER.info("run_key: %s  with info: \n %s", run_key, run_values)
        chained_params = ChainMap(run_values, DEFAULT_PARAMS, {"run": False})
        LOGGER.info("run control set to: %s", chained_params["run"])

        if not chained_params["run"]:
            LOGGER.info("Skipping simulation for %s as per config.", run_key)
            continue

        var_name = chained_params.get("var_name", "simulated_feature")
        beta_age = chained_params.get("beta_age", chained_params.get("beta", 0.05))
        beta_case_control = chained_params.get("beta_case_control", 5.0)
        residual_mean = chained_params.get("residual_mean", 0.0)
        residual_stdev = chained_params.get("residual_stdev", 1.0)
        if isinstance(var_name, (list, tuple)):
            raise ValueError(
                f"Run '{run_key}' must provide a scalar var_name for this script, got {var_name}."
            )
        if isinstance(beta_age, (list, tuple, dict)):
            raise ValueError(
                f"Run '{run_key}' must provide a scalar beta_age/beta for this script, got {beta_age}."
            )
        if isinstance(beta_case_control, (list, tuple, dict)):
            raise ValueError(
                f"Run '{run_key}' must provide a scalar beta_case_control for this script, got {beta_case_control}."
            )
        if isinstance(residual_mean, (list, tuple, dict)):
            raise ValueError(
                f"Run '{run_key}' must provide a scalar residual_mean for this script, got {residual_mean}."
            )
        if isinstance(residual_stdev, (list, tuple, dict)):
            raise ValueError(
                f"Run '{run_key}' must provide a scalar residual_stdev for this script, got {residual_stdev}."
            )
        if chained_params.get("obs_key_list") not in (
            None,
            ["Age", "case_control"],
            ("Age", "case_control"),
        ):
            raise ValueError(
                f"Run '{run_key}' is fixed to obs_key_list ['Age', 'case_control'] and cannot override it with {chained_params.get('obs_key_list')}."
            )
        if chained_params.get("obs_covar_dist_params") is not None:
            raise ValueError(
                f"Run '{run_key}' is fixed to Age plus case_control simulation and cannot override obs_covar_dist_params directly."
            )

        output_path = Path(chained_params.get("output_path", OUTPUT_DIR / run_key / run_key))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_obs_df = bool(chained_params.get("save_obs_df", False))
        save_adata_dataset = bool(chained_params.get("save_adata_dataset", True))
        also_return_adata = bool(chained_params.get("also_return_adata", True))
        save_obs_df_path = chained_params.get("save_obs_df_path")
        if save_obs_df and save_obs_df_path is None:
            save_obs_df_path = output_path.with_name(f"{output_path.name}.obs_only")

        resolved_run_params = {
            "n_obs": chained_params.get("n_obs", 100),
            "obs_names_prefix": chained_params.get("obs_names_prefix", "obs_"),
            "random_seed": chained_params.get("random_seed", 7),
            "age_mean": chained_params.get("age_mean", 50.0),
            "age_stdev": chained_params.get("age_stdev", 10.0),
            "case_control_prob": chained_params.get("case_control_prob", 0.5),
            "beta_age": beta_age,
            "beta_case_control": beta_case_control,
            "yint": chained_params.get("yint", 10.0),
            "residual_mean": residual_mean,
            "residual_stdev": residual_stdev,
            "var_name": var_name,
            "save_obs_df": save_obs_df,
            "save_obs_df_path": save_obs_df_path,
            "save_adata_dataset": save_adata_dataset,
            "also_return_adata": also_return_adata,
            "output_path": output_path,
        }
        LOGGER.info("Resolved run params for %s: %s", run_key, resolved_run_params)

        X, var_df, obs_df, adata = sim_covar_dependent_dataset(
            obs_key_list=["Age", "case_control"],
            obs_covar_dist_params={
                "Age": {
                    "dist": "normal",
                    "mean": float(resolved_run_params["age_mean"]),
                    "stdev": float(resolved_run_params["age_stdev"]),
                },
                "case_control": {
                    "dist": "binomial",
                    "prob": float(resolved_run_params["case_control_prob"]),
                },
            },
            n_obs=int(resolved_run_params["n_obs"]),
            obs_names_prefix=str(resolved_run_params["obs_names_prefix"]),
            save_obs_df=False,
            save_obs_df_path=resolved_run_params["save_obs_df_path"],
            random_seed=resolved_run_params["random_seed"],
            var_names=[str(resolved_run_params["var_name"])],
            betas=[
                float(resolved_run_params["beta_age"]),
                float(resolved_run_params["beta_case_control"]),
            ],
            yints=float(resolved_run_params["yint"]),
            residual_mean=float(resolved_run_params["residual_mean"]),
            residual_stdev=float(resolved_run_params["residual_stdev"]),
            # Keep a local AnnData object available so the script can relabel
            # case/control obs values before writing the final dataset bundle.
            also_return_adata=True,
            save_adata_dataset=False,
            output_path=resolved_run_params["output_path"],
        )
        case_control_label_map = {0: "control", 1: "case"}
        obs_df["case_control"] = obs_df["case_control"].map(case_control_label_map)
        if adata is not None:
            adata.obs["case_control"] = adata.obs["case_control"].map(case_control_label_map)
        LOGGER.info("case_control value counts:\n%s", obs_df["case_control"].value_counts(dropna=False))
        if resolved_run_params["save_adata_dataset"]:
            adtl.save_dataset(adata, resolved_run_params["output_path"], logger=LOGGER)
        if resolved_run_params["save_obs_df"]:
            Path(resolved_run_params["save_obs_df_path"]).parent.mkdir(parents=True, exist_ok=True)
            obs_df.to_csv(resolved_run_params["save_obs_df_path"])
        LOGGER.info("Finished run %s", run_key)
        LOGGER.info("X shape: %s", X.shape)
        LOGGER.info("obs_df shape: %s", obs_df.shape)
        LOGGER.info("var_df shape: %s", var_df.shape)
        if adata is not None:
            LOGGER.info("adata: %s", adata)
        LOGGER.info("obs_df head:\n%s", obs_df.head())
        LOGGER.info("var_df head:\n%s", var_df.head())
