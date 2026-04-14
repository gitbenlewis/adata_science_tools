#!/usr/bin/env bash
set -euo pipefail
# adata_science_tools/example_simulated_data/scripts/000_run_everything.bash

#########################################################
# 0) Activate conda environment that contains the Python deps for POSTPROCESS_ENV
######## set up for POSTPROCESS_ENV ########
#LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-$(git rev-parse --show-toplevel)/config/local_env.sh}"
#CONFIG_FILE="${CONFIG_FILE:-$(git rev-parse --show-toplevel)/config/local_env.sh}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-$(cd "$SCRIPT_DIR/.." && pwd)/config/local_env.sh}"

if [[ -f "$LOCAL_ENV_FILE" ]]; then
  source "$LOCAL_ENV_FILE"
else
  echo "[ERROR] LOCAL_ENV_FILE not found: $LOCAL_ENV_FILE" >&2
  exit 1
fi

NFCORE_ENV="${NFCORE_ENV:-env_nextflow}"
POSTPROCESS_ENV="${POSTPROCESS_ENV:-not_base}"

source ~/miniconda3/etc/profile.d/conda.sh
#echo "[INFO] Activating conda env: ${NFCORE_ENV}"
#conda activate "${NFCORE_ENV}"
echo "[INFO] Activating conda env: ${POSTPROCESS_ENV}"
#conda activate "${POSTPROCESS_ENV}"
set +u; conda activate "${POSTPROCESS_ENV}"; set -u
# Guard: ensure yq exists 
if ! command -v yq >/dev/null 2>&1; then
  echo "[ERROR] yq is required but not found on PATH." >&2
  exit 1
fi

# Ensure python interpreter (with pandas) is available (prefer python3, fall back to python)
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] No python or python3 executable found on PATH; cannot run post-processing step." >&2
  exit 1
fi
### END ##### set up for POSTPROCESS_ENV ########
#########################################################

# 1) get script file info for Logfile (capture stdout/stderr for provenance)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$SCRIPT_DIR/logs"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
COMBINED_LOG="${SCRIPT_DIR}/logs/${SCRIPT_NAME}.log"
exec > >(tee -a "$COMBINED_LOG") 2>&1

# 2)  CD to Repo root
#REPO_ROOT=$(git rev-parse --show-toplevel)
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"
echo "[INFO] Changed directory to repo root: $REPO_ROOT"

#########################################################

# 1) simulate the data with the simulate_1_var_covar_age.py script
"${PYTHON_BIN}" ./scripts/simulate_1_var_covar_age.py
# 2) run the plot_dotplot_simulate_1_var_covar_age.py script
"${PYTHON_BIN}" ./scripts/plot_dotplot_simulate_1_var_covar_age.py


#########################################################
