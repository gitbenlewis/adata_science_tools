#!/usr/bin/env bash
set -euo pipefail
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_PMID_33969320/scripts/000_run_everything_else.bash

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
conda activate "${POSTPROCESS_ENV}"

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
# 0) download the data
"${PYTHON_BIN}" ./scripts/py0_download_and_parse_input_files_PMID_33969320.py
# 3) run the make_diff_test_tables script
"${PYTHON_BIN}" ./scripts/make_diff_test_tables.py
# 5) run the make_annotated_adata.py script
"${PYTHON_BIN}" ./scripts/make_annotated_adata.py
# 7) run the make_diff_datapoint_plots.py script
"${PYTHON_BIN}" ./scripts/make_diff_datapoint_plots.py

##########################################################
# 1) run the make_parse_datasets script
#"${PYTHON_BIN}" ./scripts/make_parse_datasets.py 

# 2) run generate PCA tables and plots
#"${PYTHON_BIN}" ./scripts/make_pca_tables.py
#"${PYTHON_BIN}" ./scripts/make_pca_plots.py

# 3) run the make_diff_test_tables script
#"${PYTHON_BIN}" ./scripts/make_diff_test_tables.py

# 4) run the make_model_fit_tables script
#"${PYTHON_BIN}" ./scripts/make_model_fit_tables.py

# 5) run the make_annotated_adata.py script
#"${PYTHON_BIN}" ./scripts/make_annotated_adata.py

# 6) run the make_volcano_plots.py script
#"${PYTHON_BIN}" ./scripts/make_volcano_plots.py

# 7) run the make_diff_datapoint_plots.py script
#"${PYTHON_BIN}" ./scripts/make_diff_datapoint_plots.py

# 8) run the make_gseapy_dotplots.py script
#"${PYTHON_BIN}" ./scripts/make_gseapy_dotplots.py
#########################################################

