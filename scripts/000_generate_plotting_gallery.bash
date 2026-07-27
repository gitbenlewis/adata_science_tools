#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-${REPO_ROOT}/config/local_env.sh}"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
else
  echo "[ERROR] Missing LOCAL_ENV_FILE: ${LOCAL_ENV_FILE}" >&2
  exit 1
fi

WORKFLOW_ENV="${POSTPROCESS_ENV:-not_base}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] Missing conda bootstrap: ${CONDA_SH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
set +u
conda activate "${WORKFLOW_ENV}"
set -u

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] No Python interpreter found on PATH." >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs"
LOG_PATH="${SCRIPT_DIR}/logs/$(basename "${BASH_SOURCE[0]}").log"
exec > >(tee -a "${LOG_PATH}") 2>&1

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/..${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adata-science-tools-mpl}"
mkdir -p "${MPLCONFIGDIR}"

echo "[INFO] Activating conda environment: ${WORKFLOW_ENV}"
echo "[INFO] Running plotting gallery from: ${REPO_ROOT}"
"${PYTHON_BIN}" -m adata_science_tools.example_plotting_gallery.generate_gallery "$@"
