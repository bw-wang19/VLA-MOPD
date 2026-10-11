#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/config/mopd_pg.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

cd "${REPO_ROOT}"

if [[ "${NUM_PROCESSES}" == "1" ]]; then
  "${PYTHON_BIN}" -m vla_mopd.code.train_mopd "${CONFIG_PATH}"
else
  ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
  "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --module vla_mopd.code.train_mopd "${CONFIG_PATH}"
fi

