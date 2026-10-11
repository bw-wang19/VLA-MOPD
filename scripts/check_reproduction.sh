#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m vla_mopd.code.train_mopd \
  --check-config config/mopd_pg.yaml
"${PYTHON_BIN}" -m vla_mopd.code.train_mopd \
  --check-config config/mopd_topk.yaml
"${PYTHON_BIN}" tests/run_tests.py

