#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

python -m pytest -q
python -m vla_opd.cli --check-config configs/toy.yaml
