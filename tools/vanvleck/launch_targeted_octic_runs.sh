#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
outdir="${1:-targeted_runs}"
mkdir -p "$outdir"

run_one() {
  local sector="$1"
  nohup "$python_bin" tools/vanvleck/run_targeted_octic_sectors.py --sector "$sector" \
    > "$outdir/${sector}.out" 2>&1 &
  echo "$! $sector"
}

run_one H32_only
run_one H50_only
run_one H32_plus_H50
run_one H60_only
run_one highorder_full
