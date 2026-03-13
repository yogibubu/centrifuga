#!/usr/bin/env bash
set -euo pipefail
cd /Users/vincenzobarone/centrifugal
PY_BIN="/opt/miniconda3/bin/python3"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3)"
fi
exec "$PY_BIN" /Users/vincenzobarone/centrifugal/ceditt_gui.py
