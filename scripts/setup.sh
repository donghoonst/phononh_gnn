#!/usr/bin/env bash
set -euo pipefail
cd /workspace/phononh_gnn
export PIP_CACHE_DIR=/workspace/.cache/pip
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
if [ ! -x /workspace/ifc2-env/bin/python ]; then
  python3 -m venv /workspace/ifc2-env
fi
/workspace/ifc2-env/bin/python -m pip install --disable-pip-version-check 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
/workspace/ifc2-env/bin/python -m pip install --disable-pip-version-check -r requirements-lock.txt
/workspace/ifc2-env/bin/python -m pip install --disable-pip-version-check --no-deps --no-build-isolation -e .
/workspace/ifc2-env/bin/python -m pip check
