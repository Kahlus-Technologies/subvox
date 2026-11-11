#!/bin/sh
# Runs the analysis scripts on the synthetic fixture to catch crashes: sh analysis/smoke_test.sh
# Needs numpy and numba (pip install -r analysis/requirements.txt). reseat_test.py is left out: it is
# written for the real dataset (a first wear of 30+ takes followed by a re-seated wear).
set -e
cd "$(dirname "$0")"
F=fixtures/dataset_small.json
PY=${PYTHON:-python3}
$PY score.py --dataset "$F"
$PY clean_test.py --dataset "$F" --since 800000000
$PY dist_check.py --dataset "$F"
echo "analysis smoke test passed"
