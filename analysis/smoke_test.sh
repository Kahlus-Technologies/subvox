#!/bin/sh
# Runs the analysis scripts on the synthetic fixtures to catch crashes, then the unit tests: sh analysis/smoke_test.sh
# Needs numpy and numba (pip install -r analysis/requirements.txt). reseat_test.py is left out: it is
# written for the real dataset (a first wear of 30+ takes followed by a re-seated wear). reseat.py asks
# the same question of any dataset with two or more wears.
set -e
cd "$(dirname "$0")"
F=fixtures/dataset_small.json
PY=${PYTHON:-python3}
$PY score.py --dataset "$F"
$PY clean_test.py --dataset "$F" --since 800000000
$PY dist_check.py --dataset "$F"
$PY reseat.py --dataset fixtures/dataset_reseat.json
$PY -m unittest discover -s . -p 'test_*.py'
echo "analysis smoke test passed"
