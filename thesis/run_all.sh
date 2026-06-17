#!/usr/bin/env bash
# One-command reproduction of all thesis artifacts. Deterministic given seeds.
set -e
cd "$(dirname "$0")/src"
echo "[1/4] validation (V1-V3 + Castillo regime sweep)"
python3 validate.py > ../results/validation_run.txt 2>&1
echo "[2/4] main experiment grid (66 scenarios x 27 combos x 8 seeds)"
python3 run_experiments.py --lhs 60 --seeds 8 --procs 4
echo "[3/4] analysis tables + selection significance (writes results/analysis_digest.txt)"
python3 analysis.py
echo "[4/4] figures"
python3 figures.py
echo "done -> results/ and figures/"
