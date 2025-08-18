#!/usr/bin/env bash
set -euo pipefail
RES=${1:-results}
python -m src.aggregate --results_root ${RES} --out_dir ${RES}
python -m src.plots     --results_dir ${RES} --out_dir ${RES}/figs
echo "Figures are in ${RES}/figs"
