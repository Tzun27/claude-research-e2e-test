#!/bin/bash
# Full experiment matrix. ARS parallelizes internally (4 workers), so cells run sequentially.
# Override ITERS / SEEDS / EVAL via environment. Results -> results/data/*.json.
# No `set -e`: cells are independent, so one failure should not abort the batch.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
DATA=results/data
mkdir -p "$DATA"
ITERS=${ITERS:-90}
SEEDS=${SEEDS:-"0,1"}
EVAL=${EVAL:-"100-115"}
RC="python3 experiments/run_cell.py --seeds $SEEDS --eval_seeds $EVAL --iters $ITERS"

echo "===== MAIN: price-isolation, 4 objectives (RQ1, RQ2) ====="
for obj in profit throughput welfare welfare_weighted; do
  $RC --objective $obj --condition price --out $DATA/price_$obj.json
done

echo "===== THREE-WAY (Gap A0): welfare_weighted + profit ====="
for obj in welfare_weighted profit; do
  $RC --objective $obj --condition threeway --out $DATA/threeway_$obj.json
done

echo "===== FAIRNESS FRONTIER (RQ4): welfare_weighted price, sweep weight ====="
for w in 30 100 300 800; do
  $RC --objective welfare_weighted --condition fair --fair_weight $w \
      --out $DATA/fair_ww_w${w}.json
done

echo "===== SENSITIVITY (scope conditions): welfare_weighted price ====="
$RC --objective welfare_weighted --condition price --cfg '{"n_drivers":300}'  --out $DATA/sens_ndrivers_300.json
$RC --objective welfare_weighted --condition price --cfg '{"n_drivers":550}'  --out $DATA/sens_ndrivers_550.json
$RC --objective welfare_weighted --condition price --cfg '{"wtp_log_sigma":1.0}' --out $DATA/sens_elasthi.json
$RC --objective welfare_weighted --condition price --cfg '{"wtp_log_sigma":1.8}' --out $DATA/sens_elastlo.json
echo "===== DONE ====="
