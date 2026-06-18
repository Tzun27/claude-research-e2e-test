#!/bin/bash
# Full experiment matrix. ARS parallelizes internally (4 workers), so cells run sequentially.
# Override ITERS / EVAL via environment. Results -> results/data/*.json.
# No `set -e`: cells are independent, so one failure should not abort the batch.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
DATA=results/data
mkdir -p "$DATA"
ITERS=${ITERS:-50}
EVAL=${EVAL:-"100-115"}
CORE_SEEDS=${CORE_SEEDS:-"0,1,2"}     # 3 training seeds on the core cells (RQ1-3)
AUX_SEEDS=${AUX_SEEDS:-"0,1"}         # 2 training seeds on auxiliary sweeps
RC="python3 experiments/run_cell.py --eval_seeds $EVAL --iters $ITERS"

echo "===== MAIN: price-isolation, 4 objectives (RQ1, RQ2, RQ3 via by-type) ====="
for obj in profit throughput welfare welfare_weighted; do
  $RC --objective $obj --condition price --seeds $CORE_SEEDS --out $DATA/price_$obj.json
done

echo "===== THREE-WAY (Gap A0): welfare_weighted + profit (uniform baseline also optimizes radius) ====="
for obj in welfare_weighted profit; do
  $RC --objective $obj --condition threeway --seeds $AUX_SEEDS --out $DATA/threeway_$obj.json
done

echo "===== FAIRNESS FRONTIER (RQ4): welfare_weighted price, sweep weight ====="
for w in 30 100 300 800; do
  $RC --objective welfare_weighted --condition fair --fair_weight $w --seeds $AUX_SEEDS \
      --out $DATA/fair_ww_w${w}.json
done

echo "===== SENSITIVITY (scope conditions): welfare_weighted price ====="
$RC --objective welfare_weighted --condition price --seeds $AUX_SEEDS --cfg '{"n_drivers":300}'     --out $DATA/sens_ndrivers_300.json
$RC --objective welfare_weighted --condition price --seeds $AUX_SEEDS --cfg '{"wtp_log_sigma":1.8}' --out $DATA/sens_moreelastic.json
echo "===== DONE ====="
