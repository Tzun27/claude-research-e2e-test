"""Main experiment driver: run every (scenario x controller-combo x seed) and
write the results matrix. Parallelized over scenarios (value table pretrained
once per scenario). Output: thesis/results/results.csv.
"""
from __future__ import annotations
import argparse, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

from scenarios import all_scenarios
from methods import all_combos, combo_needs_value
from engine import run_combo, pretrain_value_table

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METRIC_KEYS = [
    "total_welfare", "rider_surplus", "driver_surplus", "platform_profit", "gmv",
    "service_rate", "mean_wait_min", "mean_surge", "driver_gini",
    "ds_flexible", "ds_constrained", "ds_flexible_per_capita",
    "ds_constrained_per_capita", "earn_rate_flex", "earn_rate_cons",
    "n_requests", "n_potential", "n_completed", "n_abandoned",
    "served_latent_rate", "accept_rate", "n_flexible", "n_constrained",
]


def run_scenario(args):
    scn, n_seeds = args
    t0 = time.time()
    V = pretrain_value_table(scn)
    rows = []
    feats = scn.context_features()
    for combo in all_combos():
        vt = V if combo_needs_value(combo) else None
        for s in range(n_seeds):
            m = run_combo(scn, combo, value_table=vt, seed=s)
            row = {"scenario": scn.name, "seed": s, "combo": m["combo"],
                   "match": m["match"], "price": m["price"], "rebal": m["rebal"]}
            row.update({k: m[k] for k in METRIC_KEYS})
            row.update(feats)
            rows.append(row)
    return scn.name, rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--lhs", type=int, default=42)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "results.csv"))
    args = ap.parse_args()

    scns = all_scenarios(n_lhs=args.lhs)
    print(f"{len(scns)} scenarios x {len(all_combos())} combos x {args.seeds} seeds "
          f"= {len(scns)*len(all_combos())*args.seeds} runs")
    tasks = [(s, args.seeds) for s in scns]
    all_rows = []
    t0 = time.time()
    with Pool(args.procs) as pool:
        for i, (name, rows, dt) in enumerate(pool.imap_unordered(run_scenario, tasks)):
            all_rows.extend(rows)
            print(f"  [{i+1}/{len(scns)}] {name:28s} {dt:5.1f}s "
                  f"({len(all_rows)} rows, {time.time()-t0:5.1f}s elapsed)")
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}  in {time.time()-t0:.1f}s")
    # quick sanity
    print("\nmean total_welfare by price method:")
    print(df.groupby("price")["total_welfare"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
