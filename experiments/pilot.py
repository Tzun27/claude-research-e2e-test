"""Pilot: validate the full pipeline end-to-end on the price-isolation setup.

For a given objective, train a PPO spatio-temporal surge controller (dispatch fixed,
rebalance off -> isolates price flexibility) and compare to the optimized uniform-pricing
counterfactual. Reports the four-way welfare decomposition (surge vs uniform).
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, PPOConfig, RideHailEnv
from ridehail.config import REWARD_SCALES
from ridehail.ppo import train_ppo
from ridehail.baselines import uniform_search
from ridehail.evaluate import run_policy_seeds, aggregate, mean_summary, objective_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="profit")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--fix_dispatch", type=int, default=2)
    args = ap.parse_args()
    obj = args.objective

    cfg = SimConfig(rebalance_enabled=False)
    mk = lambda: RideHailEnv(cfg, objective=obj, fix_dispatch_radius=args.fix_dispatch,
                             reward_scale=REWARD_SCALES[obj])
    eval_seeds = list(range(20, 28))

    print(f"=== Objective: {obj} ===")
    t0 = time.time()
    print("Uniform-pricing grid search (Castillo counterfactual)...")
    best_m, info = uniform_search(mk, obj, seeds=eval_seeds, dispatch_radius=args.fix_dispatch)
    uni = aggregate(info["best_summaries"])
    print(f"  best uniform multiplier = {best_m:.2f}  ({time.time()-t0:.0f}s)")

    ppo = PPOConfig(total_steps=args.steps, seed=0)
    print(f"Training PPO surge controller ({args.steps} steps)...")
    policy, ac, rms, log = train_ppo(mk, ppo, log_every=20000, verbose=True)
    surge_summaries = run_policy_seeds(mk, policy, eval_seeds)
    sur = aggregate(surge_summaries)
    # report the learned mean surge level (sanity: should move off the init ~1.9)
    env = mk(); obs = env.reset(seed=999); slv = []
    done = False
    while not done:
        raw = policy(obs); surge, dr, reb = env.decode_action(raw)
        slv.append(surge.mean()); obs, r, done, info = env.step(raw)
    print(f"  training+eval done ({time.time()-t0:.0f}s); learned mean surge={np.mean(slv):.2f} "
          f"[{np.min(slv):.2f},{np.max(slv):.2f}]")

    print("\n--- Four-way welfare decomposition (mean over seeds) ---")
    GR = uni["gross_revenue"]["mean"]
    print(f"{'component':18s} {'uniform':>12s} {'surge':>12s} {'Δ':>10s} {'Δ%ofGR':>9s}")
    for k in ["total_welfare", "rider_surplus", "driver_surplus", "platform_profit", "gross_revenue", "n_matched"]:
        u = uni[k]["mean"]; s = sur[k]["mean"]; d = s - u
        print(f"{k:18s} {u:12.1f} {s:12.1f} {d:10.1f} {100*d/GR:8.2f}%")
    print(f"\nObjective value: uniform={objective_value(mean_summary(info['best_summaries']), obj):.1f}  "
          f"surge={objective_value(mean_summary(surge_summaries), obj):.1f}")
    print("Castillo target signs: total +, rider +, driver -, platform -")


if __name__ == "__main__":
    main()
