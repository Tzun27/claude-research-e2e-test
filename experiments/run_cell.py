"""Run one experiment cell: (objective x condition x config-variant).

A "cell" computes the optimized uniform-pricing counterfactual and trains N PPO surge
controllers, then evaluates both on held-out market seeds and saves the full four-way
welfare decomposition (with per-driver-type breakdown) to a JSON result file.

Conditions:
  price    : dispatch fixed, rebalance off -> isolates the welfare incidence of PRICE flexibility
             (Castillo's exact comparison). RQ1/RQ2.
  threeway : controller learns price + dispatch + rebalance (Gap A0 three-way DRL).
  fair     : price condition + a driver-earnings fairness penalty in the reward (RQ4).

Usage:
  python run_cell.py --objective profit --condition price --seeds 0,1 --steps 150000 \
      --eval_seeds 100-115 --out results/data/price_profit.json [--cfg '{"n_drivers":300}']
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv
from ridehail.ars import ARSConfig, train_ars
from ridehail.baselines import uniform_search, MyopicSurge
from ridehail.evaluate import run_policy_seeds, aggregate, mean_summary, objective_value


def parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def make_setup(cfg, objective, condition, fair_weight=0.0):
    """Return (cfg, env_kwargs, mk) for a condition. env_kwargs are passed to RideHailEnv."""
    if condition in ("price", "fair"):
        fix_dispatch = 2
        cfg = SimConfig(**{**cfg.__dict__, "rebalance_enabled": False})
    else:  # threeway
        fix_dispatch = None
        cfg = SimConfig(**{**cfg.__dict__, "rebalance_enabled": True})
    fw = fair_weight if condition == "fair" else 0.0
    env_kwargs = dict(fix_dispatch_radius=fix_dispatch, fair_weight=fw)
    mk = lambda: RideHailEnv(cfg, objective=objective, **env_kwargs)
    return cfg, env_kwargs, fix_dispatch, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True)
    ap.add_argument("--condition", default="price")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--eval_seeds", default="100-114")
    ap.add_argument("--iters", type=int, default=70)
    ap.add_argument("--fair_weight", type=float, default=0.0)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_seeds = parse_seeds(args.seeds)
    eval_seeds = parse_seeds(args.eval_seeds)
    cfg_over = json.loads(args.cfg)
    base_cfg = SimConfig(**cfg_over)
    cfg, env_kwargs, fix_dispatch, mk = make_setup(base_cfg, args.objective, args.condition, args.fair_weight)
    weights = (0.4, 1.0, 0.3)  # welfare_weighted alphas (pi, R, D)

    t0 = time.time()
    result = dict(objective=args.objective, condition=args.condition, cfg_over=cfg_over,
                  train_seeds=train_seeds, eval_seeds=eval_seeds, iters=args.iters,
                  fair_weight=args.fair_weight, fix_dispatch=fix_dispatch)

    # --- uniform-pricing counterfactual (grid search) ---
    # price/fair: matching fixed at radius 2 for both arms (clean price isolation).
    # threeway: the uniform baseline also optimizes the dispatch radius, so the delta
    # isolates *price* flexibility on top of optimized matching (issue: avoid confound).
    if fix_dispatch is not None:
        dispatch_for_uniform = fix_dispatch
        dispatch_grid = None
    else:
        dispatch_for_uniform = 2
        dispatch_grid = [1, 2, 3, 4]
    best_m, uinfo = uniform_search(mk, args.objective, seeds=eval_seeds,
                                   dispatch_radius=dispatch_for_uniform, dispatch_grid=dispatch_grid,
                                   alpha_pi=weights[0], alpha_R=weights[1], alpha_D=weights[2])
    result["uniform_mult"] = best_m
    result["uniform_dispatch"] = uinfo.get("best_dispatch", dispatch_for_uniform)
    result["uniform_summaries"] = uinfo["best_summaries"]
    result["uniform_agg"] = aggregate(uinfo["best_summaries"])
    print(f"[{args.objective}/{args.condition}] uniform m*={best_m:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # --- myopic local surge baseline (Besbes-style) ---
    myo_sums = run_policy_seeds(mk, MyopicSurge(mk(), dispatch_radius=dispatch_for_uniform), eval_seeds)
    result["myopic_agg"] = aggregate(myo_sums)

    # --- train ARS surge controllers (one per train seed) ---
    surge_per_seed = []; thetas = []
    for ts in train_seeds:
        ars = ARSConfig(n_iters=args.iters, n_dirs=16, top_dirs=8, step_size=0.08,
                        noise_std=0.08, eval_seeds=3, seed=ts)
        policy, state = train_ars(cfg, args.objective, ars, env_kwargs=env_kwargs,
                                  hidden=0, weights=weights, n_workers=4, verbose=False)
        sums = run_policy_seeds(mk, policy, eval_seeds)
        surge_per_seed.append(sums)
        thetas.append(state["theta"].tolist())
        print(f"[{args.objective}/{args.condition}] trained seed {ts}  "
              f"obj={objective_value(mean_summary(sums), args.objective):.1f}  ({time.time()-t0:.0f}s)", flush=True)
    result["surge_thetas"] = thetas
    result["Z"] = base_cfg.n_zones()

    # pool eval summaries across train seeds for the aggregate
    pooled = [s for seed_sums in surge_per_seed for s in seed_sums]
    result["surge_summaries"] = pooled
    result["surge_agg"] = aggregate(pooled)

    # Mean-preserving-spread baseline (Castillo's definition of surge): a flat uniform price
    # set to the learned controller's OWN trip-weighted average multiplier. The surge-vs-this
    # comparison isolates the pure effect of price *variation* (allocative efficiency +
    # matching), controlling for the price level -- calibration-robust, unlike surge-vs-optimal.
    import numpy as _np
    mbar = float(_np.mean([s["mult_weighted"] / max(1, s["n_matched"]) for s in pooled]))
    result["surge_mean_mult"] = mbar
    matched_uni = run_constant_seeds(mk, mbar, dispatch_for_uniform, eval_seeds)
    result["matched_uniform_summaries"] = matched_uni
    result["matched_uniform_agg"] = aggregate(matched_uni)
    result["surge_obj_per_train_seed"] = [objective_value(mean_summary(s), args.objective) for s in surge_per_seed]
    result["uniform_obj"] = objective_value(mean_summary(uinfo["best_summaries"]), args.objective)
    result["elapsed_s"] = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.objective}/{args.condition}] saved -> {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
