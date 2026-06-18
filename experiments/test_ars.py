"""Validate ARS learns: profit-objective controller should match/beat the uniform optimum."""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv
from ridehail.ars import ARSConfig, train_ars
from ridehail.baselines import uniform_search
from ridehail.evaluate import run_policy_seeds, mean_summary, objective_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="profit")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=0)
    args = ap.parse_args()
    obj = args.objective
    cfg = SimConfig(rebalance_enabled=False)
    env_kwargs = dict(fix_dispatch_radius=2)
    mk = lambda: RideHailEnv(cfg, objective=obj, **env_kwargs)
    eval_seeds = list(range(100, 112))

    t0 = time.time()
    bm, info = uniform_search(mk, obj, seeds=eval_seeds, dispatch_radius=2)
    uni_obj = objective_value(mean_summary(info["best_summaries"]), obj)
    print(f"uniform m*={bm:.2f}  uniform_obj={uni_obj:.0f}  ({time.time()-t0:.0f}s)")

    ars = ARSConfig(n_iters=args.iters, n_dirs=16, top_dirs=8, step_size=0.05,
                    noise_std=0.06, eval_seeds=3, seed=0)
    eval_fn = lambda pol: objective_value(mean_summary(run_policy_seeds(mk, pol, eval_seeds[:6])), obj)
    policy, state = train_ars(cfg, obj, ars, env_kwargs=env_kwargs, hidden=args.hidden,
                              n_workers=4, eval_fn=eval_fn)
    sums = run_policy_seeds(mk, policy, eval_seeds)
    surge_obj = objective_value(mean_summary(sums), obj)
    # learned mean surge level
    env = mk(); obs = env.reset(seed=999); sl = []; done = False
    while not done:
        raw = policy(obs); su, dr, rb = env.decode_action(raw); sl.append(su.mean())
        obs, r, done, info = env.step(raw)
    print(f"\nARS surge_obj={surge_obj:.0f}  vs uniform={uni_obj:.0f}  "
          f"({'BEATS' if surge_obj>=uni_obj else 'below'})  learned mean surge={np.mean(sl):.2f}")
    print(f"total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
