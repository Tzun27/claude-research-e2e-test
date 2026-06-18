"""Produce the §5.0 validation numbers at the FINAL config, for the thesis.

Reports, with the exact regime each is measured in:
  - max relative consistency gap (no-leakage check)
  - demand price-elasticity (flooded regime, isolating the pure price response = Castillo's
    demand elasticity) AND the served-trip response at the operating fleet (for context)
  - match rate, acceptance rate, mean wait at the operating point (multiplier 1)
  - abandonment-vs-demand correlation and peak/trough abandonment
All numbers printed are copy-paste-ready for §5.0.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv
from ridehail.evaluate import run_constant_seeds, mean_summary


def main():
    cfg = SimConfig(rebalance_enabled=False)
    mk = lambda c=cfg: RideHailEnv(c, objective="welfare", fix_dispatch_radius=2)

    # consistency
    maxrel = 0.0
    for m in [0.9, 1.2, 1.8, 2.5]:
        for s in range(3):
            w = run_constant_seeds(mk, m, 2, [s])[0]
            maxrel = max(maxrel, w["consistency_gap"] / (abs(w["W_components_sum"]) + 1e-9))
    print(f"max relative consistency gap: {maxrel:.1e}")

    # demand elasticity, flooded (pure price response) -- Castillo's demand elasticity
    big = SimConfig(**{**cfg.__dict__, "n_drivers": 2000})
    mkb = lambda: RideHailEnv(big, objective="welfare", fix_dispatch_radius=2)
    rlo = mean_summary(run_constant_seeds(mkb, 0.95, 2, range(8)))["n_requests"]
    rhi = mean_summary(run_constant_seeds(mkb, 1.05, 2, range(8)))["n_requests"]
    eps = -((rhi - rlo) / ((rhi + rlo) / 2)) / ((1.05 - 0.95))
    pot = float(RideHailEnv(big).lam.sum())
    accb = mean_summary(run_constant_seeds(mkb, 1.0, 2, range(8)))["n_requests"] / pot
    print(f"demand price-elasticity (flooded, pure price response): {eps:.3f}")
    print(f"acceptance rate at mult=1 (flooded): {accb:.3f}")

    # operating point (fleet=400, mult=1)
    op = mean_summary(run_constant_seeds(mk, 1.0, 2, range(8)))
    print(f"operating point (fleet=400, mult=1): "
          f"match_rate={op['n_matched']/max(1,op['n_requests']):.3f}, "
          f"acceptance={op['n_requests']/pot:.3f}, "
          f"mean_wait={op['total_wait']/max(1,op['n_matched']):.2f} epochs")

    # demand downward-sloping (operating fleet)
    req = [mean_summary(run_constant_seeds(mk, m, 2, range(6)))["n_requests"] for m in [1.0, 1.5, 2.5]]
    print(f"requests vs mult [1.0,1.5,2.5]: {[round(r) for r in req]}")

    # abandonment vs demand over the day
    env = RideHailEnv(cfg, objective="welfare", fix_dispatch_radius=2)
    obs = env.reset(seed=0); done = False; ab = []; t = 0
    frac = (1.0 - cfg.surge_min) / (cfg.surge_max - cfg.surge_min)
    raw = np.zeros(env.act_dim); raw[:env.Z] = np.log(frac/(1-frac)); raw[env.Z+1:] = -10
    while not done:
        obs, r, done, info = env.step(raw); ab.append(info["n_abandon"]); t += 1
    ab = np.array(ab); dem = env.lam.sum(1)
    order = np.argsort(dem)
    print(f"abandonment-demand corr: {np.corrcoef(dem, ab)[0,1]:.3f}; "
          f"peak vs trough abandonment: {ab[order[-10:]].mean():.0f} vs {ab[order[:10]].mean():.1f}")


if __name__ == "__main__":
    main()
