"""Sanity checks for the ride-hailing simulator.

Verifies: (1) welfare consistency identity holds to fp tolerance; (2) demand falls with
surge (downward-sloping demand); (3) pickup time rises under driver scarcity (wild-goose
chase); (4) online supply tracks demand over the day.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ridehail import SimConfig, RideHailEnv


def fixed_surge_policy(env, mult):
    """Return a raw action that decodes to a uniform surge `mult`, mid dispatch radius, no rebalance."""
    cfg = env.cfg
    # invert the sigmoid mapping for surge
    frac = (mult - cfg.surge_min) / (cfg.surge_max - cfg.surge_min)
    frac = min(max(frac, 1e-4), 1 - 1e-4)
    a_surge = np.log(frac / (1 - frac))
    raw = np.zeros(env.act_dim)
    raw[:env.Z] = a_surge
    raw[env.Z] = 0.0           # mid dispatch radius
    raw[env.Z + 1:] = -10.0    # ~0 rebalance
    return raw


def run(mult, seed=0, cfg=None):
    cfg = cfg or SimConfig(seed=seed)
    env = RideHailEnv(cfg, objective="welfare")
    raw = fixed_surge_policy(env, mult)
    w = env.run_episode(lambda obs: raw, seed=seed)
    return w


def main():
    print("=" * 70)
    print("SANITY CHECK 1: welfare consistency identity (W_components == W_resource)")
    maxgap = 0.0
    for mult in [0.9, 1.0, 1.3, 1.8]:
        for seed in range(3):
            w = run(mult, seed=seed)
            gap = w['consistency_gap']
            scale = abs(w['W_components_sum']) + 1e-9
            maxgap = max(maxgap, gap / scale)
            print(f"  mult={mult:<4} seed={seed}  W={w['total_welfare']:10.1f}  "
                  f"gap={gap:.3e}  rel={gap/scale:.2e}")
    print(f"  --> max relative consistency gap = {maxgap:.2e} "
          f"({'PASS' if maxgap < 1e-6 else 'FAIL'})")

    print("=" * 70)
    print("SANITY CHECK 2: demand response to surge (matched trips vs multiplier)")
    cfg = SimConfig(seed=0)
    for mult in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
        ws = [run(mult, seed=s) for s in range(4)]
        nm = np.mean([w['n_matched'] for w in ws])
        nr = np.mean([w['n_requests'] for w in ws])
        gr = np.mean([w['gross_revenue'] for w in ws])
        print(f"  mult={mult:<4}  requests={nr:7.0f}  matched={nm:7.0f}  "
              f"match_rate={nm/max(1,nr):.2f}  gross_rev={gr:8.0f}")

    print("=" * 70)
    print("SANITY CHECK 3: wild-goose-chase (mean pickup time vs fleet size)")
    for ndrv in [120, 200, 300, 450, 600]:
        cfg = SimConfig(seed=0, n_drivers=ndrv)
        env = RideHailEnv(cfg, objective="welfare")
        raw = fixed_surge_policy(env, 1.0)
        # collect mean pickup over episode
        obs = env.reset(seed=0); done = False; pickups = []; matched = 0
        while not done:
            obs, r, done, info = env.step(raw)
            pickups.append(info['mean_pickup']); matched += info['n_matched']
        print(f"  n_drivers={ndrv:<4}  mean_pickup={np.mean(pickups):.3f} cells  "
              f"total_matched={matched}")

    print("=" * 70)
    print("SANITY CHECK 4: online supply over the day (should track bimodal demand)")
    cfg = SimConfig(seed=0)
    env = RideHailEnv(cfg, objective="welfare")
    raw = fixed_surge_policy(env, 1.0)
    obs = env.reset(seed=0); done = False; online = []; dem = []
    t = 0
    while not done:
        dem.append(env.lam[t].sum())
        obs, r, done, info = env.step(raw)
        online.append(info['online']); t += 1
    online = np.array(online); dem = np.array(dem)
    print(f"  demand range  [{dem.min():.0f}, {dem.max():.0f}]")
    print(f"  online range  [{online.min()}, {online.max()}]  (pool={cfg.n_drivers})")
    print(f"  corr(demand, online) = {np.corrcoef(dem, online)[0,1]:.3f}")


if __name__ == "__main__":
    main()
