"""Diagnostic: is the wild-goose-chase (WGC) matching friction a real force?

At a fixed (tight) fleet and uniform price, sweep the dispatch radius. WGC predicts that
serving far-away riders (large radius) lengthens pickups and ties up drivers, so beyond
some point throughput should stop rising (or fall) and mean pickup should climb. Also show
that abandonment concentrates during demand peaks (local supply depletion).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv


def policy(env, mult, dispatch_radius):
    cfg = env.cfg
    frac = (mult - cfg.surge_min) / (cfg.surge_max - cfg.surge_min)
    raw = np.zeros(env.act_dim)
    raw[:env.Z] = np.log(frac / (1 - frac))
    # invert dispatch sigmoid to hit an exact integer radius
    df = (dispatch_radius - cfg.dispatch_radius_min) / (cfg.dispatch_radius_max - cfg.dispatch_radius_min)
    df = min(max(df, 1e-3), 1 - 1e-3)
    raw[env.Z] = np.log(df / (1 - df))
    raw[env.Z + 1:] = -10.0
    return raw


def run(cfg, mult, dispatch_radius, seed):
    env = RideHailEnv(cfg, objective="welfare")
    raw = policy(env, mult, dispatch_radius)
    obs = env.reset(seed=seed); done = False
    req = match = aband = 0; pickups = []; per_epoch_aband = []
    while not done:
        obs, r, done, info = env.step(raw)
        req += info['n_requests']; match += info['n_matched']; aband += info['n_abandon']
        pickups.append(info['mean_pickup']); per_epoch_aband.append(info['n_abandon'])
    w = info['welfare']
    return dict(req=req, match=match, aband=aband, mean_pickup=float(np.nanmean(pickups)),
                W=w['total_welfare'], driver_busy_pickup=w['total_pickup'],
                per_epoch_aband=np.array(per_epoch_aband))


def main():
    cfg = SimConfig(n_drivers=400)
    print("Dispatch-radius sweep at uniform mult=1.0, n_drivers=400 (WGC test):")
    print(f"{'radius':>6} {'matched':>8} {'mean_pickup':>11} {'tot_pickup':>11} {'abandon':>8} {'W':>9}")
    for dr in [1, 2, 3, 4]:
        st = [run(cfg, 1.0, dr, s) for s in range(4)]
        m = np.mean([s['match'] for s in st]); pu = np.mean([s['mean_pickup'] for s in st])
        tp = np.mean([s['driver_busy_pickup'] for s in st]); ab = np.mean([s['aband'] for s in st])
        W = np.mean([s['W'] for s in st])
        print(f"{dr:>6} {m:>8.0f} {pu:>11.3f} {tp:>11.0f} {ab:>8.0f} {W:>9.0f}")

    print("\nTighter market (n_drivers=250), same sweep — WGC should bite harder:")
    cfg2 = SimConfig(n_drivers=250)
    print(f"{'radius':>6} {'matched':>8} {'mean_pickup':>11} {'tot_pickup':>11} {'abandon':>8} {'W':>9}")
    for dr in [1, 2, 3, 4]:
        st = [run(cfg2, 1.0, dr, s) for s in range(4)]
        m = np.mean([s['match'] for s in st]); pu = np.mean([s['mean_pickup'] for s in st])
        tp = np.mean([s['driver_busy_pickup'] for s in st]); ab = np.mean([s['aband'] for s in st])
        W = np.mean([s['W'] for s in st])
        print(f"{dr:>6} {m:>8.0f} {pu:>11.3f} {tp:>11.0f} {ab:>8.0f} {W:>9.0f}")

    print("\nAbandonment over the day (radius=2, mult=1.0, n=250): peaks vs troughs")
    st = run(cfg2, 1.0, 2, 0)
    pea = st['per_epoch_aband']
    env = RideHailEnv(cfg2); dem = env.lam.sum(1)
    # show correlation of abandonment with demand
    print(f"  corr(demand, abandonment) = {np.corrcoef(dem, pea)[0,1]:.3f}")
    print(f"  abandonment at peak epochs vs trough epochs: "
          f"{pea[np.argsort(dem)[-10:]].mean():.1f} vs {pea[np.argsort(dem)[:10]].mean():.1f}")


if __name__ == "__main__":
    main()
