"""RQ3 (targeted): the timing/exposure mechanism behind Castillo's "long-hours drivers
hurt most".

In the same-objective surge-vs-uniform comparison (Ch5.2/5.3), the learned surge's average
tracks its objective's optimal price, so there is no price-LEVEL drop and drivers do not
lose -- consistent with the decomposition (driver losses are a level-effect phenomenon).
To isolate Castillo's mechanism we impose his actual scenario directly: a high uniform price
versus a peak-targeting surge with the SAME peak prices but a LOWER average (base price
off-peak). This is a price-level reduction concentrated off-peak. We then decompose the
driver surplus change by type. Prediction: full-time drivers (online through the off-peak
periods whose price falls) bear the largest per-capita loss; peak-concentrated casual
drivers are insulated -- the timing/exposure channel, not elasticity.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv


def peak_mask(env):
    dem = env.lam.sum(1)
    return dem >= np.median(dem)   # high-demand (peak) epochs


def run(cfg, mode, m_high, seeds, dispatch=2):
    """mode: 'uniform' -> m_high everywhere; 'surge' -> m_high at peaks, base(1.0) off-peak."""
    out = []
    for sd in seeds:
        env = RideHailEnv(cfg, objective="welfare", fix_dispatch_radius=dispatch)
        env.reset(seed=sd)
        pk = peak_mask(env)
        done = False; t = 0
        while not done:
            if mode == "uniform":
                surge = np.full(env.Z, m_high)
            else:
                surge = np.full(env.Z, m_high if pk[t] else 1.0)
            _, _, done, info = env.apply(surge, dispatch, np.zeros(env.Z)); t += 1
        out.append(info["welfare"])
    return out


def by_type_percap(summaries):
    bt = {}
    for t in summaries[0]["driver_by_type"]:
        v = np.mean([s["driver_by_type"][t]["surplus"] / max(1, s["driver_by_type"][t]["n_drivers"])
                     for s in summaries])
        bt[t] = float(v)
    return bt


def main():
    cfg = SimConfig(rebalance_enabled=False)
    seeds = range(100, 116)
    m_high = 2.0
    uni = run(cfg, "uniform", m_high, seeds)
    sur = run(cfg, "surge", m_high, seeds)
    bu, bs = by_type_percap(uni), by_type_percap(sur)
    gr = np.mean([u["gross_revenue"] for u in uni])
    print(f"RQ3 targeted: uniform m={m_high} vs peak-targeting surge (base off-peak)")
    print(f"surge avg mult (trip-weighted) = "
          f"{np.mean([s['mult_weighted']/max(1,s['n_matched']) for s in sur]):.2f} (< {m_high})")
    print(f"{'type':10s} {'uniform $/drv':>13s} {'surge $/drv':>12s} {'Δ $/drv':>9s} {'Δ %':>7s}")
    res = {}
    for t in bu:
        d = bs[t] - bu[t]
        res[t] = dict(uniform=bu[t], surge=bs[t], delta=d, pct=100 * d / bu[t])
        print(f"{t:10s} {bu[t]:13.2f} {bs[t]:12.2f} {d:9.2f} {100*d/bu[t]:6.1f}%")
    # rider/total check
    ru = np.mean([u["rider_surplus"] for u in uni]); rs = np.mean([s["rider_surplus"] for s in sur])
    print(f"rider surplus Δ = {(rs-ru)/gr*100:+.2f}% of GR (riders gain from the lower off-peak price)")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "data"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "data", "rq3_exposure.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
