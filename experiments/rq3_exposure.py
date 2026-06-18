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
    """Return per-type per-capita surplus: {type: [per-seed values]}."""
    bt = {}
    for t in summaries[0]["driver_by_type"]:
        bt[t] = np.array([s["driver_by_type"][t]["surplus"] / max(1, s["driver_by_type"][t]["n_drivers"])
                          for s in summaries])
    return bt


def main():
    cfg = SimConfig(rebalance_enabled=False)
    seeds = range(100, 116)
    m_high = 2.0
    seeds = list(seeds)
    uni = run(cfg, "uniform", m_high, seeds)
    sur = run(cfg, "surge", m_high, seeds)
    bu, bs = by_type_percap(uni), by_type_percap(sur)   # {type: per-seed array}
    n = len(seeds)
    gr = np.mean([u["gross_revenue"] for u in uni])
    avg_mult = float(np.mean([s["mult_weighted"] / max(1, s["n_matched"]) for s in sur]))
    rider_pct = float(np.mean([(s["rider_surplus"] - u["rider_surplus"]) / u["gross_revenue"] * 100
                               for u, s in zip(uni, sur)]))
    print(f"RQ3 targeted: uniform m={m_high} vs peak-targeting surge (base off-peak); seeds={n}")
    print(f"surge avg mult (trip-weighted) = {avg_mult:.2f} (< {m_high}); rider Δ = {rider_pct:+.2f}% of GR")
    print(f"{'type':10s} {'Δ $/drv':>9s} {'sem':>6s} {'Δ %':>7s}")
    res = {"avg_mult": avg_mult, "rider_pct_GR": rider_pct, "n_seeds": n, "by_type": {}}
    for t in bu:
        d = bs[t] - bu[t]                       # per-seed paired delta
        mean_d = float(d.mean()); sem_d = float(d.std(ddof=1) / np.sqrt(n))
        res["by_type"][t] = dict(uniform=float(bu[t].mean()), surge=float(bs[t].mean()),
                                 delta=mean_d, delta_sem=sem_d, pct=100 * mean_d / float(bu[t].mean()))
        print(f"{t:10s} {mean_d:9.2f} {sem_d:6.2f} {100*mean_d/float(bu[t].mean()):6.1f}%")
    # pairwise significance of the full-time vs casual ordering
    fc = (bs["fulltime"] - bu["fulltime"]) - (bs["casual"] - bu["casual"])
    print(f"full-time vs casual loss difference: {fc.mean():.2f} +/- {fc.std(ddof=1)/np.sqrt(n):.2f} "
          f"(={fc.mean()/(fc.std(ddof=1)/np.sqrt(n)):.2f} s.e.m.)")
    res["fulltime_minus_casual"] = dict(mean=float(fc.mean()), sem=float(fc.std(ddof=1) / np.sqrt(n)))
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "data"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "data", "rq3_exposure.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
