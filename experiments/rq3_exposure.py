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

This script also (a) reports the proper paired t-test (not just a mean/s.e.m. ratio),
(b) reports ALL THREE pairwise type contrasts so the reader sees the monotone trend rather
than a single hand-picked contrast, and (c) DECOMPOSES the by-type Delta-DS into its
earnings / opportunity-cost / drive-cost parts, to check whether the by-type ordering is an
economic (earnings) effect or a mechanical artifact of charging the reservation wage on
online time (a reviewer's concern: full-time drivers have the most online epochs).
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv

try:
    from scipy.stats import t as student_t
    def two_sided_p(tstat, df):
        return float(student_t.sf(abs(tstat), df) * 2)
except Exception:                                   # graceful fallback (normal approx)
    from math import erf, sqrt
    def two_sided_p(tstat, df):
        return float(2 * (1 - 0.5 * (1 + erf(abs(tstat) / sqrt(2)))))


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


def by_type_components(summaries):
    """Return {type: {field: per-seed per-capita array}} for surplus and its components."""
    types = list(summaries[0]["driver_by_type"].keys())
    fields = ["surplus", "earnings", "drive_cost", "opp_cost", "online_epochs"]
    bt = {t: {} for t in types}
    for t in types:
        for f in fields:
            bt[t][f] = np.array([s["driver_by_type"][t][f] / max(1, s["driver_by_type"][t]["n_drivers"])
                                 for s in summaries])
    return bt


def paired(delta):
    """mean, s.e.m., t-stat, two-sided p (paired t-test) for a per-seed delta array."""
    n = len(delta); m = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(n))
    tstat = m / se if se > 0 else 0.0
    return dict(mean=m, sem=se, t=tstat, df=n - 1, p=two_sided_p(tstat, n - 1))


def main():
    cfg = SimConfig(rebalance_enabled=False)
    seeds = list(range(100, 116))
    m_high = 2.0
    n = len(seeds)
    uni = run(cfg, "uniform", m_high, seeds)
    sur = run(cfg, "surge", m_high, seeds)
    bu, bs = by_type_components(uni), by_type_components(sur)
    types = list(bu.keys())

    gr = np.mean([u["gross_revenue"] for u in uni])
    avg_mult = float(np.mean([s["mult_weighted"] / max(1, s["n_matched"]) for s in sur]))
    rider_pct = float(np.mean([(s["rider_surplus"] - u["rider_surplus"]) / u["gross_revenue"] * 100
                               for u, s in zip(uni, sur)]))
    print(f"RQ3 targeted: uniform m={m_high} vs peak-targeting surge (base off-peak); seeds={n}")
    print(f"surge avg mult (trip-weighted) = {avg_mult:.2f} (< {m_high}); rider Δ = {rider_pct:+.2f}% of GR")

    # ---- per-type Δ surplus (the headline), with paired test ----
    print(f"\n{'type':10s} {'Δ surplus':>10s} {'sem':>6s} {'p(t)':>7s}   "
          f"[decomp: {'Δearn':>7s} {'Δoppcost':>8s} {'Δdrivecost':>10s} {'Δonline_ep':>10s}]")
    res = {"avg_mult": avg_mult, "rider_pct_GR": rider_pct, "n_seeds": n, "by_type": {}}
    dS = {}
    for t in types:
        d_surp = bs[t]["surplus"] - bu[t]["surplus"]
        d_earn = bs[t]["earnings"] - bu[t]["earnings"]
        d_opp = bs[t]["opp_cost"] - bu[t]["opp_cost"]
        d_dc = bs[t]["drive_cost"] - bu[t]["drive_cost"]
        d_on = bs[t]["online_epochs"] - bu[t]["online_epochs"]
        dS[t] = d_surp
        st = paired(d_surp)
        res["by_type"][t] = dict(uniform=float(bu[t]["surplus"].mean()), surge=float(bs[t]["surplus"].mean()),
                                 delta=st["mean"], delta_sem=st["sem"], p=st["p"],
                                 pct=100 * st["mean"] / float(bu[t]["surplus"].mean()),
                                 d_earnings=float(d_earn.mean()), d_opp_cost=float(d_opp.mean()),
                                 d_drive_cost=float(d_dc.mean()), d_online_epochs=float(d_on.mean()))
        print(f"{t:10s} {st['mean']:10.2f} {st['sem']:6.2f} {st['p']:7.3f}   "
              f"[{d_earn.mean():7.2f} {d_opp.mean():8.2f} {d_dc.mean():10.2f} {d_on.mean():10.2f}]")

    # ---- ALL THREE pairwise contrasts (so the trend, not one hand-picked gap, is visible) ----
    print("\npairwise differential losses (paired t-test):")
    res["contrasts"] = {}
    order = ["fulltime", "parttime", "casual"]
    pairs = [("fulltime", "parttime"), ("parttime", "casual"), ("fulltime", "casual")]
    for a, b in pairs:
        c = paired(dS[a] - dS[b])
        res["contrasts"][f"{a}_minus_{b}"] = c
        print(f"  {a:9s} - {b:9s}: {c['mean']:+.2f} ± {c['sem']:.2f}  t({c['df']})={c['t']:.2f}  p={c['p']:.3f}")
    monotone = all(dS[order[i]].mean() <= dS[order[i + 1]].mean() for i in range(len(order) - 1))
    res["monotone_ft_to_casual_loss"] = bool(monotone)
    print(f"  monotone (full-time loses >= part-time >= casual)? {monotone}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "data"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "data", "rq3_exposure.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
