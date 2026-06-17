"""Simulator validity checks (DESIGN RQ2):
  (V1) realized demand price elasticity vs target (Cohen et al. 2016 ~ -0.55)
  (V2) emergent wait-vs-idle-density follows the square-root law (Arnott 1996)
  (V3) Castillo (2025) qualitative incidence: in a SCARCE market, surge raises
       total welfare & rider surplus, lowers driver surplus, and hurts the
       CONSTRAINED (low-mobility) drivers more than the flexible ones.
All numbers are simulator-computed; Castillo is a target, never an input.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from dataclasses import replace
from config import ScenarioConfig
from simulator import Market
from methods import make_methods, FluidPricing
from engine import run_combo, pretrain_value_table

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def measure_elasticity(scn, sigma_v, n_seeds=6, bump=0.10):
    """Arc elasticity of accepted demand wrt a uniform price bump."""
    base, hi = [], []
    for s in range(n_seeds):
        sc = replace(scn, seed=1000 + s)
        m, p, r = make_methods(("M2", "P1", "R1"))
        mkt = Market(sc, p, m, r, sigma_v=sigma_v)
        mkt.run(); base.append(mkt.n_requests)
        # bump price via a flat surge floor by scaling FARE through a temp method
        m2, p2, r2 = make_methods(("M2", "P1", "R1"))
        mkt2 = Market(sc, _ConstSurge(1 + bump), m2, r2, sigma_v=sigma_v)
        mkt2.run(); hi.append(mkt2.n_requests)
    q0, q1 = np.mean(base), np.mean(hi)
    # arc elasticity: %dQ / %dP
    return (q1 - q0) / ((q1 + q0) / 2) / (bump / (1 + bump / 2))


class _ConstSurge:
    code = "PB"
    def __init__(self, m): self.m = m
    def surge(self, mkt, t): return np.full(mkt.Z, self.m)


def calibrate_sigma(scn, target=-0.55):
    """Find sigma_v giving realized elasticity ~ target (monotone search)."""
    grid = np.linspace(0.25, 1.4, 12)
    best, beste = None, 1e9
    rows = []
    for sg in grid:
        e = measure_elasticity(scn, sg)
        rows.append((sg, e))
        if abs(e - target) < beste:
            best, beste = sg, abs(e - target)
    return best, rows


def sqrt_law(scn, sigma_v):
    """Square-root law: bin matches by LOCAL idle density and fit mean pickup
    distance = k * rho^-a (Larson-Odoni 1981; Arnott 1996 => a ~ 0.5).

    Measured under FLAT pricing across several supply levels so a range of local
    idle densities is observed and pickups reflect matching geometry, not surge
    rationing.
    """
    dens, dist = [], []
    for fleet in (70, 100, 130, 170, 210):
        sc = replace(scn, seed=7, fleet_size=fleet, demand_supply_ratio=0.8)
        m, p, r = make_methods(("M2", "P1", "R1"))
        mkt = Market(sc, p, m, r, sigma_v=sigma_v, collect_traces=True)
        mkt.run()
        dens.extend(mkt.trace_local_density)
        dist.extend(mkt.trace_pickup_dist)
    dens = np.array(dens, float); dist = np.array(dist, float)
    mask = dens > 0.2
    dens, dist = dens[mask], dist[mask]
    # bin by density quantiles, take mean pickup distance per bin
    qs = np.quantile(dens, np.linspace(0, 1, 13))
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        b = (dens >= lo) & (dens < hi if hi < qs[-1] else dens <= hi)
        if b.sum() >= 20:
            xs.append(dens[b].mean()); ys.append(dist[b].mean())
    xs, ys = np.log(np.array(xs)), np.log(np.array(ys) + 1e-6)
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a = -coef[1]
    corr = np.corrcoef(xs, ys)[0, 1]
    return a, corr, len(xs)


KEYS = ["total_welfare", "rider_surplus", "driver_surplus", "platform_profit",
        "gmv", "service_rate", "mean_wait_min", "ds_flexible", "ds_constrained",
        "earn_rate_flex", "earn_rate_cons", "mean_surge",
        "ds_flexible_per_capita", "ds_constrained_per_capita"]


def _avg_pricing(scn, sigma_v, pricing_obj, seeds=12):
    """Average metrics over seeds for a fixed (M2, <pricing>, R1) controller."""
    acc = {k: [] for k in KEYS}
    for s in range(seeds):
        m, _, r = make_methods(("M2", "P1", "R1"))
        mkt = Market(replace(scn, seed=s), pricing_obj, m, r, sigma_v=sigma_v)
        met = mkt.run()
        for k in KEYS:
            acc[k].append(met[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def castillo_incidence(sigma_v):
    """V3: Castillo's comparison -- smart (fluid, elasticity-aware) surge vs a
    REVENUE-MATCHED uniform multiplier (his optimal uniform = 1.174). This
    isolates the ALLOCATION effect of surge from the price-level effect. In the
    moderate-scarcity regime that matches Castillo's Houston setting, expect:
    surge raises rider surplus & total welfare, slightly LOWERS driver surplus,
    and hurts CONSTRAINED (low-mobility) drivers while flexible drivers gain.
    """
    from methods import FluidPricing
    scn = ScenarioConfig(name="houston_peak", grid_n=6, n_steps=180,
                         demand_supply_ratio=0.95, fleet_size=120,
                         spatial_concentration=0.65, temporal_peakedness=3.5,
                         demand_elasticity=-0.55, flex_frac=0.5,
                         trip_length_mean=2.5, patience=4, dispatch_radius=4,
                         seed=0)
    surge = _avg_pricing(scn, sigma_v, FluidPricing())
    mu = surge["mean_surge"]                          # match the average price level
    uniform = _avg_pricing(scn, sigma_v, _ConstSurge(mu))
    return uniform, surge, mu


def regime_sweep(sigma_v, seeds=10):
    """Castillo incidence across market regimes (dsr x concentration). Saves CSV.
    Shows the incidence direction (constrained worse off than flexible) is robust,
    while the aggregate driver-surplus sign brackets Castillo's -0.98%.
    """
    rows = []
    for dsr in [0.8, 0.95, 1.1, 1.25]:
        for conc in [0.6, 0.75]:
            scn = ScenarioConfig(name="hp", grid_n=6, n_steps=180,
                                 demand_supply_ratio=dsr, fleet_size=120,
                                 spatial_concentration=conc, temporal_peakedness=3.5,
                                 demand_elasticity=-0.55, flex_frac=0.5,
                                 trip_length_mean=2.5, patience=4, dispatch_radius=4)
            surge = _avg_pricing(scn, sigma_v, FluidPricing(), seeds=seeds)
            mu = surge["mean_surge"]
            uni = _avg_pricing(scn, sigma_v, _ConstSurge(mu), seeds=seeds)
            g = uni["gmv"]
            rows.append({
                "dsr": dsr, "concentration": conc, "mean_surge": round(mu, 3),
                "d_welfare_pct": round((surge["total_welfare"] - uni["total_welfare"]) / g * 100, 2),
                "d_rider_pct": round((surge["rider_surplus"] - uni["rider_surplus"]) / g * 100, 2),
                "d_driver_pct": round((surge["driver_surplus"] - uni["driver_surplus"]) / g * 100, 2),
                "d_platform_pct": round((surge["platform_profit"] - uni["platform_profit"]) / g * 100, 2),
                "d_ds_flexible": round(surge["ds_flexible"] - uni["ds_flexible"], 1),
                "d_ds_constrained": round(surge["ds_constrained"] - uni["ds_constrained"], 1),
                "d_earn_flex": round(surge["earn_rate_flex"] - uni["earn_rate_flex"], 3),
                "d_earn_cons": round(surge["earn_rate_cons"] - uni["earn_rate_cons"], 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "castillo_regime_sweep.csv"), index=False)
    return df


if __name__ == "__main__":
    import sys
    from scenarios import elas_to_sigma
    base = ScenarioConfig(name="calib", grid_n=5, n_steps=140,
                          demand_supply_ratio=0.9, fleet_size=120,
                          spatial_concentration=0.5, temporal_peakedness=2.5,
                          demand_elasticity=-0.55, flex_frac=0.5, seed=0)
    full = "--full" in sys.argv
    print("=== V1: demand elasticity at calibrated sigma_v ===")
    if full:
        sg, rows = calibrate_sigma(base, target=-0.55)
        for s, e in rows:
            print(f"  sigma_v={s:.3f}  elasticity={e:+.3f}")
        print(f"  -> chosen sigma_v={sg:.3f}")
    else:
        sg = elas_to_sigma(-0.55)
        e = measure_elasticity(base, sg)
        print(f"  sigma_v={sg:.3f} (from map) -> realized elasticity={e:+.3f} (target -0.55)")

    print("\n=== V2: square-root law (mean pickup distance vs local idle density) ===")
    a, corr, nb = sqrt_law(base, sg)
    print(f"  exponent a={a:.3f} (square-root law => ~0.5), |corr|={abs(corr):.3f}, bins={nb}")

    print("\n=== V3: Castillo incidence (surge vs REVENUE-MATCHED uniform) ===")
    uniform, surge, mu = castillo_incidence(sg)
    print(f"  counterfactual = uniform multiplier {mu:.3f} (matches surge avg level)")
    def pct(a, b):  # change as % of uniform gmv (mirrors Castillo '% of gross revenue')
        return (b - a) / uniform["gmv"] * 100
    print(f"  {'metric':26s} {'uniform':>10s} {'surge':>10s} {'Δ(%GMV)':>9s}")
    for k in ["total_welfare", "rider_surplus", "driver_surplus", "platform_profit"]:
        print(f"  {k:26s} {uniform[k]:10.1f} {surge[k]:10.1f} {pct(uniform[k],surge[k]):+9.2f}")
    print(f"  {'service_rate':26s} {uniform['service_rate']:10.3f} {surge['service_rate']:10.3f}")
    print(f"  {'mean_wait_min':26s} {uniform['mean_wait_min']:10.2f} {surge['mean_wait_min']:10.2f}")
    print("  -- driver incidence by type (Δ surge - uniform), Castillo: constrained hurt most --")
    for k in ["ds_flexible", "ds_constrained", "earn_rate_flex", "earn_rate_cons"]:
        print(f"  {k:26s} {uniform[k]:10.2f} {surge[k]:10.2f} {surge[k]-uniform[k]:+9.2f}")

    print("\n=== V3b: Castillo incidence across market regimes (saving CSV) ===")
    sweep = regime_sweep(sg)
    print(sweep.to_string(index=False))

    # persist a machine-readable validation report for the thesis
    report = {
        "V1_realized_elasticity": round(float(e), 4),
        "V1_target_elasticity": -0.55,
        "V1_sigma_v": round(float(sg), 4),
        "V2_sqrt_law_exponent": round(float(a), 4),
        "V2_corr": round(float(abs(corr)), 4),
        "V3_counterfactual_uniform_mult": round(float(mu), 4),
        "V3_d_welfare_pctGMV": round(pct(uniform["total_welfare"], surge["total_welfare"]), 3),
        "V3_d_rider_pctGMV": round(pct(uniform["rider_surplus"], surge["rider_surplus"]), 3),
        "V3_d_driver_pctGMV": round(pct(uniform["driver_surplus"], surge["driver_surplus"]), 3),
        "V3_d_platform_pctGMV": round(pct(uniform["platform_profit"], surge["platform_profit"]), 3),
        "V3_d_ds_flexible": round(surge["ds_flexible"] - uniform["ds_flexible"], 2),
        "V3_d_ds_constrained": round(surge["ds_constrained"] - uniform["ds_constrained"], 2),
        "V3_d_earn_flex": round(surge["earn_rate_flex"] - uniform["earn_rate_flex"], 3),
        "V3_d_earn_cons": round(surge["earn_rate_cons"] - uniform["earn_rate_cons"], 3),
    }
    with open(os.path.join(RESULTS_DIR, "validation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved results/validation.json and results/castillo_regime_sweep.csv")
