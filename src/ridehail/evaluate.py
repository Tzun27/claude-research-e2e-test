"""Evaluation utilities: run policies over seeds, aggregate welfare, score objectives."""
from __future__ import annotations

import numpy as np


WELFARE_KEYS = ["total_welfare", "rider_surplus", "driver_surplus", "platform_profit",
                "gross_revenue", "n_requests", "n_matched", "n_abandon",
                "total_pickup", "total_drive_cost", "total_opp_cost", "surge_trips"]


def fair_dispersion(s):
    """Across-type dispersion (max-min) of per-capita driver surplus, from a welfare summary."""
    bt = s.get("driver_by_type", {})
    if not bt:
        return 0.0
    pc = [v["surplus"] / max(1, v["n_drivers"]) for v in bt.values()]
    return float(max(pc) - min(pc))


def objective_value(s, objective, alpha_pi=0.4, alpha_R=1.0, alpha_D=0.3):
    if objective == "profit":
        return s["platform_profit"]
    if objective == "throughput":
        return s["n_matched"]
    if objective == "welfare":
        return s["total_welfare"]
    if objective == "welfare_weighted":
        return alpha_pi * s["platform_profit"] + alpha_R * s["rider_surplus"] + alpha_D * s["driver_surplus"]
    raise ValueError(objective)


def run_policy_seeds(make_env, policy_fn, seeds):
    """policy_fn(obs)->raw_action. Returns list of welfare summaries."""
    out = []
    for sd in seeds:
        env = make_env()
        w = env.run_episode(policy_fn, seed=sd)
        out.append(w)
    return out


def run_constant_seeds(make_env, mult, dispatch_radius, seeds, rebalance_scale=0.0):
    """Directly drive env.apply with a constant spatial multiplier (uniform baseline)."""
    out = []
    for sd in seeds:
        env = make_env()
        env.reset(seed=sd)
        surge = np.full(env.Z, float(mult))
        reb = np.zeros(env.Z)
        if rebalance_scale > 0:
            reb = np.full(env.Z, rebalance_scale * 0.5 * float(np.mean(env.drivers.rho)))
        done = False
        while not done:
            dr = dispatch_radius if env.fix_dispatch_radius is None else env.fix_dispatch_radius
            _, _, done, info = env.apply(surge, int(dr), reb)
        out.append(info["welfare"])
    return out


def aggregate(summaries):
    """Mean/std/sem over seeds for the welfare keys and by-type driver surplus."""
    agg = {}
    for k in WELFARE_KEYS:
        vals = np.array([s[k] for s in summaries], float)
        agg[k] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                      sem=float(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0))
    # by-type driver surplus
    types = summaries[0].get("driver_by_type", {})
    agg["driver_by_type"] = {}
    for tname in types:
        for field in ["surplus", "earnings", "opp_cost", "online_epochs", "trips", "n_drivers"]:
            vals = np.array([s["driver_by_type"][tname][field] for s in summaries], float)
            agg["driver_by_type"].setdefault(tname, {})[field] = dict(
                mean=float(vals.mean()), sem=float(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0))
    agg["n_seeds"] = len(summaries)
    return agg


def mean_summary(summaries):
    """Plain dict of means (for objective scoring during grid search)."""
    out = {}
    for k in WELFARE_KEYS:
        out[k] = float(np.mean([s[k] for s in summaries]))
    return out
