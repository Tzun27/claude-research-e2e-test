"""Baseline pricing policies.

  * uniform_search: the Castillo (2025) uniform-pricing counterfactual -- a single constant
    multiplier for the whole market, with its level optimized for the platform's objective.
  * myopic_surge: a Besbes (2021)-style local/myopic surge heuristic that prices each zone
    from its instantaneous local supply-demand imbalance, ignoring network effects.
"""
from __future__ import annotations

import numpy as np

from .evaluate import run_constant_seeds, mean_summary, objective_value


def uniform_search(make_env, objective, seeds, mult_grid=None,
                   alpha_pi=0.4, alpha_R=1.0, alpha_D=0.3, dispatch_radius=2):
    """Grid-search the best constant multiplier for the objective. Returns (best_mult, info)."""
    if mult_grid is None:
        mult_grid = np.round(np.arange(0.8, 3.01, 0.1), 2)
    best = None
    table = []
    for m in mult_grid:
        sums = run_constant_seeds(make_env, m, dispatch_radius, seeds)
        ms = mean_summary(sums)
        val = objective_value(ms, objective, alpha_pi, alpha_R, alpha_D)
        table.append((float(m), val, ms))
        if best is None or val > best[1]:
            best = (float(m), val, sums)
    return best[0], dict(best_value=best[1], best_summaries=best[2], table=table)


class MyopicSurge:
    """Besbes-style local myopic surge: surge_z = clip(1 + k*(demand_z - supply_z)/scale).

    Prices the local imbalance only; no anticipation of network/relocation effects. Serves
    as the 'local-only surge leaves money on the table' baseline (Besbes-Castro-Lobel 2021).
    """
    def __init__(self, env, gain=0.6, dispatch_radius=2):
        self.env = env
        self.gain = gain
        self.dispatch_radius = dispatch_radius

    def __call__(self, obs):
        env = self.env
        cfg = env.cfg
        t = min(env.t, cfg.T - 1)
        avail = env.drivers.available_mask(t)
        supply = np.bincount(env.drivers.loc[avail], minlength=env.Z).astype(float)
        demand = env.lam[t]
        imbalance = (demand - supply) / (demand + supply + 1.0)
        surge = np.clip(1.0 + self.gain * imbalance, cfg.surge_min, cfg.surge_max)
        # build a raw action that decodes to this surge (invert sigmoid), no rebalance
        frac = (surge - cfg.surge_min) / (cfg.surge_max - cfg.surge_min)
        frac = np.clip(frac, 1e-4, 1 - 1e-4)
        raw = np.zeros(env.act_dim)
        raw[:env.Z] = np.log(frac / (1 - frac))
        # dispatch handled by fix_dispatch_radius if set; else aim for self.dispatch_radius
        df = (self.dispatch_radius - cfg.dispatch_radius_min) / max(1, (cfg.dispatch_radius_max - cfg.dispatch_radius_min))
        df = min(max(df, 1e-3), 1 - 1e-3)
        raw[env.Z] = np.log(df / (1 - df))
        raw[env.Z + 1:] = -10.0
        return raw
