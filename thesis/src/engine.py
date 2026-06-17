"""Engine helpers: value-table pretraining and single-combo runs.

Value-based methods (M3, R3) consume a spatiotemporal value table V[zone,bucket].
Following Xu (2018) -- "learning + planning" -- we learn V OFFLINE on a bootstrap
policy (optimal-bipartite matching, flat pricing, no rebalancing) and then FREEZE
it for the planning/online phase. The frozen table is cached per scenario so it is
not relearned for every controller combo.
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace

from config import ScenarioConfig
from simulator import Market
from methods import make_methods, combo_needs_value
from scenarios import elas_to_sigma


def pretrain_value_table(scn: ScenarioConfig, n_episodes: int = 4) -> np.ndarray:
    """Learn V[zone,bucket] via TD on a bootstrap policy across a few episodes."""
    V = None
    sig = elas_to_sigma(scn.demand_elasticity)
    for e in range(n_episodes):
        s = replace(scn, seed=scn.seed * 100 + e + 1)
        m, p, r = make_methods(("M2", "P1", "R1"))
        mkt = Market(s, p, m, r, value_table=V, learn_value=True, sigma_v=sig)
        mkt.run()
        V = mkt.V                       # carry learned table forward
    return V


def run_combo(scn: ScenarioConfig, combo, value_table=None, seed=None):
    """Run one controller combo on one scenario; return metrics dict + combo tag."""
    s = scn if seed is None else replace(scn, seed=seed)
    matching, pricing, rebal = make_methods(combo)
    mkt = Market(s, pricing, matching, rebal, value_table=value_table,
                 sigma_v=elas_to_sigma(s.demand_elasticity))
    met = mkt.run()
    met["combo"] = "+".join(combo)
    met["match"], met["price"], met["rebal"] = combo
    return met


if __name__ == "__main__":
    import time
    from methods import all_combos
    scn = ScenarioConfig(name="smoke", seed=1)
    print("pretraining value table ...")
    t0 = time.time()
    V = pretrain_value_table(scn)
    print(f"  V shape {V.shape}, range [{V.min():.2f},{V.max():.2f}], {time.time()-t0:.2f}s")

    print("\nsingle run M1+P1+R1:")
    t0 = time.time()
    m = run_combo(scn, ("M1", "P1", "R1"))
    dt = time.time() - t0
    for k in ["total_welfare", "rider_surplus", "driver_surplus", "ds_flexible",
              "ds_constrained", "platform_profit", "gmv", "service_rate",
              "mean_wait_min", "mean_surge", "n_requests", "n_completed"]:
        print(f"  {k:24s} {m[k]:.3f}")
    print(f"  run time {dt*1000:.1f} ms")

    print("\nall 27 combos (welfare, gmv, service_rate, mean_surge, ds_cons_pc):")
    V = pretrain_value_table(scn)
    t0 = time.time()
    for combo in all_combos():
        vt = V if combo_needs_value(combo) else None
        m = run_combo(scn, combo, value_table=vt)
        print(f"  {m['combo']:12s} W={m['total_welfare']:9.1f} GMV={m['gmv']:9.1f} "
              f"SR={m['service_rate']:.3f} surge={m['mean_surge']:.3f} "
              f"DScons/c={m['ds_constrained_per_capita']:7.2f} DSflex/c={m['ds_flexible_per_capita']:7.2f}")
    print(f"  27 combos in {time.time()-t0:.2f}s")
