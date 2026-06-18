"""Reproducibility assertions on the simulator (run: python experiments/test_invariants.py).

Asserts: (1) the four-way welfare identity W = RS+DS+PP holds to fp tolerance and equals the
net-real-surplus form; (2) demand is downward-sloping in price; (3) calibrated elasticity is
in the inelastic range consistent with Castillo (~0.27). Exits non-zero on failure.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv
from ridehail.evaluate import run_constant_seeds, mean_summary

FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        FAIL.append(name)


def main():
    cfg = SimConfig(rebalance_enabled=False)
    mk = lambda: RideHailEnv(cfg, objective="welfare", fix_dispatch_radius=2)

    # 1) consistency identity
    maxrel = 0.0
    for m in [0.9, 1.2, 1.8, 2.5]:
        for s in range(3):
            w = run_constant_seeds(mk, m, 2, [s])[0]
            rel = w["consistency_gap"] / (abs(w["W_components_sum"]) + 1e-9)
            maxrel = max(maxrel, rel)
    check("welfare consistency identity", maxrel < 1e-9, f"max rel gap={maxrel:.1e}")

    # 2) demand downward sloping
    reqs = [mean_summary(run_constant_seeds(mk, m, 2, range(4)))["n_requests"] for m in [1.0, 1.5, 2.5]]
    check("demand downward-sloping in price", reqs[0] > reqs[1] > reqs[2], f"requests={[round(r) for r in reqs]}")

    # 3) elasticity in inelastic range (measured uncongested)
    big = SimConfig(**{**cfg.__dict__, "n_drivers": 2000})
    mkb = lambda: RideHailEnv(big, objective="welfare", fix_dispatch_radius=2)
    rlo = mean_summary(run_constant_seeds(mkb, 0.95, 2, range(6)))["n_requests"]
    rhi = mean_summary(run_constant_seeds(mkb, 1.05, 2, range(6)))["n_requests"]
    eps = -((rhi - rlo) / ((rhi + rlo) / 2)) / ((1.05 - 0.95) / 1.0)
    check("price elasticity inelastic (0.15-0.5)", 0.15 < eps < 0.5, f"elasticity={eps:.3f}")

    if FAIL:
        print(f"\n{len(FAIL)} CHECK(S) FAILED: {FAIL}"); sys.exit(1)
    print("\nAll invariants hold.")


if __name__ == "__main__":
    main()
