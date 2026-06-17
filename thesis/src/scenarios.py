"""City-scenario generator: each scenario is a point in observable-context space.

We draw a diverse set via Latin Hypercube Sampling over the context ranges, plus
a handful of named corner cases (slack/uniform, scarce/concentrated/peaky, ...).
The selector (RQ3) sees ONLY the context features; it never sees outcomes.

`demand_elasticity` is the nominal/target rider price sensitivity; it is realized
in the value distribution via `elas_to_sigma` (calibrated in validate.py) and also
consumed directly by the fluid pricing method.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import qmc
from config import ScenarioConfig

# Context ranges
RANGES = {
    "demand_supply_ratio": (0.5, 1.6),
    "spatial_concentration": (0.1, 0.9),
    "temporal_peakedness": (1.2, 4.0),
    "demand_elasticity": (-0.85, -0.30),
    "flex_frac": (0.2, 0.8),
    "trip_length_mean": (1.8, 3.4),
}
GRID_CHOICES = [5, 6]

# Calibrated map target-elasticity -> sigma_v (lognormal spread of rider value).
# Fitted by OLS on the validate.py V1 sweep (sigma_v in [0.6,1.4] spans realized
# elasticity ~ [-0.83,-0.47]); higher sigma_v => less elastic (less negative).
_SIG_A, _SIG_B = 2.12, 1.69   # sigma_v = _SIG_A + _SIG_B * elasticity  (elasticity<0)

def elas_to_sigma(elasticity: float) -> float:
    s = _SIG_A + _SIG_B * elasticity
    return float(np.clip(s, 0.25, 1.5))


def named_scenarios() -> list[ScenarioConfig]:
    """Interpretable corner cases that anchor the analysis."""
    common = dict(n_steps=170, fleet_size=120, patience=4, dispatch_radius=4)
    out = [
        ScenarioConfig(name="slack_uniform", grid_n=5, demand_supply_ratio=0.55,
                       spatial_concentration=0.15, temporal_peakedness=1.4,
                       demand_elasticity=-0.55, flex_frac=0.5, trip_length_mean=2.5,
                       **common),
        ScenarioConfig(name="scarce_concentrated_peaky", grid_n=6, demand_supply_ratio=1.45,
                       spatial_concentration=0.8, temporal_peakedness=3.6,
                       demand_elasticity=-0.55, flex_frac=0.5, trip_length_mean=2.6,
                       **common),
        ScenarioConfig(name="elastic_dispersed", grid_n=5, demand_supply_ratio=1.0,
                       spatial_concentration=0.25, temporal_peakedness=2.0,
                       demand_elasticity=-0.8, flex_frac=0.6, trip_length_mean=2.2,
                       **common),
        ScenarioConfig(name="inelastic_scarce", grid_n=6, demand_supply_ratio=1.4,
                       spatial_concentration=0.6, temporal_peakedness=3.0,
                       demand_elasticity=-0.35, flex_frac=0.4, trip_length_mean=2.8,
                       **common),
        ScenarioConfig(name="mostly_constrained", grid_n=5, demand_supply_ratio=1.1,
                       spatial_concentration=0.7, temporal_peakedness=3.0,
                       demand_elasticity=-0.55, flex_frac=0.25, trip_length_mean=2.5,
                       **common),
        ScenarioConfig(name="mostly_flexible", grid_n=5, demand_supply_ratio=1.1,
                       spatial_concentration=0.7, temporal_peakedness=3.0,
                       demand_elasticity=-0.55, flex_frac=0.75, trip_length_mean=2.5,
                       **common),
    ]
    return out


def lhs_scenarios(n: int, seed: int = 42) -> list[ScenarioConfig]:
    keys = list(RANGES.keys())
    sampler = qmc.LatinHypercube(d=len(keys) + 1, seed=seed)
    sample = sampler.random(n)
    lo = np.array([RANGES[k][0] for k in keys] + [0.0])
    hi = np.array([RANGES[k][1] for k in keys] + [1.0])
    scaled = qmc.scale(sample, lo, hi)
    out = []
    for i, row in enumerate(scaled):
        d = {k: float(row[j]) for j, k in enumerate(keys)}
        grid_n = GRID_CHOICES[int(row[-1] > 0.5)]
        out.append(ScenarioConfig(
            name=f"lhs{i:03d}", grid_n=grid_n, n_steps=170, fleet_size=120,
            patience=4, dispatch_radius=4, **d))
    return out


def all_scenarios(n_lhs: int = 60, seed: int = 42) -> list[ScenarioConfig]:
    return named_scenarios() + lhs_scenarios(n_lhs, seed)


if __name__ == "__main__":
    scns = all_scenarios()
    print(f"{len(scns)} scenarios")
    for s in scns[:8]:
        print(f"  {s.name:26s} dsr={s.demand_supply_ratio:.2f} conc={s.spatial_concentration:.2f} "
              f"peak={s.temporal_peakedness:.2f} elas={s.demand_elasticity:.2f} "
              f"flex={s.flex_frac:.2f} grid={s.grid_n} sigma={elas_to_sigma(s.demand_elasticity):.2f}")
