"""Calibration constants and scenario/driver configuration.

All economic constants are sourced from INDEPENDENT literature (NOT Castillo's
welfare numbers, which we reserve as a validation target). See DESIGN.md §7 and
the thesis Methodology chapter for citations. Values are sanity-bounded and the
qualitative results are shown to be robust to them in the sensitivity analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

# --------------------------------------------------------------------------
# Economic calibration constants (independent sources)
# --------------------------------------------------------------------------
# Platform commission / take rate. Uber/Lyft public figures ~ 20-30%.
COMMISSION = 0.25

# Fare structure (USD). Simplified from Castillo's unsurged fare
#   $3.30 + $0.87*miles + $0.11*minutes (booking fee $2.30).
# We collapse to a base + per-minute term on the grid (1 cell ~ CELL_MIN minutes,
# trip "minutes" = path_cells * CELL_MIN). Magnitudes are illustrative; only
# RELATIVE welfare comparisons across controllers are interpreted.
FARE_BASE = 2.50          # booking/base component
FARE_PER_MIN = 0.40       # per-minute-of-trip component
CELL_MIN = 3.0            # minutes to traverse one grid cell (1 sim step)

# Rider value of time (USD/min). Goldszmidt et al. (2020, NBER w28208) estimate
# VOT ~ $19/hr (~$0.317/min) from two Lyft field experiments; waiting time is
# valued ~1.5-2x in-vehicle time (Abrantes & Wardman 2011). We use ~1.6x the
# base => ~$0.50/min for waiting, and vary it in sensitivity analysis.
VOT_RIDER = 0.50          # USD per minute of expected wait

# Demand price elasticity target (Cohen, Hahn, Hall, Levitt, Metcalfe 2016,
# NBER w22627): ride-hailing demand elasticity ~ -0.5 .. -0.6.
TARGET_DEMAND_ELASTICITY = -0.55

# Driver opportunity cost / reservation wage (USD per online minute) and per-cell
# driving cost. Driving cost ~ $0.26/mile (Castillo cites internal Uber est.);
# at city speeds one cell ~ a fraction of a mile -> small per-cell cost.
DRIVER_OPP_COST_PER_MIN = 0.18   # opportunity cost of driver time (USD/min online)
DRIVING_COST_PER_CELL = 0.20     # fuel/depreciation per cell driven

# Heterogeneous driver labor-supply elasticities. Aggregate ride-hailing labor
# supply is highly elastic: ~1.7 (Chen, Chevalier, Rossi & Oehlsen 2019, JPE) and
# ~1.2-1.4 intertemporal substitution (Angrist, Caldwell & Hall 2021, AEJ:Applied).
# By-type entry elasticities differ (Caldwell & Oehlsen 2022, the source Castillo
# uses for the supply side: full-time men ~0.3 ... part-time women ~1.0). We model
# the RELATIVE gap between a flexible (elastic, part-time-like) and a constrained
# (inelastic, fixed-shift) type. NOTE: this is an abstract spatial-flexibility /
# hours axis, NOT a gender mechanism. Cook et al. (2021) show the gender earnings
# gap is driven by driving speed, experience, and LOCATION sorting -- not an
# inability to chase surge -- so we model the location/hours channel directly.
SUPPLY_ELASTICITY = {"flexible": 1.2, "constrained": 0.3}

# Surge multipliers clipped to [SURGE_FLOOR, SURGE_CAP]. The floor < 1 lets
# spatial/temporal pricing LOWER fares where supply is slack (this is essential:
# Castillo's surge raises peak prices but lowers average/off-peak prices, which is
# how stuck drivers are hurt). The cap is a regulatory/UX bound.
SURGE_CAP = 3.0
SURGE_FLOOR = 0.7

# --------------------------------------------------------------------------
# Driver-type behavioural parameters (the heterogeneity mechanism, DESIGN §3)
# --------------------------------------------------------------------------
# H1 surge-chasing / spatial flexibility: probability an idle driver of each type
#    actually relocates toward a high-value zone when given the chance. Kept
#    MODEST to match Castillo's estimated limited mobility (his movement
#    responsiveness delta=0.09 implies a ~$3 surge lifts P(move) only ~0.05);
#    flexible drivers are more mobile than constrained, but neither chases hard.
CHASE_PROB = {"flexible": 0.45, "constrained": 0.10}
# Radius (in cells) a constrained driver is willing to stray from its home zone.
HOME_RADIUS = {"flexible": 99, "constrained": 1}
# H2 extensive margin: how strongly each type enters/exits with recent earnings.
#    Implemented via SUPPLY_ELASTICITY above in the labor-supply update.

# --------------------------------------------------------------------------
# Scenario configuration
# --------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    """A city scenario = a point in observable-context feature space.

    The fields tagged (CONTEXT) are the observable features a planner could
    measure WITHOUT running controller experiments; they feed the selector.
    """
    name: str = "default"
    grid_n: int = 5                     # NxN zones
    n_steps: int = 160                  # sim steps (1 step = CELL_MIN minutes)
    # (CONTEXT) demand/supply pressure: mean potential requests per driver per step
    demand_supply_ratio: float = 0.6
    fleet_size: int = 120
    # (CONTEXT) spatial concentration of demand: 0 = uniform, 1 = single CBD hotspot
    spatial_concentration: float = 0.5
    # (CONTEXT) temporal peakedness: ratio of peak to trough arrival rate
    temporal_peakedness: float = 2.5
    # (CONTEXT) demand price elasticity (riders' price sensitivity)
    demand_elasticity: float = -0.55
    # (CONTEXT) fraction of fleet that is the flexible type
    flex_frac: float = 0.5
    # (CONTEXT) mean trip length in cells
    trip_length_mean: float = 2.5
    # rider patience (steps a request waits before abandoning)
    patience: int = 4
    # max dispatch radius (cells) for matching
    dispatch_radius: int = 4
    seed: int = 0

    def context_features(self) -> dict:
        """Observable features exposed to the selector (no controller results)."""
        return {
            "demand_supply_ratio": self.demand_supply_ratio,
            "spatial_concentration": self.spatial_concentration,
            "temporal_peakedness": self.temporal_peakedness,
            "demand_elasticity": self.demand_elasticity,
            "flex_frac": self.flex_frac,
            "trip_length_mean": self.trip_length_mean,
            "grid_n": float(self.grid_n),
            "fleet_size": float(self.fleet_size),
        }

    def to_dict(self) -> dict:
        return asdict(self)


# Driver status codes
OFFLINE, IDLE, TO_PICKUP, ON_TRIP = 0, 1, 2, 3
FLEXIBLE, CONSTRAINED = 0, 1
TYPE_NAME = {FLEXIBLE: "flexible", CONSTRAINED: "constrained"}
