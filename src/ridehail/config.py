"""Configuration for the spatial ride-hailing market simulator.

All defaults are grounded in the literature where possible; the grounding note
for each parameter is given inline. Sources:
  - Castillo (2025) "Who Benefits from Surge Pricing?" Econometrica 93(5).  [C25]
  - Gammelli et al. (2021) GNN-RL for AMoD, CDC.                            [G21]
  - Lin et al. (2018) Large-Scale Fleet Management MARL, KDD.              [L18]
  - Xu et al. (2018) Large-Scale Order Dispatch, Didi, KDD.                [X18]
  - Zhang et al. (2023) Future-Aware Pricing & Matching, AAAI.            [Z23]

Monetary units are abstract "dollars," scaled to roughly match C25's Houston
UberX fare structure (base + per-mile + per-minute, commission ~26%).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class SimConfig:
    # ----- Spatial structure -----
    grid_size: int = 5            # G x G grid. 5x5=25 zones, single central CBD.
                                  # G21 uses 4x4 as the tractable default; 5x5 gives a clean center.
    cell_km: float = 1.0          # ~1 km^2 cells (G21 NYC). Sets distance->fare scale.
    move_speed_cells: int = 1     # cells a vehicle traverses per epoch (L18: 1 hop/step).

    # ----- Temporal structure -----
    T: int = 48                   # epochs per episode (one stylized operating "day").
                                  # L18 uses 144x10min; we compress to 48 for cheap RL episodes.
    epoch_min: float = 10.0       # nominal minutes/epoch (for flavor / value-of-time scaling).

    # ----- Demand (elastic, discrete-choice) -----
    # Potential requests per zone per epoch ~ Poisson(lambda_{z,t}).
    base_demand: float = 3.0      # baseline potential requests / zone / epoch (off-peak floor).
    peak_demand: float = 14.0     # additional peak intensity (Gaussian peaks in time).
    # Bimodal temporal profile (AM/PM peaks): centers & widths as fractions of T.
    peak_centers: tuple = (0.25, 0.70)
    peak_width: float = 0.09      # std of each Gaussian peak (fraction of T).
    cbd_pull: float = 0.65        # fraction of peak demand directed toward/away from CBD (OD imbalance).

    # Rider willingness-to-pay: v = fare_at_mult1 * m, with m ~ LogNormal(mu, sigma) the
    # WTP-to-base-fare ratio. Calibrated so the short-run price elasticity at the operating
    # point ~0.27 (Castillo 2025: 0.268, inelastic) with high baseline acceptance.
    wtp_log_mu: float = 1.0       # median WTP/fare ratio = exp(1.0) ~ 2.7x
    wtp_log_sigma: float = 1.4
    wtp_ratio_cap: float = 4.0    # truncate the WTP/fare tail so rider surplus stays a
                                  # moderate multiple of revenue (keeps the four welfare
                                  # components comparable in scale, as in Castillo 2025).
    value_of_wait_per_epoch: float = 1.40   # rider disutility per epoch waited ($/epoch).
                                            # C25 value of time ~$2.25/min; riders value time highly.

    # ----- Fare structure (C25 Houston UberX, p.1818, scaled) -----
    fare_base: float = 3.0        # base fare ($). C25: $3.30 + $2.30 booking fee.
    fare_per_cell: float = 1.6    # $/cell of trip distance (~ $0.87/mi * miles + $0.11/min).
    commission: float = 0.25      # platform take rate. C25: 26.3%. (flagged as ~assumption)
    drive_cost_per_cell: float = 0.30  # driver fuel/maintenance per cell moved. C25 $0.26/mi; L18 ~0.5RMB/km.

    # ----- Surge / control bounds -----
    surge_min: float = 0.8        # Z23 multiplier action set lower bound.
    surge_max: float = 4.0        # = WTP ratio cap; beyond this demand vanishes, so profit-max
                                  # finds an interior optimum rather than binding on the cap.
    dispatch_radius_min: int = 1  # matching lever: min max-pickup radius (cells).
    dispatch_radius_max: int = 4  # max max-pickup radius. C25 abandons beyond ~10km (~a few cells).
    abandon_radius: int = 5       # hard cap: no driver within this radius -> rider abandons.
    # Congestion (matching-function) friction: when local demand exceeds available supply,
    # effective pickup time lengthens for everyone (riders wait longer; drivers are tied up
    # longer -> wild-goose chase). This is the reduced form of Castillo's pickup-time
    # distribution G(.) depending on driver availability; it makes low prices costly.
    congestion_wait_coef: float = 0.8   # calibrated: welfare-weighted (Castillo-platform)
                                        # optimal uniform multiplier ~1.2 (Castillo: 1.17),
                                        # with a clean spread of optima across objectives.
    congestion_wait_cap: float = 5.0

    # ----- Supply: heterogeneous drivers -----
    n_drivers: int = 400          # fleet pool size (online subset varies via entry/exit).
                                  # Z23 sweeps 1000-2000 on Manhattan; scaled to 25-zone grid.
    # Driver types: (name, share, reservation_wage_per_epoch, reposition_beta, share_of_pool)
    # reservation_wage = opportunity cost per epoch ONLINE ($). Low => full-time/long-hours.
    # reposition_beta = logit sensitivity to expected earnings when relocating (C25 delta=0.09 is weak).
    # full-time drivers: low reservation wage (drive throughout, incl. low-surge off-peak).
    # part-time drivers: high reservation wage (online only when earnings high => peaks).
    driver_types: tuple = (
        # name,        share, reservation_wage, reposition_beta
        ("fulltime",   0.35,  0.18,             0.9),
        ("parttime",   0.45,  0.55,             0.9),
        ("casual",     0.20,  0.95,             0.6),
    )
    online_window: int = 6        # epochs over which drivers average earnings for entry/exit.
    entry_exit_smooth: float = 0.5  # smoothing for the online/offline hysteresis.
    trip_accept_prob: float = 0.95  # driver accepts an offered dispatch (C25 phi_a~0.69; we use high).

    # ----- Driver belief / repositioning -----
    belief_lr: float = 0.25       # learning rate for drivers' zonal earnings estimates (adaptive eq.).
    reposition_cost_weight: float = 1.0  # weight on travel cost in repositioning logit.

    # ----- Misc -----
    seed: int = 0
    rebalance_cost_per_cell: float = 0.30  # platform pays to incentivize a reposition (= drive cost).
    rebalance_enabled: bool = True

    def n_zones(self) -> int:
        return self.grid_size * self.grid_size

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)


@dataclass
class PPOConfig:
    total_steps: int = 300_000    # environment steps for training.
    rollout_len: int = 48 * 16    # steps per update (~16 episodes).
    epochs: int = 10
    minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0


# Platform objective treatments (the RQ2 core).
PLATFORM_OBJECTIVES = ("profit", "throughput", "welfare", "welfare_weighted")

# Per-objective reward scales, chosen so episode returns are O(50-100) for stable PPO
# (the raw objective magnitudes differ by orders of magnitude across objectives).
REWARD_SCALES = dict(profit=0.005, throughput=0.01, welfare=0.0005, welfare_weighted=0.0005)
