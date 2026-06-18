"""Batch matching: nearest-driver-first assignment at zone granularity.

Faithful to Castillo (2025) matching technology: riders matched in random order to the
nearest available driver; pickup time = travel time from driver to rider; beyond the
dispatch radius the rider abandons. Endogenous pickup time + drivers tied up during
pickup/trip produce the "wild-goose-chase" feedback (scarcity -> long pickups -> drivers
busy -> more scarcity) that drives the welfare results.

Matching is value-blind on the driver side (drivers paid by meter); surge's allocative
efficiency arises on the demand side (high price screens out low-WTP riders).
"""
from __future__ import annotations

import numpy as np


def expected_pickup_per_zone(geo, supply_by_zone: np.ndarray, abandon_radius: int) -> np.ndarray:
    """ETA proxy shown to riders: distance to nearest zone with an available driver.

    Returns exp_pickup[z] (epochs); large value (abandon_radius+1) if none within radius.
    """
    Z = geo.Z
    has = supply_by_zone > 0
    exp = np.full(Z, abandon_radius + 1, dtype=float)
    if not has.any():
        return exp
    # for each zone, min distance to a zone that has supply
    d = geo.dist[:, has]  # (Z, n_has)
    nearest = d.min(axis=1)
    exp = np.minimum(exp, nearest.astype(float))
    return exp


def batch_match(geo, rider_origin: np.ndarray, driver_idx_by_zone, dispatch_radius: int,
                rng: np.random.Generator):
    """Greedy nearest-driver-first matching.

    Args:
      rider_origin: (R,) origin zone of each requesting rider.
      driver_idx_by_zone: list length Z; driver_idx_by_zone[z] = np.array of available driver ids in z.
      dispatch_radius: max pickup distance (cells) the platform allows this epoch.
    Returns:
      matches: list of (rider_index, driver_id, pickup_dist).
    """
    Z = geo.Z
    # riders grouped by origin zone, shuffled order within zone
    riders_by_zone = [[] for _ in range(Z)]
    for ridx, o in enumerate(rider_origin):
        riders_by_zone[o].append(ridx)
    for z in range(Z):
        rng.shuffle(riders_by_zone[z])
    # mutable driver queues
    drv = [list(driver_idx_by_zone[z]) for z in range(Z)]
    for z in range(Z):
        rng.shuffle(drv[z])

    matches = []
    # precompute, for each zone, the list of zones sorted by distance (within dispatch_radius)
    for d in range(0, dispatch_radius + 1):
        for zr in range(Z):
            if not riders_by_zone[zr]:
                continue
            # driver zones at exactly distance d from zr
            zds = np.where(geo.dist[zr] == d)[0]
            for zd in zds:
                if not drv[zd] or not riders_by_zone[zr]:
                    continue
                k = min(len(riders_by_zone[zr]), len(drv[zd]))
                for _ in range(k):
                    ridx = riders_by_zone[zr].pop()
                    did = drv[zd].pop()
                    matches.append((ridx, did, d))
    return matches
