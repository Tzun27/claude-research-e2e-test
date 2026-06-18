"""Grid geometry, travel times, and the spatio-temporal demand profile.

Pure, deterministic helpers (given an RNG) so they can be unit-tested in isolation.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig


class Geometry:
    """G x G grid with Manhattan (L1) travel distance in cells == travel time in epochs.

    Zone index z = r * G + c for cell (r, c).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.G = cfg.grid_size
        self.Z = cfg.n_zones()
        coords = np.array([(z // self.G, z % self.G) for z in range(self.Z)])
        self.coords = coords  # (Z, 2) row,col
        # Pairwise L1 distance (cells) == travel time (epochs), at move_speed 1 cell/epoch.
        diff = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
        self.dist = diff.astype(np.int32)  # (Z, Z)
        self.center = (self.G // 2) * self.G + (self.G // 2)  # CBD zone index

    def travel_time(self, z_from: int, z_to: int) -> int:
        # ceil(distance / speed); speed is cells/epoch.
        d = int(self.dist[z_from, z_to])
        s = self.cfg.move_speed_cells
        return (d + s - 1) // s if d > 0 else 0

    def neighbors_within(self, z: int, radius: int) -> np.ndarray:
        return np.where(self.dist[z] <= radius)[0]


def demand_profile(cfg: SimConfig) -> np.ndarray:
    """Return potential-request Poisson rates lambda[t, z_origin] over the episode.

    Bimodal in time (AM/PM peaks). Spatially, off-peak demand is roughly uniform;
    peak demand is directed to create center<->periphery imbalance (commute pattern):
      - AM peak: periphery -> center  (origins concentrated in periphery)
      - PM peak: center -> periphery  (origins concentrated in center)
    This produces the spatial+temporal supply-demand imbalance that surge addresses.
    """
    T, Z, G = cfg.T, cfg.n_zones(), cfg.grid_size
    geo = Geometry(cfg)
    center = geo.center
    # distance of each zone from center, normalized to [0,1]
    dctr = geo.dist[center].astype(float)
    dctr = dctr / dctr.max()
    periphery_w = dctr / dctr.sum()           # weight mass on periphery
    center_w = (1.0 - dctr); center_w /= center_w.sum()  # weight mass near center

    lam = np.full((T, Z), cfg.base_demand, dtype=float)
    ts = np.arange(T) / T
    for k, c in enumerate(cfg.peak_centers):
        g = np.exp(-0.5 * ((ts - c) / cfg.peak_width) ** 2)  # (T,)
        # AM peak (first) origins in periphery; PM peak (second) origins in center.
        spatial = periphery_w if k == 0 else center_w
        uniform = np.full(Z, 1.0 / Z)
        mix = cfg.cbd_pull * spatial + (1 - cfg.cbd_pull) * uniform
        lam += cfg.peak_demand * np.outer(g, mix * Z)  # *Z so mix averages to 1 per zone
    return lam  # (T, Z)


def od_destination_weights(cfg: SimConfig) -> np.ndarray:
    """Return, for each (t, origin), a destination distribution over zones.

    During AM peak trips go toward center; PM peak away from center; off-peak ~ gravity
    (nearby + center bias). Returns w[t, o, d] normalized over d.
    """
    T, Z = cfg.T, cfg.n_zones()
    geo = Geometry(cfg)
    center = geo.center
    dctr = geo.dist[center].astype(float)
    to_center = np.exp(-(dctr) / max(1.0, geo.G))      # high near center
    from_center = 1.0 / (to_center + 1e-6)
    from_center /= from_center.sum()
    to_center_n = to_center / to_center.sum()
    # gravity baseline: prefer nearby destinations (not self)
    gravity = np.exp(-geo.dist / max(1.0, geo.G / 1.5)).astype(float)
    np.fill_diagonal(gravity, 0.2)  # some intra-zone trips
    gravity = gravity / gravity.sum(1, keepdims=True)

    ts = np.arange(T) / T
    w = np.zeros((T, Z, Z))
    for t in range(T):
        am = np.exp(-0.5 * ((ts[t] - cfg.peak_centers[0]) / cfg.peak_width) ** 2)
        pm = np.exp(-0.5 * ((ts[t] - cfg.peak_centers[1]) / cfg.peak_width) ** 2)
        for o in range(Z):
            base = gravity[o].copy()
            mix = base + am * cfg.cbd_pull * to_center_n + pm * cfg.cbd_pull * from_center
            w[t, o] = mix / mix.sum()
    return w
