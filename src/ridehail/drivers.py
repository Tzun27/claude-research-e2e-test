"""Heterogeneous, strategically-responding driver population.

Design (grounded in grounding/driver_behavior_notes.md and castillo_model_notes.md):
  * Repositioning: idle drivers move toward higher expected net earnings via a logit
    over reachable neighbor zones (temperature beta_i). This nests the Besbes (2021)
    Wardrop spatial equilibrium (earnings equalized net of travel cost) and Castillo's
    movement logit (eq. 5, delta=0.09 -> weak response).
  * Entry/exit: each driver participates iff the local expected earnings rate clears
    their heterogeneous reservation wage rho_i (Afeche 2023 participation margin;
    Hall-Horton-Knoepfle 2023 re-equilibration). Low rho => full-time/long-hours
    drivers who stay online through low-surge off-peak periods; high rho => casual
    drivers active only at peaks. This timing/exposure structure is the mechanism
    behind Castillo's "long-hours drivers hurt most" finding.
  * Pay homogeneity: drivers are paid by the meter (fare * (1-commission) - drive cost),
    independent of which rider -> no allocative gain to drivers (Castillo eq. 3).
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .geometry import Geometry


class DriverPool:
    def __init__(self, cfg: SimConfig, geo: Geometry, rng: np.random.Generator):
        self.cfg = cfg
        self.geo = geo
        self.rng = rng
        M = cfg.n_drivers
        self.M = M
        # Assign types by share.
        names = [t[0] for t in cfg.driver_types]
        shares = np.array([t[1] for t in cfg.driver_types], dtype=float)
        shares = shares / shares.sum()
        self.type_names = names
        self.type_id = rng.choice(len(names), size=M, p=shares)
        self.rho = np.array([cfg.driver_types[i][2] for i in self.type_id], dtype=float)
        self.beta = np.array([cfg.driver_types[i][3] for i in self.type_id], dtype=float)
        self.reset()

    def reset(self):
        cfg, geo = self.cfg, self.geo
        M, Z = self.M, geo.Z
        rng = self.rng
        # Initial locations: weighted toward periphery (drivers live in periphery).
        dctr = geo.dist[geo.center].astype(float)
        pw = dctr / dctr.sum()
        self.loc = rng.choice(Z, size=M, p=pw)
        self.busy_until = np.zeros(M, dtype=np.int32)   # free when busy_until <= t
        self.dest_when_free = self.loc.copy()
        self.online = np.ones(M, dtype=bool)            # start optimistic; entry/exit adjusts
        self.earn_total = np.zeros(M)
        self.cost_total = np.zeros(M)                   # drive cost (fuel) borne by driver
        self.online_epochs = np.zeros(M, dtype=np.int32)
        self.trips = np.zeros(M, dtype=np.int32)
        # population belief: expected net earnings per online-epoch for a driver in zone z
        self.belief_rate = np.full(Z, 0.5 * float(np.mean(self.rho)))
        self._epoch_zone_earn = np.zeros(Z)             # accumulator within an epoch
        self._epoch_zone_navail = np.zeros(Z)

    # ---- availability ----
    def available_mask(self, t: int) -> np.ndarray:
        return self.online & (self.busy_until <= t)

    def positions_when_free(self, t: int) -> np.ndarray:
        """Where each driver is now: dest_when_free if free, else current loc placeholder."""
        return np.where(self.busy_until <= t, self.loc, self.loc)

    # ---- start-of-epoch: entry/exit + repositioning of idle drivers ----
    def update_online(self, t: int):
        """Entry/exit: P(online) = sigmoid((belief_rate[loc] - rho)/scale). Smoothed."""
        cfg = self.cfg
        # Only drivers that are free can flip online status (busy drivers finish their trip).
        free = self.busy_until <= t
        local_rate = self.belief_rate[self.loc]
        scale = 0.15 + 0.1 * np.abs(self.rho)
        p_on = 1.0 / (1.0 + np.exp(-(local_rate - self.rho) / scale))
        # hysteresis via smoothing toward current state
        s = cfg.entry_exit_smooth
        p_on = s * self.online + (1 - s) * p_on
        draw = self.rng.random(self.M)
        new_online = draw < p_on
        self.online = np.where(free, new_online, self.online)

    def reposition(self, t: int, rebalance_bonus: np.ndarray):
        """Idle, online, free drivers move one cell toward higher expected earnings.

        rebalance_bonus[z] is the platform's repositioning incentive signal (>=0) added
        to the perceived attractiveness of moving toward zone z. Vectorized by grouping
        idle drivers on (zone, type): drivers in the same zone share the earnings
        landscape and drivers of the same type share the logit temperature beta.
        """
        cfg, geo = self.cfg, self.geo
        idle = self.available_mask(t)
        idx = np.where(idle)[0]
        if idx.size == 0:
            return
        zones = self.loc[idx]
        for z in np.unique(zones):
            reach = geo.reach[z]
            rd = geo.reach_dist[z]
            attract = (self.belief_rate[reach]
                       - cfg.reposition_cost_weight * cfg.drive_cost_per_cell * rd
                       + rebalance_bonus[reach])
            in_z = idx[zones == z]
            tz = self.type_id[in_z]
            for k in np.unique(tz):
                beta_k = cfg.driver_types[k][3]
                logits = beta_k * attract
                logits -= logits.max()
                p = np.exp(logits); p /= p.sum()
                members = in_z[tz == k]
                pick = self.rng.choice(len(reach), size=len(members), p=p)
                choices = reach[pick]
                self.loc[members] = choices
                self.cost_total[members] += cfg.drive_cost_per_cell * geo.dist[z, choices]

    # ---- matching credits a trip to a driver ----
    def assign_trip(self, i: int, t: int, pickup_time: int, trip_time: int,
                    dest: int, driver_payment: float, drive_dist: int):
        cfg = self.cfg
        self.busy_until[i] = t + max(1, pickup_time + trip_time)
        self.loc[i] = dest
        self.earn_total[i] += driver_payment
        self.cost_total[i] += cfg.drive_cost_per_cell * drive_dist
        self.trips[i] += 1

    # ---- end-of-epoch: belief update + online time accounting ----
    def begin_epoch_accounting(self, t: int):
        self._z_earn = np.zeros(self.geo.Z)
        self._z_cnt = np.zeros(self.geo.Z)
        # count online time for all online drivers this epoch
        self.online_epochs += self.online.astype(np.int32)

    def record_zone_rate(self, from_zone: int, rate: float):
        self._z_earn[from_zone] += rate
        self._z_cnt[from_zone] += 1

    def record_idle_zone(self, from_zone: int):
        # idle available driver earned 0 this epoch
        self._z_earn[from_zone] += 0.0
        self._z_cnt[from_zone] += 1

    def end_epoch_update_beliefs(self):
        cfg = self.cfg
        mask = self._z_cnt > 0
        realized = np.zeros(self.geo.Z)
        realized[mask] = self._z_earn[mask] / self._z_cnt[mask]
        lr = cfg.belief_lr
        self.belief_rate[mask] = (1 - lr) * self.belief_rate[mask] + lr * realized[mask]
        # gentle decay of belief toward 0 in zones with no observation (forgetting)
        self.belief_rate[~mask] *= (1 - 0.05)

    def type_net_per_capita(self):
        """Running per-capita net surplus by type (for the fairness reward). Returns (n_types,)."""
        n_types = len(self.type_names)
        out = np.zeros(n_types)
        net = self.earn_total - self.cost_total - self.rho * self.online_epochs
        for k in range(n_types):
            m = self.type_id == k
            nk = int(m.sum())
            out[k] = net[m].sum() / max(1, nk)
        return out

    # ---- surplus accounting ----
    def driver_surplus(self):
        """DS_i = earnings_i - drive_cost_i - rho_i * online_epochs_i. Returns total + by type."""
        net = self.earn_total - self.cost_total - self.rho * self.online_epochs
        by_type = {}
        for k, name in enumerate(self.type_names):
            m = self.type_id == k
            by_type[name] = dict(
                surplus=float(net[m].sum()),
                earnings=float(self.earn_total[m].sum()),
                drive_cost=float(self.cost_total[m].sum()),
                opp_cost=float((self.rho * self.online_epochs)[m].sum()),
                online_epochs=int(self.online_epochs[m].sum()),
                trips=int(self.trips[m].sum()),
                n_drivers=int(m.sum()),
            )
        return float(net.sum()), by_type
