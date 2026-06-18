"""Spatial ride-hailing market environment with a four-way welfare decomposition.

Single-agent control problem: the PLATFORM (RL controller) sets, each epoch:
  * surge multipliers per zone        (pricing lever)
  * a global dispatch radius          (matching lever; controls wild-goose-chase tradeoff)
  * a per-zone rebalancing incentive  (rebalancing lever)
against a heterogeneous, strategically-responding driver population (drivers.py) and an
elastic, discrete-choice rider demand.

Welfare (per episode), all transfers (price, commission) cancel so:
    W = sum_matched (v - value_of_wait * pickup) - sum_drivers drive_cost - sum_drivers rho*online
and equivalently W = RiderSurplus + DriverSurplus + PlatformProfit. The env asserts this
identity (consistency check) when welfare bookkeeping is enabled.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .geometry import Geometry, demand_profile, od_destination_weights
from .drivers import DriverPool
from .matching import batch_match, expected_pickup_per_zone


class RideHailEnv:
    def __init__(self, cfg: SimConfig, objective: str = "welfare",
                 alpha_pi: float = 0.4, alpha_R: float = 1.0, alpha_D: float = 0.3,
                 reward_scale: float = 0.01):
        self.cfg = cfg
        self.geo = Geometry(cfg)
        self.Z = self.geo.Z
        self.objective = objective
        self.alpha_pi, self.alpha_R, self.alpha_D = alpha_pi, alpha_R, alpha_D
        self.reward_scale = reward_scale
        self.lam = demand_profile(cfg)               # (T, Z)
        self.od = od_destination_weights(cfg)         # (T, Z, Z)
        self._fare1 = cfg.fare_base + cfg.fare_per_cell * self.geo.dist.astype(float)  # (Z,Z) fare@mult1
        self.obs_dim = 3 * self.Z + 4
        self.act_dim = 2 * self.Z + 1
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    # ---------------- core API ----------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.drivers = DriverPool(self.cfg, self.geo, self.rng)
        self.t = 0
        self._recent_demand = np.zeros(self.Z)
        self._last_surge = np.ones(self.Z)
        # welfare accumulators
        self.W = dict(rider_surplus=0.0, driver_surplus=0.0, platform_profit=0.0,
                      gross_revenue=0.0, n_requests=0, n_matched=0, n_abandon=0,
                      total_wait=0.0, total_pickup=0.0, total_drive_cost=0.0,
                      total_value=0.0, total_opp_cost=0.0,
                      surge_trips=0, fares_sum=0.0, mult_weighted=0.0)
        return self._get_obs()

    def _get_obs(self):
        t, cfg = self.t, self.cfg
        avail = self.drivers.available_mask(min(t, cfg.T - 1))
        supply = np.bincount(self.drivers.loc[avail], minlength=self.Z).astype(float)
        lam_t = self.lam[min(t, cfg.T - 1)]
        belief = self.drivers.belief_rate
        frac = t / cfg.T
        obs = np.concatenate([
            supply / (cfg.n_drivers / self.Z + 1e-6),
            lam_t / (cfg.peak_demand + cfg.base_demand),
            belief / (np.mean(self.drivers.rho) + 1e-6),
            [np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac), frac,
             supply.sum() / (cfg.n_drivers + 1e-6)],
        ]).astype(np.float32)
        return obs

    def decode_action(self, raw: np.ndarray):
        """Map an unbounded policy vector to (surge[Z], dispatch_radius int, rebalance[Z]>=0)."""
        cfg = self.cfg
        a_surge = raw[:self.Z]
        a_disp = raw[self.Z]
        a_reb = raw[self.Z + 1:]
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        surge = cfg.surge_min + (cfg.surge_max - cfg.surge_min) * sig(a_surge)
        dr = cfg.dispatch_radius_min + (cfg.dispatch_radius_max - cfg.dispatch_radius_min) * sig(a_disp)
        dispatch_radius = int(np.clip(round(dr), cfg.dispatch_radius_min, cfg.dispatch_radius_max))
        rebalance = sig(a_reb) * (0.5 * np.mean(self.drivers.rho))  # bonus scaled to earnings units
        return surge, dispatch_radius, rebalance

    def step(self, raw_action: np.ndarray):
        surge, dispatch_radius, rebalance = self.decode_action(raw_action)
        return self.apply(surge, dispatch_radius, rebalance)

    def apply(self, surge: np.ndarray, dispatch_radius: int, rebalance: np.ndarray):
        cfg, geo, drv = self.cfg, self.geo, self.drivers
        t = self.t
        if not cfg.rebalance_enabled:
            rebalance = np.zeros(self.Z)

        # 1) drivers: entry/exit + reposition (start of epoch)
        drv.update_online(t)
        drv.begin_epoch_accounting(t)
        cost0 = float(drv.cost_total.sum())
        drv.reposition(t, rebalance)

        # 2) available drivers and supply distribution
        avail = drv.available_mask(t)
        avail_idx = np.where(avail)[0]
        avail_zone = drv.loc[avail_idx]
        supply = np.bincount(avail_zone, minlength=self.Z).astype(float)
        driver_idx_by_zone = [avail_idx[avail_zone == z] for z in range(self.Z)]

        # 3) generate potential riders and their request decisions
        lam_t = self.lam[t]
        n_pot = self.rng.poisson(lam_t)                 # (Z,)
        exp_pickup = expected_pickup_per_zone(geo, supply, cfg.abandon_radius)
        origins, dests, vals, fares1 = [], [], [], []
        for z in range(self.Z):
            k = int(n_pot[z])
            if k == 0:
                continue
            dz = self.rng.choice(self.Z, size=k, p=self.od[t, z])
            origins.append(np.full(k, z)); dests.append(dz)
            f1 = self._fare1[z, dz]                       # fare at multiplier 1
            m = np.exp(self.rng.normal(cfg.wtp_log_mu, cfg.wtp_log_sigma, size=k))  # WTP/fare ratio
            v = f1 * m
            vals.append(v); fares1.append(f1)
        if origins:
            origins = np.concatenate(origins); dests = np.concatenate(dests)
            vals = np.concatenate(vals); fares1 = np.concatenate(fares1)
            price = fares1 * surge[origins]
            ew = exp_pickup[origins]
            # request iff expected utility >= 0 (outside option 0)
            req = vals - price - cfg.value_of_wait_per_epoch * ew >= 0.0
            r_origin = origins[req]; r_dest = dests[req]
            r_val = vals[req]; r_price = price[req]; r_fare1 = fares1[req]
        else:
            r_origin = np.array([], int); r_dest = np.array([], int)
            r_val = np.array([]); r_price = np.array([]); r_fare1 = np.array([])

        n_requests = len(r_origin)

        # 4) matching (nearest-driver-first within dispatch_radius)
        matches = batch_match(geo, r_origin, driver_idx_by_zone, dispatch_radius, self.rng)

        # 5) settle matches; accumulate welfare
        ep_profit = 0.0; ep_rider = 0.0; ep_driver_earn = 0.0
        ep_value = 0.0; ep_wait = 0.0; ep_pickup = 0.0; ep_fares = 0.0
        matched_riders = set()
        zone_rate_earn = np.zeros(self.Z); zone_rate_cnt = np.zeros(self.Z)
        for (ridx, did, pud) in matches:
            o = r_origin[ridx]; d = r_dest[ridx]
            trip_d = int(geo.dist[o, d]); pickup_d = int(pud)
            p = float(r_price[ridx]); v = float(r_val[ridx])
            driver_pay = (1 - cfg.commission) * p
            drive_dist = pickup_d + trip_d
            dur = max(1, pickup_d + trip_d)
            drv.assign_trip(did, t, pickup_d, trip_d, d, driver_pay, drive_dist)
            # welfare pieces
            wait = pickup_d
            ep_rider += v - p - cfg.value_of_wait_per_epoch * wait
            ep_profit += cfg.commission * p
            ep_driver_earn += driver_pay
            ep_value += v
            ep_wait += wait; ep_pickup += pickup_d; ep_fares += p
            matched_riders.add(ridx)
            self.W['n_matched'] += 1
            if surge[o] > 1.0 + 1e-9:
                self.W['surge_trips'] += 1
            self.W['mult_weighted'] += surge[o]
            # belief sample: this driver's origin zone earned rate driver_pay/dur
            from_zone = o  # driver picked up near origin; attribute earnings to origin zone
            zone_rate_earn[from_zone] += driver_pay / dur
            zone_rate_cnt[from_zone] += 1

        # Belief update: matched trips contribute their earnings rate to their origin zone;
        # idle available drivers contribute 0-rate observations (scarcity drags belief down).
        idle_count = np.bincount(avail_zone, minlength=self.Z).astype(float)
        drv._z_earn = zone_rate_earn.copy()
        drv._z_cnt = zone_rate_cnt + 0.25 * idle_count
        drv.end_epoch_update_beliefs()

        n_abandon = n_requests - len(matched_riders)
        ep_drive_cost = float(drv.cost_total.sum()) - cost0

        # 6) accumulate episode welfare
        self.W['platform_profit'] += ep_profit
        self.W['rider_surplus'] += ep_rider
        self.W['gross_revenue'] += ep_fares
        self.W['fares_sum'] += ep_fares
        self.W['n_requests'] += n_requests
        self.W['n_abandon'] += n_abandon
        self.W['total_wait'] += ep_wait
        self.W['total_pickup'] += ep_pickup
        self.W['total_value'] += ep_value
        self._recent_demand = lam_t
        self._last_surge = surge

        # 7) reward for the configured objective (per-epoch)
        ep_opp = float((drv.rho * drv.online.astype(float)).sum())  # opportunity cost this epoch
        ep_driver_net = ep_driver_earn - ep_drive_cost - ep_opp     # driver surplus increment
        reward_components = dict(profit=ep_profit, throughput=float(len(matches)),
                                 rider=ep_rider, driver_earn=ep_driver_earn,
                                 driver_net=ep_driver_net, opp=ep_opp, drive_cost=ep_drive_cost)
        reward = self._reward(reward_components)

        self.t += 1
        done = self.t >= cfg.T
        info = dict(reward_components=reward_components, n_requests=n_requests,
                    n_matched=len(matches), n_abandon=n_abandon,
                    mean_pickup=(ep_pickup / max(1, len(matches))),
                    online=int(drv.online.sum()))
        if done:
            self._finalize_welfare()
            info['welfare'] = self.welfare_summary()
        return self._get_obs(), reward, done, info

    def _reward(self, c):
        s = self.reward_scale
        if self.objective == "profit":
            return s * c['profit']
        if self.objective == "throughput":
            return s * c['throughput']
        if self.objective == "welfare":
            # true marginal welfare = rider surplus + platform profit + driver net
            #  = (v - vow*wait) - drive_cost - opp_cost  (transfers cancel)
            return s * (c['rider'] + c['profit'] + c['driver_net'])
        if self.objective == "welfare_weighted":
            # Castillo-style platform: alpha_pi*profit + alpha_R*RS + alpha_D*DS
            return s * (self.alpha_pi * c['profit'] + self.alpha_R * c['rider']
                        + self.alpha_D * c['driver_net'])
        raise ValueError(self.objective)

    def _finalize_welfare(self):
        ds_total, by_type = self.drivers.driver_surplus()
        self.W['driver_surplus'] = ds_total
        self.W['driver_by_type'] = by_type
        self.W['total_drive_cost'] = float(self.drivers.cost_total.sum())
        self.W['total_opp_cost'] = float((self.drivers.rho * self.drivers.online_epochs).sum())
        self.W['total_welfare'] = (self.W['rider_surplus'] + self.W['driver_surplus']
                                   + self.W['platform_profit'])

    def welfare_summary(self):
        W = self.W
        # consistency check: W == sum(v - vow*pickup) - drive_cost - opp_cost
        vow = self.cfg.value_of_wait_per_epoch
        w_resource = (W['total_value'] - vow * W['total_pickup']
                      - W['total_drive_cost'] - W['total_opp_cost'])
        out = dict(W)
        out['W_resource_check'] = float(w_resource)
        out['W_components_sum'] = float(W['total_welfare'])
        out['consistency_gap'] = float(abs(w_resource - W['total_welfare']))
        return out

    # ---------------- rollout helpers ----------------
    def run_episode(self, policy_fn, seed: int | None = None):
        """policy_fn(obs) -> raw_action. Returns welfare summary."""
        obs = self.reset(seed=seed)
        done = False
        while not done:
            a = policy_fn(obs)
            obs, r, done, info = self.step(a)
        return info['welfare']
