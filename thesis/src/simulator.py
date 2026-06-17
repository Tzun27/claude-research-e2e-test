"""Zone-based, time-stepped two-sided ride-hailing market simulator.

The simulator is method-agnostic: pricing, matching and rebalancing policies are
injected as objects implementing the interfaces in `methods.py`. It produces a
full Castillo-style welfare decomposition (rider surplus, driver surplus by type,
platform profit) plus operational metrics. See DESIGN.md for the spec.

Driver heterogeneity (the corpus-novel piece) has two grounded mechanisms:
  H1  spatial flexibility / surge-chasing: flexible idle drivers reposition toward
      high-value (surge/demand) zones; constrained drivers stay near a home zone.
  H2  labor-supply elasticity (extensive margin): flexible drivers go on/offline
      with recent earnings (elastic); constrained drivers work ~fixed shifts.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from config import (
    COMMISSION, FARE_BASE, FARE_PER_MIN, CELL_MIN, VOT_RIDER,
    DRIVER_OPP_COST_PER_MIN, DRIVING_COST_PER_CELL, SUPPLY_ELASTICITY,
    SURGE_CAP, SURGE_FLOOR, CHASE_PROB, HOME_RADIUS,
    ScenarioConfig, OFFLINE, IDLE, TO_PICKUP, ON_TRIP, FLEXIBLE, CONSTRAINED,
)

@dataclass(slots=True)
class Request:
    oz: int          # origin zone
    dz: int          # destination zone
    value: float     # rider willingness-to-pay ($)
    price: float     # price offered & accepted ($)
    fare_base: float # unsurged fare ($)
    trip_cells: int  # trip length in cells
    create_t: int    # step created
    matched: bool
    driver: int      # assigned driver idx or -1
    _wait_min: float = 0.0   # pickup wait (min), set at pickup


class Market:
    def __init__(self, scn: ScenarioConfig, pricing, matching, rebalancing,
                 value_table: np.ndarray | None = None,
                 mu_v: float = 0.30, sigma_v: float = 0.55,
                 n_time_buckets: int = 8, learn_value: bool = False,
                 collect_traces: bool = False):
        self.scn = scn
        self.rng = np.random.default_rng(scn.seed)
        self.N = scn.grid_n
        self.Z = self.N * self.N
        self.pricing = pricing
        self.matching = matching
        self.rebalancing = rebalancing
        self.mu_v, self.sigma_v = mu_v, sigma_v
        self.n_tb = n_time_buckets
        self.learn_value = learn_value
        self.collect_traces = collect_traces

        # Value table V[zone, time_bucket] (Xu2018-style). Frozen if provided.
        if value_table is not None:
            self.V = value_table.copy()
        else:
            self.V = np.zeros((self.Z, self.n_tb))
        self.alpha_td = 0.10
        self.gamma = 0.95

        self._build_geometry()
        self._build_demand_field()
        self._init_drivers()

        # accounting
        self.rider_surplus = 0.0
        self.gmv = 0.0
        self.platform_profit = 0.0
        self.n_requests = 0          # potential requests that accepted price
        self.n_completed = 0
        self.n_abandoned = 0
        self.wait_times = []         # minutes, completed trips
        self.surge_now = np.ones(self.Z)
        self.pending: list[Request] = []   # accepted, not-yet-matched requests
        self.active: dict[int, Request] = {}  # matched, in-progress trips
        self._sc = np.zeros(self.Z)        # cached idle supply count for the step
        self._surge_acc = 0.0
        self._surge_steps = 0
        # per-step traces for square-root-law validation
        self.trace_idle_density = []
        self.trace_mean_wait = []
        # per-match traces: (local idle density at rider, pickup distance in cells)
        self.trace_local_density = []
        self.trace_pickup_dist = []

    # ------------------------------------------------------------------ setup
    def _build_geometry(self):
        N = self.N
        coords = np.array([(i // N, i % N) for i in range(self.Z)])
        self.coords = coords
        diff = np.abs(coords[:, None, :] - coords[None, :, :])
        self.dist = diff.sum(-1).astype(np.int16)          # Manhattan, in cells
        self.adj = (self.dist == 1).astype(np.float64)     # 4-neighbour adjacency

    def _build_demand_field(self):
        scn = self.scn
        N = self.N
        center = np.array([(N - 1) / 2, (N - 1) / 2])
        d_center = np.abs(self.coords - center).sum(1)
        d_center = d_center / max(d_center.max(), 1)
        # spatial weight: uniform blended with a CBD hotspot (closer = more demand)
        hotspot = np.exp(-3.0 * d_center)
        uniform = np.ones(self.Z)
        w = (1 - scn.spatial_concentration) * uniform + scn.spatial_concentration * hotspot
        self.zone_weight = w / w.sum()
        # temporal profile: two peaks (AM/PM), peakedness controls trough/peak ratio
        t = np.arange(scn.n_steps)
        peaks = (np.exp(-0.5 * ((t - scn.n_steps * 0.25) / (scn.n_steps * 0.09)) ** 2)
                 + np.exp(-0.5 * ((t - scn.n_steps * 0.72) / (scn.n_steps * 0.09)) ** 2))
        base = 1.0
        peaks = peaks / peaks.max()
        prof = base + (scn.temporal_peakedness - 1.0) * peaks
        self.temporal = prof / prof.mean()                 # mean 1
        # mean potential requests per step = dsr * fleet
        self.mean_potential = scn.demand_supply_ratio * scn.fleet_size

    def _init_drivers(self):
        scn = self.scn
        M = scn.fleet_size
        self.M = M
        n_flex = int(round(scn.flex_frac * M))
        types = np.array([FLEXIBLE] * n_flex + [CONSTRAINED] * (M - n_flex))
        self.rng.shuffle(types)
        self.dtype = types
        # Placement encodes the heterogeneity mechanism (Cook 2021 'location'
        # channel; Castillo limited-mobility delta): flexible drivers start near
        # demand (and can reposition freely); constrained drivers are anchored to
        # home zones drawn UNIFORMLY across the city, so many sit away from surge
        # hotspots and cannot reach them.
        zones = np.empty(M, dtype=np.int32)
        flex_mask = types == FLEXIBLE
        n_f = int(flex_mask.sum())
        zones[flex_mask] = self.rng.choice(self.Z, size=n_f, p=self.zone_weight)
        zones[~flex_mask] = self.rng.integers(0, self.Z, size=M - n_f)
        self.dzone = zones.astype(np.int32)
        self.drow = (self.dzone // self.N).astype(np.int32)
        self.dcol = (self.dzone % self.N).astype(np.int32)
        self.home = self.dzone.copy()
        self.status = np.full(M, IDLE, dtype=np.int8)
        # constrained drivers work fixed shift (online whole episode); flexible
        # start online and re-decide on the extensive margin.
        self.online = np.ones(M, dtype=bool)
        self.target = self.dzone.copy()                    # movement target cell
        self.trip_dz = np.full(M, -1, dtype=np.int32)      # trip dest after pickup
        self.req_idx = np.full(M, -1, dtype=np.int64)      # assigned request key
        self._next_key = 0
        # accounting per driver
        self.earn = np.zeros(M)             # gross fare share before opp cost
        self.online_steps = np.zeros(M)
        self.idle_steps = np.zeros(M)
        self.miles = np.zeros(M)            # cells driven
        self.trips = np.zeros(M, dtype=np.int32)
        self.recent_earn = np.full(M, DRIVER_OPP_COST_PER_MIN * CELL_MIN)  # $/step seed

    # ------------------------------------------------------------- helpers
    def tbucket(self, t):
        return min(self.n_tb - 1, int(t / self.scn.n_steps * self.n_tb))

    def base_fare(self, trip_cells):
        return FARE_BASE + FARE_PER_MIN * trip_cells * CELL_MIN

    def idle_indices(self):
        return np.where((self.status == IDLE) & self.online)[0]

    def supply_count(self):
        """idle drivers per zone."""
        c = np.zeros(self.Z)
        idle = self.idle_indices()
        if len(idle):
            np.add.at(c, self.dzone[idle], 1.0)
        return c

    def demand_forecast(self):
        """expected potential requests per zone at current step (smoothed)."""
        return self.mean_potential * self.zone_weight * self.temporal[self._t]

    # -------------------------------------------------------------- dynamics
    def _generate_demand(self, t):
        """Vectorised demand generation for the step (riders accept/reject)."""
        rng = self.rng
        sc = self._sc
        # per-zone expected wait via square-root law (vectorised neighbour sum)
        local = sc + 0.5 * (self.adj @ sc)
        ew_zone = CELL_MIN * (0.5 + 1.6 / np.sqrt(local + 0.5))
        zone_rate = self.mean_potential * self.temporal[t] * self.zone_weight
        counts = rng.poisson(zone_rate)
        Ntot = int(counts.sum())
        if Ntot == 0:
            return
        origins = np.repeat(np.arange(self.Z), counts)
        # trip lengths ~ geometric, capped
        L = rng.geometric(1.0 / max(self.scn.trip_length_mean, 1.05), size=Ntot)
        L = np.minimum(L, 2 * (self.N - 1))
        a = (rng.random(Ntot) * (L + 1)).astype(int)
        b = L - a
        r0, c0 = origins // self.N, origins % self.N
        sr = rng.choice((-1, 1), size=Ntot)
        scn_ = rng.choice((-1, 1), size=Ntot)
        r2 = np.clip(r0 + sr * a, 0, self.N - 1)
        c2 = np.clip(c0 + scn_ * b, 0, self.N - 1)
        dz = r2 * self.N + c2
        tc = np.maximum(self.dist[origins, dz].astype(int), 1)
        fb = FARE_BASE + FARE_PER_MIN * tc * CELL_MIN
        price = fb * self.surge_now[origins]
        val = fb * np.exp(self.mu_v + self.sigma_v * rng.standard_normal(Ntot))
        accept = val >= price + VOT_RIDER * ew_zone[origins]
        idx = np.nonzero(accept)[0]
        self.n_requests += len(idx)
        for j in idx:
            self.pending.append(Request(int(origins[j]), int(dz[j]), float(val[j]),
                                        float(price[j]), float(fb[j]), int(tc[j]),
                                        t, False, -1))

    def _neighbors(self, z):
        r, c = z // self.N, z % self.N
        nb = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.N and 0 <= cc < self.N:
                nb.append(rr * self.N + cc)
        return nb

    def _move_one(self, i):
        """advance driver i one cell toward its target; return True if arrived."""
        tr, tc = self.target[i] // self.N, self.target[i] % self.N
        if self.drow[i] == tr and self.dcol[i] == tc:
            return True
        if abs(int(tr) - int(self.drow[i])) >= abs(int(tc) - int(self.dcol[i])):
            self.drow[i] += 1 if tr > self.drow[i] else -1
        else:
            self.dcol[i] += 1 if tc > self.dcol[i] else -1
        self.dzone[i] = self.drow[i] * self.N + self.dcol[i]
        self.miles[i] += 1
        return self.drow[i] == tr and self.dcol[i] == tc

    def _advance_movers(self, t):
        for i in np.where(np.isin(self.status, (TO_PICKUP, ON_TRIP)))[0]:
            arrived = self._move_one(i)
            if not arrived:
                continue
            if self.status[i] == TO_PICKUP:
                # picked up rider -> start trip
                r = self.active.get(int(self.req_idx[i]))
                if r is None:           # request vanished (shouldn't happen)
                    self.status[i] = IDLE
                    self.req_idx[i] = -1
                    continue
                self.status[i] = ON_TRIP
                self.target[i] = r.dz
                self.trip_dz[i] = r.dz
                # record wait (create -> pickup)
                wait_min = (t - r.create_t) * CELL_MIN
                self.wait_times.append(wait_min)
                r._wait_min = wait_min
            else:  # ON_TRIP arrived -> dropoff, complete
                self._complete_trip(i, t)

    def _complete_trip(self, i, t):
        r = self.active.pop(int(self.req_idx[i]), None)
        self.status[i] = IDLE
        self.req_idx[i] = -1
        self.trip_dz[i] = -1
        if r is None:
            return
        price = r.price
        driver_share = (1 - COMMISSION) * price
        self.earn[i] += driver_share
        self.platform_profit += COMMISSION * price
        self.gmv += price
        self.trips[i] += 1
        wait_min = getattr(r, "_wait_min", (t - r.create_t) * CELL_MIN)
        # rider surplus realized: value - price - VOT*wait
        self.rider_surplus += r.value - price - VOT_RIDER * wait_min
        self.n_completed += 1
        # TD update of value table on the served transition
        if self.learn_value:
            b0 = self.tbucket(r.create_t)
            b1 = self.tbucket(t)
            rew = driver_share - DRIVING_COST_PER_CELL * (self.dist[r.oz, r.dz])
            target = rew + self.gamma * self.V[r.dz, b1]
            self.V[r.oz, b0] += self.alpha_td * (target - self.V[r.oz, b0])

    # ------------------------------------------------------- rebalancing/H1
    def _rebalance(self, t):
        idle = self.idle_indices()
        if len(idle) == 0:
            return
        self._sc = self.supply_count()                # refresh after matching/movement
        g = self.rebalancing.guidance(self, t)        # per-zone guidance score
        surge = self.surge_now
        # normalize fields
        if g.max() > g.min():
            g = (g - g.min()) / (g.max() - g.min())
        s = (surge - 1.0)
        s = s / (s.max() + 1e-9) if s.max() > 0 else s
        for i in idle:
            self.online_steps[i] += 1
            self.idle_steps[i] += 1
            typ = self.dtype[i]
            if typ == FLEXIBLE:
                w_self, chase = 1.0, CHASE_PROB["flexible"]
                radius = HOME_RADIUS["flexible"]
            else:
                w_self, chase = 0.0, CHASE_PROB["constrained"]
                radius = HOME_RADIUS["constrained"]
            if self.rng.random() > chase:
                continue                                  # stays put this step
            z = self.dzone[i]
            # desirability over zones within reach & home radius
            reach = self.dist[z] <= 3
            home_ok = self.dist[self.home[i]] <= radius
            cand = np.where(reach & home_ok)[0]
            if len(cand) == 0:
                continue
            desir = (g[cand]
                     + w_self * s[cand]
                     - 0.05 * self.dist[z, cand])
            best = int(cand[int(np.argmax(desir))])
            if best != z:
                self._reposition_step(i, best)   # idle driver drifts one cell

    def _reposition_step(self, i, dest):
        # treat as a transient move (status stays IDLE but position changes)
        self.target[i] = dest
        tr, tc = dest // self.N, dest % self.N
        if abs(int(tr) - int(self.drow[i])) >= abs(int(tc) - int(self.dcol[i])):
            self.drow[i] += 1 if tr > self.drow[i] else (-1 if tr < self.drow[i] else 0)
        else:
            self.dcol[i] += 1 if tc > self.dcol[i] else (-1 if tc < self.dcol[i] else 0)
        self.dzone[i] = self.drow[i] * self.N + self.dcol[i]
        self.miles[i] += 1

    # ------------------------------------------------------------- labor H2
    def _labor_update(self, t):
        """Extensive-margin: flexible drivers enter/exit with recent earnings.

        Only drivers that are currently IDLE or OFFLINE re-decide (mid-trip
        drivers finish their trip first), so online time is counted cleanly.
        """
        if t % 16 != 0 or t == 0:
            return
        reservation = DRIVER_OPP_COST_PER_MIN * CELL_MIN     # $/step
        elas = SUPPLY_ELASTICITY["flexible"]
        for i in range(self.M):
            if self.dtype[i] == CONSTRAINED:
                continue                                     # fixed shift, stays online
            if self.status[i] in (TO_PICKUP, ON_TRIP):
                continue                                     # decide after trip
            ratio = self.recent_earn[i] / max(reservation, 1e-6)
            # logistic participation increasing in earnings ratio, scaled by elas
            p_on = 1.0 / (1.0 + np.exp(-elas * (ratio - 1.0) * 2.0))
            p_on = float(np.clip(p_on, 0.04, 0.98))
            self.online[i] = self.rng.random() < p_on
            if self.online[i]:
                if self.status[i] == OFFLINE:
                    self.status[i] = IDLE
                    self.dzone[i] = self.home[i]
                    self.drow[i] = self.home[i] // self.N
                    self.dcol[i] = self.home[i] % self.N
            else:
                self.status[i] = OFFLINE

    # --------------------------------------------------------------- step
    def step(self, t):
        self._t = t
        self._sc = self.supply_count()    # cache idle supply for pricing/demand/wait
        # 1. pricing (per-zone surge multiplier for this step)
        self.surge_now = np.clip(self.pricing.surge(self, t), SURGE_FLOOR, SURGE_CAP)
        self._surge_acc += float(self.surge_now.mean())
        self._surge_steps += 1
        # 2. demand generation (riders see price+wait, accept/reject)
        self._generate_demand(t)
        # 3. matching (assign idle drivers to pending requests)
        self._do_matching(t)
        # 4. advance movers (pickups, trips, completions)
        self._advance_movers(t)
        # 5. abandonment (waited beyond patience, unmatched)
        self._abandon(t)
        # 6. rebalancing + H1 surge-chasing for still-idle drivers
        self._rebalance(t)
        # 7. labor extensive margin (H2)
        self._labor_update(t)
        # 8. bookkeeping: opp cost accrues to online drivers; recent earn EWMA
        self._accrue(t)
        if self.collect_traces:
            sc = self.supply_count()
            self.trace_idle_density.append(sc.mean())
            if self.wait_times:
                self.trace_mean_wait.append(np.mean(self.wait_times[-50:]))

    def _do_matching(self, t):
        idle = self.idle_indices()
        if len(idle) == 0 or len(self.pending) == 0:
            return
        assigns = self.matching.match(self, list(self.pending), idle)
        used = set()
        for r, i in assigns:
            if r.matched or i in used:
                continue
            r.matched = True
            r.driver = int(i)
            key = self._next_key
            self._next_key += 1
            self.active[key] = r
            self.req_idx[i] = key
            self.status[i] = TO_PICKUP
            self.target[i] = r.oz
            used.add(i)
            if self.collect_traces:
                local = self._sc[r.oz] + sum(self._sc[nz] for nz in self._neighbors(r.oz))
                self.trace_local_density.append(local)
                self.trace_pickup_dist.append(int(self.dist[self.dzone[i], r.oz]))
        # requests that got matched leave the pending pool
        self.pending = [r for r in self.pending if not r.matched]

    def _abandon(self, t):
        keep = []
        for r in self.pending:                 # pending holds only unmatched
            if t - r.create_t >= self.scn.patience:
                self.n_abandoned += 1
            else:
                keep.append(r)
        self.pending = keep

    def _accrue(self, t):
        # opportunity cost accrues for all online drivers each step;
        # online_steps for movers (to-pickup/on-trip) counted here, idle counted in rebalance
        movers = np.where(np.isin(self.status, (TO_PICKUP, ON_TRIP)) & self.online)[0]
        self.online_steps[movers] += 1
        # EWMA recent earnings per online driver (for H2)
        on = self.online
        per_step = self.earn / np.maximum(self.online_steps, 1)
        self.recent_earn = 0.85 * self.recent_earn + 0.15 * per_step

    # --------------------------------------------------------------- run
    def run(self):
        for t in range(self.scn.n_steps):
            self.step(t)
        return self.metrics()

    # ------------------------------------------------------------ welfare
    def metrics(self):
        opp = DRIVER_OPP_COST_PER_MIN * CELL_MIN
        driver_net = self.earn - opp * self.online_steps - DRIVING_COST_PER_CELL * self.miles
        flex = self.dtype == FLEXIBLE
        cons = self.dtype == CONSTRAINED
        ds_flex = driver_net[flex].sum()
        ds_cons = driver_net[cons].sum()
        driver_surplus = driver_net.sum()
        total_welfare = self.rider_surplus + driver_surplus + self.platform_profit
        n_pot = self.n_requests + self.n_abandoned * 0  # accepted-price requests
        # service rate = completed / accepted-price requests
        denom = max(self.n_requests, 1)
        # earnings/hour by type (online time)
        def earn_rate(mask):
            steps = self.online_steps[mask].sum()
            return (self.earn[mask].sum() / steps / CELL_MIN * 60) if steps > 0 else 0.0
        # fairness: Gini over per-driver net surplus among drivers with online time
        active = self.online_steps > 0
        gini = _gini(driver_net[active]) if active.sum() > 1 else 0.0
        return {
            "rider_surplus": self.rider_surplus,
            "driver_surplus": driver_surplus,
            "ds_flexible": ds_flex,
            "ds_constrained": ds_cons,
            "platform_profit": self.platform_profit,
            "total_welfare": total_welfare,
            "gmv": self.gmv,
            "n_requests": self.n_requests,
            "n_completed": self.n_completed,
            "n_abandoned": self.n_abandoned,
            "service_rate": self.n_completed / denom,
            "mean_wait_min": float(np.mean(self.wait_times)) if self.wait_times else 0.0,
            "earn_rate_flex": earn_rate(flex),
            "earn_rate_cons": earn_rate(cons),
            "driver_gini": gini,
            "mean_surge": self._surge_acc / max(self._surge_steps, 1),
            "ds_constrained_per_capita": ds_cons / max(int(cons.sum()), 1),
            "ds_flexible_per_capita": ds_flex / max(int(flex.sum()), 1),
            "n_flexible": int(flex.sum()),
            "n_constrained": int(cons.sum()),
        }


def _gini(x):
    x = np.sort(np.asarray(x, float) - min(0.0, np.min(x)))  # shift to non-negative
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)
