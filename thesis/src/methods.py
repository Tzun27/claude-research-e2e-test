"""The method library: pluggable pricing, matching and rebalancing policies.

Three families x three methods = 27 controller combinations (the "menu" over
which the brute-force oracle is computed). Each method is a faithful but
deliberately simplified member of a recognizable family from the corpus:

  Matching     M1 greedy-nearest | M2 optimal-bipartite (Hungarian, Xu2018/Enders)
               | M3 value-based (TD value table -> bipartite weights, Xu2018)
  Pricing      P1 flat | P2 reactive multiplicative surge (Besbes "local myopic")
               | P3 fluid market-clearing, elasticity-aware (OR/econ)
  Rebalancing  R1 none | R2 greedy-to-demand | R3 value-gradient (Gammelli bi-level)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import COMMISSION, DRIVING_COST_PER_CELL, SURGE_CAP, SURGE_FLOOR

BIG = 1e6

# ----------------------------------------------------------------- matching
def _req_arrays(mkt, requests):
    oz = np.fromiter((r.oz for r in requests), dtype=np.int64, count=len(requests))
    dz = np.fromiter((r.dz for r in requests), dtype=np.int64, count=len(requests))
    price = np.fromiter((r.price for r in requests), dtype=np.float64, count=len(requests))
    return oz, dz, price


class GreedyMatching:
    code, name = "M1", "greedy-nearest"
    def match(self, mkt, requests, idle):
        zones = mkt.dzone
        R = mkt.scn.dispatch_radius
        free = list(int(i) for i in idle)
        assigns = []
        for r in sorted(requests, key=lambda r: r.create_t):
            if not free:
                break
            fz = zones[free]
            d = mkt.dist[fz, r.oz]
            j = int(np.argmin(d))
            if d[j] <= R:
                assigns.append((r, free[j]))
                free.pop(j)
        return assigns


class OptimalBipartite:
    code, name = "M2", "optimal-bipartite"
    def match(self, mkt, requests, idle):
        idle = np.asarray(idle, dtype=np.int64)
        if idle.size == 0 or not requests:
            return []
        oz, _, _ = _req_arrays(mkt, requests)
        C = mkt.dist[np.ix_(mkt.dzone[idle], oz)].astype(np.float64)  # pickup dist
        C[C > mkt.scn.dispatch_radius] = BIG
        rows, cols = linear_sum_assignment(C)
        return [(requests[b], int(idle[a])) for a, b in zip(rows, cols) if C[a, b] < BIG]


class ValueMatching:
    code, name = "M3", "value-based"
    def match(self, mkt, requests, idle):
        idle = np.asarray(idle, dtype=np.int64)
        if idle.size == 0 or not requests:
            return []
        oz, dz, price = _req_arrays(mkt, requests)
        D = mkt.dist[np.ix_(mkt.dzone[idle], oz)].astype(np.float64)  # pickup dist
        b0 = mkt.tbucket(mkt._t)
        b1 = min(mkt.n_tb - 1, b0 + 1)
        driver_share = (1 - COMMISSION) * price                       # per request
        # anticipatory value (Xu2018): immediate net + discounted dest value - origin value
        future = mkt.gamma * mkt.V[dz, b1] - mkt.V[oz, b0]            # per request
        val = (driver_share + future)[None, :] - DRIVING_COST_PER_CELL * D
        C = -val
        C[D > mkt.scn.dispatch_radius] = BIG
        rows, cols = linear_sum_assignment(C)
        return [(requests[b], int(idle[a])) for a, b in zip(rows, cols) if C[a, b] < BIG]


# ------------------------------------------------------------------ pricing
class FlatPricing:
    code, name = "P1", "flat"
    def surge(self, mkt, t):
        return np.ones(mkt.Z)


class ReactiveSurge:
    """Local myopic multiplicative surge (Besbes 2021's 'local' benchmark).

    Raises fares where demand exceeds idle supply and LOWERS them (below 1) where
    supply is slack -- a spatial reallocation, not a pure markup.
    """
    code, name = "P2", "reactive-surge"
    def __init__(self, k=0.6):
        self.k = k
    def surge(self, mkt, t):
        D = mkt.demand_forecast()
        S = mkt._sc
        m = 1.0 + self.k * (D - S) / (D + S + 1.0)
        return np.clip(m, SURGE_FLOOR, SURGE_CAP)


class FluidPricing:
    """Elasticity-aware market-clearing price (OR/econ family).

    Sets the local multiplier so expected demand at that price approaches the
    local idle-supply throughput, using the calibrated demand elasticity:
    D(m) = D0 * m**eps  ->  set D(m) = C  ->  m = (C/D0)**(1/eps).
    """
    code, name = "P3", "fluid-clearing"
    def __init__(self, theta=0.5):
        self.theta = theta          # partial adjustment toward full market clearing
    def surge(self, mkt, t):
        D = mkt.demand_forecast()
        S = mkt._sc
        eps = mkt.scn.demand_elasticity
        D0 = np.maximum(D, 1e-6)
        C = np.maximum(S, 1e-6)
        ratio = C / D0                               # scarce: <1 -> m>1; slack: >1 -> m<1
        m = ratio ** (self.theta / eps)              # eps<0; theta damps to realism
        return np.clip(m, SURGE_FLOOR, SURGE_CAP)


# -------------------------------------------------------------- rebalancing
class NoRebalancing:
    code, name = "R1", "none"
    def guidance(self, mkt, t):
        return np.zeros(mkt.Z)


class GreedyToDemand:
    code, name = "R2", "greedy-to-demand"
    def guidance(self, mkt, t):
        D = mkt.demand_forecast()
        S = mkt._sc
        return np.maximum(D - S, 0.0)


class ValueGradient:
    """Anticipatory rebalancing toward high future-value zones (Gammelli-style)."""
    code, name = "R3", "value-gradient"
    def guidance(self, mkt, t):
        b = mkt.tbucket(t)
        v = mkt.V[:, b].astype(float)
        D = mkt.demand_forecast()
        S = mkt._sc
        gap = np.maximum(D - S, 0.0)
        def norm(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        return norm(v) + 0.5 * norm(gap)


# ------------------------------------------------------------------ registry
MATCHING = {m.code: m for m in [GreedyMatching, OptimalBipartite, ValueMatching]}
PRICING = {p.code: p for p in [FlatPricing, ReactiveSurge, FluidPricing]}
REBAL = {r.code: r for r in [NoRebalancing, GreedyToDemand, ValueGradient]}

MATCH_CODES = ["M1", "M2", "M3"]
PRICE_CODES = ["P1", "P2", "P3"]
REBAL_CODES = ["R1", "R2", "R3"]


def make_methods(combo):
    """combo like ('M2','P2','R2') -> instantiated (matching, pricing, rebal)."""
    mc, pc, rc = combo
    return MATCHING[mc](), PRICING[pc](), REBAL[rc]()


def all_combos():
    return [(m, p, r) for m in MATCH_CODES for p in PRICE_CODES for r in REBAL_CODES]


NEEDS_VALUE = {"M3", "R3"}
def combo_needs_value(combo):
    return any(c in NEEDS_VALUE for c in combo)
