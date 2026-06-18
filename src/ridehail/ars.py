"""Augmented Random Search (ARS-v2) for the ride-hailing platform controller.

ARS (Mania, Guy & Recht, NeurIPS 2018) optimizes the *episode* objective directly via
finite-difference gradient estimates over random parameter perturbations, with top-direction
selection. It is robust in the weak-per-step-signal regime where the exogenous demand cycle
dominates the per-epoch reward and a per-step value baseline absorbs the action's effect
(see thesis Sec 4.5, where PPO failed for this reason).

Controller: a COMPACT, zone-shared (location-invariant) policy. The surge and rebalancing
rules are the same function applied to each zone's local features (supply, demand, earnings
belief) plus global time features; the dispatch radius is a function of global features.
This ~23-parameter policy lets ARS find the price *level* (a single bias) efficiently and
encodes the sensible inductive bias that the pricing rule depends on local imbalance.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool

from .config import SimConfig
from .env import RideHailEnv
from .evaluate import objective_value, fair_dispersion


@dataclass
class ARSConfig:
    n_iters: int = 90
    n_dirs: int = 16
    top_dirs: int = 8
    step_size: float = 0.08
    noise_std: float = 0.08
    eval_seeds: int = 3
    seed: int = 0
    hidden: int = 0   # unused (kept for API compatibility); policy is zone-shared


class ZoneSharedPolicy:
    """Location-invariant policy over the env observation [supply(Z), demand(Z), belief(Z),
    sin_t, cos_t, frac_t, total_supply_frac]. Outputs the raw action vector the env decodes:
    [surge_logits(Z), dispatch_logit(1), rebalance_logits(Z)]."""
    N_LOCAL = 3      # supply, demand, belief
    N_GLOBAL = 4     # sin_t, cos_t, frac_t, total_supply_frac
    N_FEAT = N_LOCAL + N_GLOBAL          # per-zone feature vector length = 7
    N_DISP_FEAT = 2 + N_GLOBAL           # mean_supply, mean_demand + globals = 6

    def __init__(self, Z, params=None):
        self.Z = Z
        self.params = params if params is not None else np.zeros(self.n_params())

    def n_params(self):
        return (self.N_FEAT + 1) * 2 + (self.N_DISP_FEAT + 1)   # surge, rebal, dispatch

    def _split(self, p):
        nf = self.N_FEAT
        ws = p[:nf]; bs = p[nf]
        wr = p[nf + 1:2 * nf + 1]; br = p[2 * nf + 1]
        wd = p[2 * nf + 2:2 * nf + 2 + self.N_DISP_FEAT]; bd = p[2 * nf + 2 + self.N_DISP_FEAT]
        return ws, bs, wr, br, wd, bd

    def act(self, obs, params=None):
        p = self.params if params is None else params
        Z = self.Z
        supply = obs[:Z]; demand = obs[Z:2 * Z]; belief = obs[2 * Z:3 * Z]
        glob = obs[3 * Z:3 * Z + self.N_GLOBAL]                 # (4,)
        ws, bs, wr, br, wd, bd = self._split(p)
        # per-zone features: (Z, 7)
        local = np.stack([supply, demand, belief], axis=1)     # (Z,3)
        gl = np.broadcast_to(glob, (Z, self.N_GLOBAL))         # (Z,4)
        feat = np.concatenate([local, gl], axis=1)             # (Z,7)
        surge_logits = feat @ ws + bs                          # (Z,)
        rebal_logits = feat @ wr + br                          # (Z,)
        disp_feat = np.concatenate([[supply.mean(), demand.mean()], glob])  # (6,)
        disp_logit = disp_feat @ wd + bd
        return np.concatenate([surge_logits, [disp_logit], rebal_logits])


def _rollout(params, cfg_dict, objective, env_kwargs, Z, weights, seeds):
    cfg = SimConfig(**cfg_dict)
    api, ar, ad = weights
    pol = ZoneSharedPolicy(Z, params=params)
    objs = []
    for sd in seeds:
        env = RideHailEnv(cfg, objective=objective, alpha_pi=api, alpha_R=ar, alpha_D=ad, **env_kwargs)
        obs = env.reset(seed=sd); done = False
        while not done:
            obs, r, done, info = env.step(pol.act(obs))
        w = info["welfare"]
        obj = objective_value(w, objective, api, ar, ad) - env.fair_weight * fair_dispersion(w)
        objs.append(obj)
    return float(np.mean(objs))


def _worker(args):
    (theta, delta, nu, cfg_dict, objective, env_kwargs, Z, weights, seeds) = args
    rp = _rollout(theta + nu * delta, cfg_dict, objective, env_kwargs, Z, weights, seeds)
    rm = _rollout(theta - nu * delta, cfg_dict, objective, env_kwargs, Z, weights, seeds)
    return rp, rm


def train_ars(cfg: SimConfig, objective, ars: ARSConfig, env_kwargs=None, hidden=0,
              weights=(0.4, 1.0, 0.3), n_workers=4, verbose=False, eval_fn=None):
    env_kwargs = env_kwargs or {}
    rng = np.random.default_rng(ars.seed)
    Z = cfg.n_zones()
    pol = ZoneSharedPolicy(Z)
    theta = np.zeros(pol.n_params())
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
    log = []
    pool = Pool(n_workers) if n_workers > 1 else None
    base_seed = 1000
    best_theta = theta.copy(); best_obj = -np.inf
    for it in range(ars.n_iters):
        deltas = [rng.standard_normal(theta.size) for _ in range(ars.n_dirs)]
        seeds = [int(base_seed + it * 7 + s) for s in range(ars.eval_seeds)]
        args = [(theta, d, ars.noise_std, cfg_dict, objective, env_kwargs, Z, weights, seeds) for d in deltas]
        results = pool.map(_worker, args) if pool else [_worker(a) for a in args]
        rps = np.array([r[0] for r in results]); rms = np.array([r[1] for r in results])
        maxr = np.maximum(rps, rms)
        idx = np.argsort(-maxr)[:ars.top_dirs]
        sigma_r = np.concatenate([rps[idx], rms[idx]]).std() + 1e-6
        grad = sum((rps[i] - rms[i]) * deltas[i] for i in idx)
        theta = theta + (ars.step_size / (ars.top_dirs * sigma_r)) * grad
        cur = (rps.mean() + rms.mean()) / 2
        if cur > best_obj:
            best_obj = cur; best_theta = theta.copy()
        if verbose and (it % max(1, ars.n_iters // 10) == 0 or it == ars.n_iters - 1):
            msg = f"  [ARS it {it:>3}] mean_obj={cur:.1f}  best_dir={maxr.max():.1f}"
            if eval_fn is not None:
                msg += f"  eval={eval_fn(make_ars_policy(theta, Z)):.1f}"
            print(msg, flush=True)
        log.append(dict(it=it, mean_obj=float(cur), best=float(maxr.max())))
    if pool:
        pool.close(); pool.join()
    return make_ars_policy(best_theta, Z), dict(theta=best_theta, Z=Z, log=log)


def make_ars_policy(theta, Z):
    pol = ZoneSharedPolicy(Z, params=theta)
    return lambda obs: pol.act(obs)
