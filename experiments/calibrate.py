"""Calibrate demand elasticity, fleet size, and entry/exit dynamics.

Target: short-run price elasticity ~0.27 (Castillo 2025) at the operating point, with a
market that is neither trivially over- nor under-supplied (match rate ~0.7-0.9 at mult=1),
and visible entry/exit so part-time drivers concentrate at peaks.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ridehail import SimConfig, RideHailEnv


def fixed_policy(env, mult, dispatch_logit=0.0):
    cfg = env.cfg
    frac = (mult - cfg.surge_min) / (cfg.surge_max - cfg.surge_min)
    frac = min(max(frac, 1e-4), 1 - 1e-4)
    raw = np.zeros(env.act_dim)
    raw[:env.Z] = np.log(frac / (1 - frac))
    raw[env.Z] = dispatch_logit
    raw[env.Z + 1:] = -10.0
    return raw


def episode_stats(cfg, mult, seed, dispatch_logit=0.0):
    env = RideHailEnv(cfg, objective="welfare")
    raw = fixed_policy(env, mult, dispatch_logit)
    obs = env.reset(seed=seed); done = False
    req = match = aband = 0; pickups = []
    while not done:
        obs, r, done, info = env.step(raw)
        req += info['n_requests']; match += info['n_matched']; aband += info['n_abandon']
        pickups.append(info['mean_pickup'])
    w = info['welfare']
    return dict(req=req, match=match, aband=aband, match_rate=match / max(1, req),
                mean_pickup=float(np.mean(pickups)), W=w['total_welfare'],
                gross=w['gross_revenue'])


def measure_elasticity(cfg, seeds=range(6)):
    """Demand price-elasticity at mult=1 using high fleet (uncongested) so requests==demand."""
    big = SimConfig(**{**cfg.__dict__})
    big.n_drivers = 2000  # flood supply so exp_wait ~ 0 and requests reflect pure price response
    lo, hi = 0.95, 1.05
    rlo = np.mean([episode_stats(big, lo, s)['req'] for s in seeds])
    rhi = np.mean([episode_stats(big, hi, s)['req'] for s in seeds])
    eps = -((rhi - rlo) / ((rhi + rlo) / 2)) / ((hi - lo) / ((hi + lo) / 2))
    rbase = np.mean([episode_stats(big, 1.0, s)['req'] for s in seeds])
    # acceptance rate at mult=1 = requests / potential (potential ~ sum of lambda)
    pot = big.lam_sum if hasattr(big, 'lam_sum') else None
    env = RideHailEnv(big); pot = float(env.lam.sum())
    return eps, rbase / pot


def main():
    base = SimConfig()
    print(f"WTP: median ratio = exp(mu)={np.exp(base.wtp_log_mu):.2f}, sigma={base.wtp_log_sigma}")
    eps, acc = measure_elasticity(base)
    print(f"Short-run price elasticity at mult=1: {eps:.3f}  (target ~0.27)")
    print(f"Acceptance rate at mult=1 (uncongested): {acc:.3f}")
    print("-" * 60)
    print("Fleet-size sweep (match rate & welfare at mult=1.0):")
    for nd in [200, 300, 400, 500, 700, 1000]:
        cfg = SimConfig(n_drivers=nd)
        st = [episode_stats(cfg, 1.0, s) for s in range(4)]
        mr = np.mean([s['match_rate'] for s in st])
        pu = np.mean([s['mean_pickup'] for s in st])
        W = np.mean([s['W'] for s in st])
        print(f"  n_drivers={nd:<5} match_rate={mr:.2f}  mean_pickup={pu:.2f}  W={W:9.0f}")
    print("-" * 60)
    print("Surge sweep at chosen fleet (match rate & gross revenue):")
    cfg = SimConfig(n_drivers=400)
    for mult in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
        st = [episode_stats(cfg, mult, s) for s in range(4)]
        mr = np.mean([s['match_rate'] for s in st])
        rq = np.mean([s['req'] for s in st])
        print(f"  mult={mult:<4} requests={rq:7.0f} match_rate={mr:.2f} "
              f"gross={np.mean([s['gross'] for s in st]):8.0f}")


if __name__ == "__main__":
    main()
