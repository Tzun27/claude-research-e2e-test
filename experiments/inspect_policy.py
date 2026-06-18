"""Characterize what a learned controller does: surge by zone (map), surge over the day,
and surge vs. local supply-demand imbalance. Reads a result JSON's saved theta.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ridehail import SimConfig, RideHailEnv
from ridehail.ars import make_ars_policy

FIG = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIG, exist_ok=True)


def rollout_record(res, n_seeds=8):
    cfg = SimConfig(**res.get("cfg_over", {}))
    if res["condition"] in ("price", "fair"):
        cfg = SimConfig(**{**cfg.__dict__, "rebalance_enabled": False})
        env_kwargs = dict(fix_dispatch_radius=res["fix_dispatch"] or 2)
    else:
        cfg = SimConfig(**{**cfg.__dict__, "rebalance_enabled": True})
        env_kwargs = dict(fix_dispatch_radius=None)
    Z, G = cfg.n_zones(), cfg.grid_size
    theta = np.array(res["surge_thetas"][0]); pol = make_ars_policy(theta, Z)
    surge_zt = np.zeros((cfg.T, Z)); demand_t = np.zeros(cfg.T)
    imb, surges = [], []
    for sd in range(200, 200 + n_seeds):
        env = RideHailEnv(cfg, objective=res["objective"], **env_kwargs)
        obs = env.reset(seed=sd); t = 0; done = False
        while not done:
            raw = pol(obs); su, dr, rb = env.decode_action(raw)
            surge_zt[t] += su / n_seeds; demand_t[t] += env.lam[t].sum() / n_seeds
            supply = obs[:Z] * (cfg.n_drivers / Z); dem = env.lam[t]
            imb.extend(((dem - supply) / (dem + supply + 1)).tolist()); surges.extend(su.tolist())
            obs, r, done, info = env.step(raw); t += 1
    return cfg, surge_zt, demand_t, np.array(imb), np.array(surges)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--result", required=True)
    args = ap.parse_args()
    with open(args.result) as f:
        res = json.load(f)
    cfg, surge_zt, demand_t, imb, surges = rollout_record(res)
    G = cfg.grid_size; tag = os.path.basename(args.result)[:-5]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) mean surge by zone (peak epoch)
    peak_t = int(np.argmax(demand_t))
    im = ax[0].imshow(surge_zt[peak_t].reshape(G, G), cmap="hot_r", vmin=cfg.surge_min)
    ax[0].set_title(f"Mean surge by zone (peak epoch {peak_t})"); fig.colorbar(im, ax=ax[0])
    # (b) surge over the day vs demand
    ax[1].plot(surge_zt.mean(1), "b-", label="mean surge")
    ax2 = ax[1].twinx(); ax2.plot(demand_t, "r--", alpha=0.6, label="demand")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean surge"); ax2.set_ylabel("demand")
    ax[1].set_title("Surge vs demand over the day")
    # (c) surge vs local imbalance
    ax[2].scatter(imb, surges, s=2, alpha=0.1)
    ax[2].set_xlabel("local imbalance (demand-supply)/(.)"); ax[2].set_ylabel("surge")
    ax[2].set_title("Learned surge vs local imbalance")
    fig.suptitle(f"{tag}: {res['objective']} ({res['condition']})")
    fig.tight_layout(); out = os.path.join(FIG, f"policy_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig); print(f"saved {out}")


if __name__ == "__main__":
    main()
