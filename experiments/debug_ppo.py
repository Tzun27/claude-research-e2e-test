"""Fast PPO diagnostic: does the controller learn the profit-maximizing surge level?

Logs, per rollout, the learned mean surge and the objective, under a few hyperparameter
settings, to find configs that learn quickly. Profit-max should drive surge up toward the
WTP-cap region; we compare the achieved objective to the uniform grid-search optimum.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from ridehail import SimConfig, PPOConfig, RideHailEnv
from ridehail.config import REWARD_SCALES
from ridehail.ppo import ActorCritic, RunningNorm, make_policy
from ridehail.baselines import uniform_search
from ridehail.evaluate import run_policy_seeds, mean_summary, objective_value


def train_logged(mk, obj, total_steps, lr, ent_coef, std0, anneal_std, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    env = mk()
    ac = ActorCritic(env.obs_dim, env.act_dim, 128)
    with torch.no_grad():
        ac.log_std.fill_(np.log(std0))
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    rms = RunningNorm(env.obs_dim)
    cfg = PPOConfig(total_steps=total_steps, lr=lr, ent_coef=ent_coef)
    obs = env.reset(seed=seed); steps = 0; rollout = cfg.rollout_len
    test_seed = 999
    while steps < total_steps:
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(rollout):
            rms.update(obs[None]); on = rms.norm(obs)
            ot = torch.as_tensor(on, dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = ac.act(ot)
            nobs, r, done, info = env.step(a.numpy())
            O.append(on); A.append(a.numpy()); LP.append(lp.item()); V.append(v.item())
            R.append(r); D.append(float(done)); obs = nobs
            if done: obs = env.reset()
            steps += 1
        with torch.no_grad():
            _, _, last_v = ac.forward(torch.as_tensor(rms.norm(obs), dtype=torch.float32))
        O = np.array(O, np.float32); A = np.array(A, np.float32); LP = np.array(LP, np.float32)
        V = np.array(V, np.float32); R = np.array(R, np.float32); D = np.array(D, np.float32)
        adv = np.zeros_like(R); gae = 0; nextv = last_v.item()
        for t in reversed(range(len(R))):
            nt = 1 - D[t]; delta = R[t] + cfg.gamma * nextv * nt - V[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nt * gae; adv[t] = gae; nextv = V[t]
        ret = adv + V; expl_var = 1 - np.var(ret - V) / (np.var(ret) + 1e-8)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        Ot = torch.as_tensor(O); At = torch.as_tensor(A); LPt = torch.as_tensor(LP)
        ADVt = torch.as_tensor(adv); RETt = torch.as_tensor(ret)
        n = len(R); idxs = np.arange(n); mb = n // cfg.minibatches
        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for s in range(0, n, mb):
                j = idxs[s:s + mb]
                nlp, ent, vv = ac.evaluate(Ot[j], At[j])
                ratio = torch.exp(nlp - LPt[j])
                s1 = ratio * ADVt[j]; s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADVt[j]
                pol = -torch.min(s1, s2).mean(); vf = ((vv - RETt[j]) ** 2).mean()
                loss = pol + cfg.vf_coef * vf - ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm); opt.step()
        if anneal_std:
            with torch.no_grad():
                frac = 1 - steps / total_steps
                ac.log_std.clamp_(np.log(0.05), np.log(std0 * frac + 0.05))
        # log learned mean surge deterministically
        pol_fn = make_policy(ac, rms)
        e2 = mk(); o2 = e2.reset(seed=test_seed); sl = []; done = False
        while not done:
            raw = pol_fn(o2); su, dr, rb = e2.decode_action(raw); sl.append(su.mean())
            o2, r, done, info = e2.step(raw)
        print(f"    step {steps:>6} surge={np.mean(sl):.2f} std={ac.log_std.exp().mean():.2f} "
              f"explVar={expl_var:.2f}", flush=True)
    return make_policy(ac, rms)


def main():
    obj = "profit"
    cfg = SimConfig(rebalance_enabled=False)
    mk = lambda: RideHailEnv(cfg, objective=obj, fix_dispatch_radius=2, reward_scale=REWARD_SCALES[obj])
    seeds = list(range(20, 26))
    bm, info = uniform_search(mk, obj, seeds=seeds, dispatch_radius=2)
    uni_obj = objective_value(mean_summary(info["best_summaries"]), obj)
    print(f"uniform m*={bm:.2f}  uniform_obj={uni_obj:.0f}")
    for tag, kw in [
        ("lr1e-3 ent0 std0.8 anneal", dict(lr=1e-3, ent_coef=0.0, std0=0.8, anneal_std=True)),
        ("lr5e-4 ent.003 std0.6", dict(lr=5e-4, ent_coef=0.003, std0=0.6, anneal_std=False)),
    ]:
        print(f"--- {tag} ---")
        t0 = time.time()
        pol = train_logged(mk, obj, total_steps=60000, **kw)
        sums = run_policy_seeds(mk, pol, seeds)
        print(f"  => surge_obj={objective_value(mean_summary(sums), obj):.0f} "
              f"(uniform {uni_obj:.0f})  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
