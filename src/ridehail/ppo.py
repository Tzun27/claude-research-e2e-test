"""Minimal, self-contained PPO for the ride-hailing platform controller.

Continuous actions (Gaussian policy over the unbounded raw-action vector; the env squashes
to bounded surge / dispatch / rebalance). Includes GAE, value clipping, entropy bonus, and
running observation normalization. CPU-friendly (small MLPs, low-dim control).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import PPOConfig


class RunningNorm:
    """Welford running mean/std for observation normalization."""
    def __init__(self, dim):
        self.mean = np.zeros(dim); self.var = np.ones(dim); self.count = 1e-4

    def update(self, x):  # x: (N, dim)
        b_mean = x.mean(0); b_var = x.var(0); b_n = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m_a = self.var * self.count; m_b = b_var * b_n
        self.var = (m_a + m_b + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mu.bias)
        self.mu.weight.data *= 0.01

    def forward(self, x):
        h = self.trunk(x)
        return self.mu(h), self.log_std.exp(), self.v(x).squeeze(-1)

    def act(self, x):
        mu, std, v = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), v

    def evaluate(self, x, a):
        mu, std, v = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(a).sum(-1), dist.entropy().sum(-1), v


def train_ppo(make_env, cfg: PPOConfig, eval_fn=None, log_every=20000, verbose=True):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    env = make_env()
    obs_dim, act_dim = env.obs_dim, env.act_dim
    ac = ActorCritic(obs_dim, act_dim, cfg.hidden)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    rms = RunningNorm(obs_dim)

    obs = env.reset(seed=cfg.seed)
    log = {"step": [], "ep_reward": [], "eval": []}
    ep_rew = 0.0; ep_rews = []
    steps = 0
    while steps < cfg.total_steps:
        # ---- collect rollout ----
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(cfg.rollout_len):
            rms.update(obs[None])
            on = rms.norm(obs)
            ot = torch.as_tensor(on, dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = ac.act(ot)
            nobs, r, done, info = env.step(a.numpy())
            O.append(on); A.append(a.numpy()); LP.append(lp.item()); V.append(v.item())
            R.append(r); D.append(float(done))
            ep_rew += r
            obs = nobs
            if done:
                ep_rews.append(ep_rew); ep_rew = 0.0
                obs = env.reset()
            steps += 1
        # bootstrap value
        with torch.no_grad():
            on = rms.norm(obs)
            _, _, last_v = ac.forward(torch.as_tensor(on, dtype=torch.float32))
        # ---- GAE ----
        O = np.array(O, np.float32); A = np.array(A, np.float32)
        LP = np.array(LP, np.float32); V = np.array(V, np.float32)
        R = np.array(R, np.float32); D = np.array(D, np.float32)
        adv = np.zeros_like(R); gae = 0.0; nextv = last_v.item()
        for t in reversed(range(len(R))):
            nonterminal = 1.0 - D[t]
            delta = R[t] + cfg.gamma * nextv * nonterminal - V[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae; nextv = V[t]
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # ---- PPO update ----
        Ot = torch.as_tensor(O); At = torch.as_tensor(A)
        LPt = torch.as_tensor(LP); ADVt = torch.as_tensor(adv); RETt = torch.as_tensor(ret)
        Vt = torch.as_tensor(V)
        n = len(R); idxs = np.arange(n); mb = n // cfg.minibatches
        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for s in range(0, n, mb):
                j = idxs[s:s + mb]
                nlp, ent, v = ac.evaluate(Ot[j], At[j])
                ratio = torch.exp(nlp - LPt[j])
                s1 = ratio * ADVt[j]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADVt[j]
                pol_loss = -torch.min(s1, s2).mean()
                # Plain MSE value loss. (Value clipping at clip=0.2 would freeze a value
                # function whose returns are O(10s), starving advantage estimation.)
                vf = ((v - RETt[j]) ** 2).mean()
                loss = pol_loss + cfg.vf_coef * vf - cfg.ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
        if verbose and ep_rews:
            recent = np.mean(ep_rews[-20:])
            log["step"].append(steps); log["ep_reward"].append(float(recent))
            if steps % log_every < cfg.rollout_len:
                msg = f"  [step {steps:>7}] ep_reward(mean20)={recent:8.3f}"
                if eval_fn is not None:
                    ev = eval_fn(make_policy(ac, rms))
                    log["eval"].append((steps, ev)); msg += f"  eval={ev}"
                print(msg, flush=True)
    return make_policy(ac, rms), ac, rms, log


def make_policy(ac, rms, deterministic=True):
    """Return a policy_fn(obs)->raw_action for evaluation/rollout."""
    def policy_fn(obs):
        on = rms.norm(obs)
        ot = torch.as_tensor(on, dtype=torch.float32)
        with torch.no_grad():
            mu, std, v = ac.forward(ot)
        a = mu if deterministic else torch.distributions.Normal(mu, std).sample()
        return a.numpy()
    return policy_fn
