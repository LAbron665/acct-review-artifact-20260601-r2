"""Run a compact MAPPO-style delayed-reward MPE2 diagnostic.

This script is still a local diagnostic, not a full MAPPO reproduction.  It
adds the key missing ingredient from the lighter PPO smoke test: a centralized
state-value critic trained on delayed return-to-go targets and a GAE-style
shared-advantage baseline.  Learned ACCT variants use the same trajectories and
an additional centralized Q critic for counterfactual influence.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

from acct.credit import acct_advantages
from experiments.run_mpe_delayed_experiment import Actor, CentralQ, critic_influence, evaluate_greedy, make_env
from experiments.run_mpe_ppo_experiment import ppo_update


Method = Literal[
    "mappo_shared_gae",
    "mappo_learned_agent_time_cf",
    "mappo_learned_acct",
    "mappo_gae_plus_agent_time_cf",
    "mappo_gae_plus_acct",
]


class CentralV(nn.Module):
    def __init__(self, joint_obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.net(joint_obs).squeeze(-1)


@dataclass
class Trajectory:
    joint_obs: np.ndarray
    local_obs: np.ndarray
    actions: np.ndarray
    log_probs: torch.Tensor
    delayed_rewards: np.ndarray
    terminal_return: float
    greedy_return: float = float("nan")


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def collect_delayed_episode(env, actor: Actor, seed: int | None = None) -> Trajectory:
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    total_team_reward = 0.0
    local_obs = []
    joint_obs = []
    actions = []
    log_probs = []

    while env.agents:
        obs_matrix = np.stack([obs[agent] for agent in agents]).astype(np.float32)
        logits = actor(torch.as_tensor(obs_matrix, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        action_tensor = dist.sample()
        action_dict = {agent: int(action_tensor[idx].item()) for idx, agent in enumerate(agents)}

        next_obs, rewards, terms, truncs, _ = env.step(action_dict)
        local_obs.append(obs_matrix)
        joint_obs.append(obs_matrix.reshape(-1))
        actions.append(action_tensor.detach().cpu().numpy())
        log_probs.append(dist.log_prob(action_tensor))
        total_team_reward += float(np.mean([rewards[agent] for agent in agents]))
        obs = next_obs
        if any(terms.values()) or any(truncs.values()):
            break

    horizon = max(len(actions), 1)
    terminal_return = total_team_reward / horizon
    delayed_rewards = np.zeros(horizon, dtype=np.float32)
    delayed_rewards[-1] = terminal_return
    return Trajectory(
        joint_obs=np.asarray(joint_obs, dtype=np.float32),
        local_obs=np.asarray(local_obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        log_probs=torch.stack(log_probs),
        delayed_rewards=delayed_rewards,
        terminal_return=terminal_return,
    )


def train_value(critic: CentralV, optimizer: torch.optim.Optimizer, replay: deque[Trajectory], gamma: float, epochs: int) -> None:
    if not replay:
        return
    joint_obs = np.concatenate([ep.joint_obs for ep in replay], axis=0)
    targets = np.concatenate([discounted_returns(ep.delayed_rewards, gamma) for ep in replay], axis=0)
    joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    for _ in range(epochs):
        pred = critic(joint_obs_t)
        loss = F.mse_loss(pred, targets_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
        optimizer.step()


def train_q(critic: CentralQ, optimizer: torch.optim.Optimizer, replay: deque[Trajectory], gamma: float, epochs: int) -> None:
    if not replay:
        return
    joint_obs = np.concatenate([ep.joint_obs for ep in replay], axis=0)
    actions = np.concatenate([ep.actions for ep in replay], axis=0)
    targets = np.concatenate([discounted_returns(ep.delayed_rewards, gamma) for ep in replay], axis=0)
    joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.long)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    for _ in range(epochs):
        pred = critic(joint_obs_t, actions_t)
        loss = F.mse_loss(pred, targets_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
        optimizer.step()


def value_td_and_gae(value_critic: CentralV, ep: Trajectory, gamma: float, gae_lam: float) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        values = value_critic(torch.as_tensor(ep.joint_obs, dtype=torch.float32)).cpu().numpy()
    next_values = np.concatenate([values[1:], np.asarray([0.0], dtype=np.float32)])
    td = ep.delayed_rewards + gamma * next_values - values
    advantages = np.zeros_like(td, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(td))):
        running = float(td[t]) + gamma * gae_lam * running
        advantages[t] = running
    return td.astype(np.float32), advantages


def standardize_signal(signal: np.ndarray) -> np.ndarray:
    centered = signal.astype(np.float32) - float(np.mean(signal))
    scale = float(np.std(centered))
    if scale <= 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    return centered / scale


def compute_credits(
    method: Method,
    ep: Trajectory,
    actor: Actor,
    value_critic: CentralV,
    q_critic: CentralQ,
    gamma: float,
    gae_lam: float,
    hybrid_weight: float,
) -> np.ndarray:
    td_residuals, shared_gae = value_td_and_gae(value_critic, ep, gamma=gamma, gae_lam=gae_lam)
    shared = np.repeat(shared_gae[:, None], ep.actions.shape[1], axis=1)
    if method == "mappo_shared_gae":
        return shared

    n_agents = ep.actions.shape[1]
    n_actions = q_critic.n_actions
    obs_dim = ep.local_obs.shape[-1]
    with torch.no_grad():
        logits = actor(torch.as_tensor(ep.local_obs.reshape(-1, obs_dim), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(ep.actions.shape[0], n_agents, n_actions)
    influence = critic_influence(q_critic, ep.joint_obs, ep.actions, probs)
    if method == "mappo_learned_agent_time_cf":
        return influence
    if method == "mappo_gae_plus_agent_time_cf":
        return shared + hybrid_weight * standardize_signal(influence)
    if method == "mappo_learned_acct":
        return acct_advantages(influence, td_residuals, gamma=gamma, lam=0.90, residual_weight=0.5)
    if method == "mappo_gae_plus_acct":
        acct_credit = acct_advantages(influence, td_residuals, gamma=gamma, lam=0.90, residual_weight=0.5)
        return shared + hybrid_weight * standardize_signal(acct_credit)
    raise ValueError(method)


def train_method(
    method: Method,
    seed: int,
    episodes: int,
    max_cycles: int,
    batch_size: int,
    eval_every: int,
    ppo_epochs: int,
    gamma: float,
    gae_lam: float,
    hybrid_weight: float,
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe_env = make_env(max_cycles=max_cycles)
    obs, _ = probe_env.reset(seed=seed)
    agent0 = probe_env.agents[0]
    obs_dim = int(obs[agent0].shape[0])
    n_actions = int(probe_env.action_space(agent0).n)
    n_agents = len(probe_env.agents)
    probe_env.close()

    actor = Actor(obs_dim, n_actions)
    value_critic = CentralV(obs_dim * n_agents)
    q_critic = CentralQ(obs_dim * n_agents, n_agents, n_actions)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-3)
    value_opt = torch.optim.Adam(value_critic.parameters(), lr=3e-3)
    q_opt = torch.optim.Adam(q_critic.parameters(), lr=3e-3)
    replay: deque[Trajectory] = deque(maxlen=128)
    rows: list[dict[str, float | int | str]] = []
    episode_idx = 0

    while episode_idx < episodes:
        batch: list[Trajectory] = []
        for _ in range(min(batch_size, episodes - episode_idx)):
            episode_idx += 1
            env = make_env(max_cycles=max_cycles)
            ep = collect_delayed_episode(env, actor, seed=seed * 1000 + episode_idx)
            env.close()
            batch.append(ep)
            replay.append(ep)

        train_value(value_critic, value_opt, replay, gamma=gamma, epochs=6)
        train_q(q_critic, q_opt, replay, gamma=gamma, epochs=6)
        credits = [
            compute_credits(
                method,
                ep,
                actor,
                value_critic,
                q_critic,
                gamma=gamma,
                gae_lam=gae_lam,
                hybrid_weight=hybrid_weight,
            )
            for ep in batch
        ]
        ppo_update(actor, actor_opt, batch, credits, clip_eps=0.2, epochs=ppo_epochs)

        if episode_idx % eval_every == 0 or episode_idx == episodes:
            greedy = evaluate_greedy(
                actor,
                seed=seed * 100_000 + episode_idx,
                max_cycles=max_cycles,
                episodes=5,
            )
            rows.append(
                {
                    "env": "delayed_simple_spread_mappo",
                    "method": method,
                    "seed": seed,
                    "episode": episode_idx,
                    "sample_return": float(np.mean([ep.terminal_return for ep in batch])),
                    "greedy_return": greedy,
                    "baseline": float("nan"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lam", type=float, default=0.95)
    parser.add_argument("--hybrid-weight", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/mpe_mappo_results.csv"))
    args = parser.parse_args()

    methods: tuple[Method, ...] = (
        "mappo_shared_gae",
        "mappo_learned_agent_time_cf",
        "mappo_learned_acct",
        "mappo_gae_plus_agent_time_cf",
        "mappo_gae_plus_acct",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env", "method", "seed", "episode", "sample_return", "greedy_return", "baseline"],
        )
        writer.writeheader()
        row_count = 0
        for method in methods:
            for seed in range(args.seeds):
                print(f"running {method} seed={seed}", flush=True)
                rows = train_method(
                    method,
                    seed=seed,
                    episodes=args.episodes,
                    max_cycles=args.max_cycles,
                    batch_size=args.batch_size,
                    eval_every=args.eval_every,
                    ppo_epochs=args.ppo_epochs,
                    gamma=args.gamma,
                    gae_lam=args.gae_lam,
                    hybrid_weight=args.hybrid_weight,
                )
                writer.writerows(rows)
                f.flush()
                row_count += len(rows)
                print(f"finished {method} seed={seed} rows={len(rows)}", flush=True)
    print(f"wrote {row_count} rows to {args.out}")


if __name__ == "__main__":
    main()
