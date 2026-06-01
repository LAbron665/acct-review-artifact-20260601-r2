"""Run a lightweight delayed-reward MPE2 diagnostic.

This is not a tuned MAPPO benchmark.  It is a small CTDE actor-critic diagnostic
that uses MPE2 Simple Spread dynamics, delays the team reward to the end of the
episode, and compares shared global advantages with learned agent-time
counterfactual credit and learned ACCT transport.
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
from mpe2 import simple_spread_v3

from acct.credit import acct_advantages


Method = Literal["shared", "learned_agent_time_cf", "learned_acct"]


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralQ(nn.Module):
    def __init__(self, joint_obs_dim: int, n_agents: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        input_dim = joint_obs_dim + n_agents * n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(actions.long(), num_classes=self.n_actions).float().reshape(actions.shape[0], -1)
        return self.net(torch.cat([joint_obs, one_hot], dim=-1)).squeeze(-1)


@dataclass
class Episode:
    joint_obs: np.ndarray
    local_obs: np.ndarray
    actions: np.ndarray
    log_probs: torch.Tensor
    terminal_return: float
    greedy_return: float


def make_env(max_cycles: int):
    return simple_spread_v3.parallel_env(N=3, max_cycles=max_cycles, continuous_actions=False, render_mode=None)


def collect_episode(
    env,
    actor: Actor,
    rng: np.random.Generator,
    seed: int | None = None,
    greedy_eval_episodes: int = 3,
) -> Episode:
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    total_team_reward = 0.0
    local_obs = []
    joint_obs = []
    actions = []
    log_probs = []

    while env.agents:
        obs_matrix = np.stack([obs[agent] for agent in agents]).astype(np.float32)
        obs_tensor = torch.as_tensor(obs_matrix, dtype=torch.float32)
        logits = actor(obs_tensor)
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

    # Simple Spread rewards are negative distances/collisions. Larger is better,
    # so normalize by horizon to keep policy-gradient scales stable.
    normalized_return = total_team_reward / max(len(actions), 1)
    if greedy_eval_episodes > 0:
        greedy_return = evaluate_greedy(
            actor,
            seed=int(rng.integers(0, 2**31 - 1)),
            max_cycles=len(actions),
            episodes=greedy_eval_episodes,
        )
    else:
        greedy_return = float("nan")
    return Episode(
        joint_obs=np.asarray(joint_obs, dtype=np.float32),
        local_obs=np.asarray(local_obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        log_probs=torch.stack(log_probs),
        terminal_return=normalized_return,
        greedy_return=greedy_return,
    )


def evaluate_greedy(actor: Actor, seed: int, max_cycles: int, episodes: int = 3) -> float:
    returns = []
    for offset in range(episodes):
        env = make_env(max_cycles=max_cycles)
        obs, _ = env.reset(seed=seed + offset)
        agents = list(env.agents)
        total = 0.0
        steps = 0
        while env.agents:
            obs_matrix = np.stack([obs[agent] for agent in agents]).astype(np.float32)
            with torch.no_grad():
                action_tensor = actor(torch.as_tensor(obs_matrix, dtype=torch.float32)).argmax(dim=-1)
            action_dict = {agent: int(action_tensor[idx].item()) for idx, agent in enumerate(agents)}
            obs, rewards, terms, truncs, _ = env.step(action_dict)
            total += float(np.mean([rewards[agent] for agent in agents]))
            steps += 1
            if any(terms.values()) or any(truncs.values()):
                break
        env.close()
        returns.append(total / max(steps, 1))
    return float(np.mean(returns))


def train_critic(critic: CentralQ, optimizer: torch.optim.Optimizer, replay: deque[Episode], epochs: int = 4) -> None:
    if not replay:
        return
    joint_obs = np.concatenate([ep.joint_obs for ep in replay], axis=0)
    actions = np.concatenate([ep.actions for ep in replay], axis=0)
    targets = np.concatenate([np.full(len(ep.actions), ep.terminal_return, dtype=np.float32) for ep in replay], axis=0)
    joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.long)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    for _ in range(epochs):
        pred = critic(joint_obs_t, actions_t)
        loss = F.mse_loss(pred, targets_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def critic_influence(critic: CentralQ, joint_obs: np.ndarray, actions: np.ndarray, action_probs: np.ndarray) -> np.ndarray:
    horizon, n_agents = actions.shape
    influence = np.zeros((horizon, n_agents), dtype=np.float32)
    with torch.no_grad():
        joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32)
        action_t = torch.as_tensor(actions, dtype=torch.long)
        actual = critic(joint_obs_t, action_t).cpu().numpy()
        for t in range(horizon):
            for i in range(n_agents):
                expected = 0.0
                for action in range(action_probs.shape[-1]):
                    replaced = actions.copy()
                    replaced[t, i] = action
                    value = critic(
                        joint_obs_t[t : t + 1],
                        torch.as_tensor(replaced[t : t + 1], dtype=torch.long),
                    ).item()
                    expected += float(action_probs[t, i, action]) * value
                influence[t, i] = float(actual[t] - expected)
    return influence


def train_method(method: Method, seed: int, episodes: int, max_cycles: int, eval_every: int) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    probe_env = make_env(max_cycles=max_cycles)
    obs, _ = probe_env.reset(seed=seed)
    agent0 = probe_env.agents[0]
    obs_dim = int(obs[agent0].shape[0])
    n_actions = int(probe_env.action_space(agent0).n)
    n_agents = len(probe_env.agents)
    probe_env.close()

    actor = Actor(obs_dim, n_actions)
    critic = CentralQ(obs_dim * n_agents, n_agents, n_actions)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    replay: deque[Episode] = deque(maxlen=64)
    baseline = 0.0
    rows: list[dict[str, float | int | str]] = []

    for episode_idx in range(1, episodes + 1):
        env = make_env(max_cycles=max_cycles)
        episode = collect_episode(env, actor, rng, seed=seed * 1000 + episode_idx)
        env.close()
        replay.append(episode)
        train_critic(critic, critic_opt, replay)

        baseline = 0.95 * baseline + 0.05 * episode.terminal_return
        advantage = episode.terminal_return - baseline
        if method == "shared":
            credits = np.full(episode.actions.shape, advantage, dtype=np.float32)
        else:
            with torch.no_grad():
                logits = actor(torch.as_tensor(episode.local_obs.reshape(-1, obs_dim), dtype=torch.float32))
                probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(
                    episode.actions.shape[0], n_agents, n_actions
                )
            influence = critic_influence(critic, episode.joint_obs, episode.actions, probs)
            if method == "learned_agent_time_cf":
                credits = influence
            elif method == "learned_acct":
                td_residuals = np.zeros(episode.actions.shape[0], dtype=np.float32)
                td_residuals[-1] = advantage
                credits = acct_advantages(influence, td_residuals, gamma=0.99, lam=0.90, residual_weight=0.5)
            else:
                raise ValueError(method)

        credit_t = torch.as_tensor(credits, dtype=torch.float32)
        credit_t = (credit_t - credit_t.mean()) / (credit_t.std(unbiased=False) + 1e-6)
        actor_loss = -(episode.log_probs * credit_t).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        if episode_idx % eval_every == 0 or episode_idx == 1:
            rows.append(
                {
                    "env": "delayed_simple_spread",
                    "method": method,
                    "seed": seed,
                    "episode": episode_idx,
                    "sample_return": episode.terminal_return,
                    "greedy_return": episode.greedy_return,
                    "baseline": baseline,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/mpe_delayed_results.csv"))
    args = parser.parse_args()

    methods: tuple[Method, ...] = ("shared", "learned_agent_time_cf", "learned_acct")
    rows = []
    for method in methods:
        for seed in range(args.seeds):
            rows.extend(train_method(method, seed=seed, episodes=args.episodes, max_cycles=args.max_cycles, eval_every=args.eval_every))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env", "method", "seed", "episode", "sample_return", "greedy_return", "baseline"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
