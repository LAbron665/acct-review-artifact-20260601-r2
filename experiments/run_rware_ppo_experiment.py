"""Run a tiny PPO-style delayed-reward RWARE diagnostic."""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from collections import deque
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import rware  # noqa: F401 - registers Gymnasium environments
import torch

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

from acct.credit import acct_advantages
from experiments.run_mpe_delayed_experiment import Actor, CentralQ, Episode, critic_influence, train_critic
from experiments.run_mpe_ppo_experiment import ppo_update


Method = Literal["rware_ppo_shared", "rware_ppo_learned_agent_time_cf", "rware_ppo_learned_acct"]


def make_env(env_id: str, max_steps: int):
    warnings.filterwarnings("ignore", message=".*reward returned by `step`.*")
    return gym.make(env_id, max_steps=max_steps, disable_env_checker=True)


def collect_episode(env, actor: Actor, seed: int | None = None) -> Episode:
    obs, _ = env.reset(seed=seed)
    n_agents = len(obs)
    total_team_reward = 0.0
    local_obs = []
    joint_obs = []
    actions = []
    log_probs = []

    while True:
        obs_matrix = np.stack(obs).astype(np.float32)
        logits = actor(torch.as_tensor(obs_matrix, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        action_tensor = dist.sample()
        action_tuple = tuple(int(action_tensor[idx].item()) for idx in range(n_agents))

        obs, rewards, terminated, truncated, _ = env.step(action_tuple)
        local_obs.append(obs_matrix)
        joint_obs.append(obs_matrix.reshape(-1))
        actions.append(action_tensor.detach().cpu().numpy())
        log_probs.append(dist.log_prob(action_tensor))
        total_team_reward += float(np.mean(rewards))
        if terminated or truncated:
            break

    return Episode(
        joint_obs=np.asarray(joint_obs, dtype=np.float32),
        local_obs=np.asarray(local_obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        log_probs=torch.stack(log_probs),
        terminal_return=total_team_reward,
        greedy_return=float("nan"),
    )


def evaluate_greedy(actor: Actor, env_id: str, seed: int, max_steps: int, episodes: int = 8) -> float:
    returns = []
    for offset in range(episodes):
        env = make_env(env_id=env_id, max_steps=max_steps)
        obs, _ = env.reset(seed=seed + offset)
        total = 0.0
        while True:
            obs_matrix = np.stack(obs).astype(np.float32)
            with torch.no_grad():
                action_tensor = actor(torch.as_tensor(obs_matrix, dtype=torch.float32)).argmax(dim=-1)
            action_tuple = tuple(int(action.item()) for action in action_tensor)
            obs, rewards, terminated, truncated, _ = env.step(action_tuple)
            total += float(np.mean(rewards))
            if terminated or truncated:
                break
        env.close()
        returns.append(total)
    return float(np.mean(returns))


def compute_credits(method: Method, ep: Episode, actor: Actor, critic: CentralQ, baseline: float) -> np.ndarray:
    advantage = ep.terminal_return - baseline
    if method == "rware_ppo_shared":
        return np.full(ep.actions.shape, advantage, dtype=np.float32)

    n_agents = ep.actions.shape[1]
    n_actions = critic.n_actions
    obs_dim = ep.local_obs.shape[-1]
    with torch.no_grad():
        logits = actor(torch.as_tensor(ep.local_obs.reshape(-1, obs_dim), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(ep.actions.shape[0], n_agents, n_actions)
    influence = critic_influence(critic, ep.joint_obs, ep.actions, probs)
    if method == "rware_ppo_learned_agent_time_cf":
        return influence
    if method == "rware_ppo_learned_acct":
        td_residuals = np.zeros(ep.actions.shape[0], dtype=np.float32)
        td_residuals[-1] = advantage
        return acct_advantages(influence, td_residuals, gamma=0.99, lam=0.90, residual_weight=0.5)
    raise ValueError(method)


def train_method(
    method: Method,
    seed: int,
    env_id: str,
    episodes: int,
    max_steps: int,
    batch_size: int,
    eval_every: int,
    ppo_epochs: int,
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe_env = make_env(env_id=env_id, max_steps=max_steps)
    obs, _ = probe_env.reset(seed=seed)
    obs_dim = int(obs[0].shape[0])
    n_actions = int(probe_env.action_space[0].n)
    n_agents = len(obs)
    probe_env.close()

    actor = Actor(obs_dim, n_actions)
    critic = CentralQ(obs_dim * n_agents, n_agents, n_actions)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    replay: deque[Episode] = deque(maxlen=128)
    baseline = 0.0
    rows: list[dict[str, float | int | str]] = []
    episode_idx = 0

    while episode_idx < episodes:
        batch: list[Episode] = []
        for _ in range(min(batch_size, episodes - episode_idx)):
            episode_idx += 1
            env = make_env(env_id=env_id, max_steps=max_steps)
            ep = collect_episode(env, actor, seed=seed * 1000 + episode_idx)
            env.close()
            batch.append(ep)
            replay.append(ep)
            baseline = 0.97 * baseline + 0.03 * ep.terminal_return

        train_critic(critic, critic_opt, replay, epochs=6)
        credits = [compute_credits(method, ep, actor, critic, baseline) for ep in batch]
        ppo_update(actor, actor_opt, batch, credits, clip_eps=0.2, epochs=ppo_epochs)

        if episode_idx % eval_every == 0 or episode_idx == episodes:
            greedy = evaluate_greedy(actor, env_id=env_id, seed=seed * 100_000 + episode_idx, max_steps=max_steps)
            rows.append(
                {
                    "env": env_id,
                    "method": method,
                    "seed": seed,
                    "episode": episode_idx,
                    "sample_return": float(np.mean([ep.terminal_return for ep in batch])),
                    "greedy_return": greedy,
                    "baseline": baseline,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="rware-tiny-2ag-easy-v2")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/rware_ppo_results.csv"))
    args = parser.parse_args()

    methods: tuple[Method, ...] = ("rware_ppo_shared", "rware_ppo_learned_agent_time_cf", "rware_ppo_learned_acct")
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
                    env_id=args.env_id,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    eval_every=args.eval_every,
                    ppo_epochs=args.ppo_epochs,
                )
                writer.writerows(rows)
                f.flush()
                row_count += len(rows)
                print(f"finished {method} seed={seed} rows={len(rows)}", flush=True)
    print(f"wrote {row_count} rows to {args.out}")


if __name__ == "__main__":
    main()
