"""Run PPO-style delayed-reward MPE2 diagnostics.

This script is a compact MAPPO-style check: a shared decentralized actor is
updated with a clipped PPO surrogate, while a centralized Q critic supplies
counterfactual or sampled-Shapley credit for the assignment variants.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from pathlib import Path
from typing import Literal

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

from acct.credit import acct_advantages
from experiments.run_mpe_delayed_experiment import (
    Actor,
    CentralQ,
    Episode,
    collect_episode,
    critic_influence,
    evaluate_greedy,
    make_env,
    train_critic,
)


Method = Literal[
    "ppo_shared",
    "ppo_learned_agent_time_cf",
    "ppo_learned_agent_shapley",
    "ppo_learned_acct",
    "ppo_learned_directional_acct",
]


def flatten_episode(ep: Episode) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_dim = ep.local_obs.shape[-1]
    obs = torch.as_tensor(ep.local_obs.reshape(-1, obs_dim), dtype=torch.float32)
    actions = torch.as_tensor(ep.actions.reshape(-1), dtype=torch.long)
    old_log_probs = ep.log_probs.detach().reshape(-1)
    return obs, actions, old_log_probs


def compute_credits(
    method: Method,
    ep: Episode,
    actor: Actor,
    critic: CentralQ,
    baseline: float,
    rng: np.random.Generator,
) -> np.ndarray:
    advantage = ep.terminal_return - baseline
    if method == "ppo_shared":
        return np.full(ep.actions.shape, advantage, dtype=np.float32)

    n_agents = ep.actions.shape[1]
    n_actions = critic.n_actions
    obs_dim = ep.local_obs.shape[-1]
    with torch.no_grad():
        logits = actor(torch.as_tensor(ep.local_obs.reshape(-1, obs_dim), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(ep.actions.shape[0], n_agents, n_actions)
    influence = critic_influence(critic, ep.joint_obs, ep.actions, probs)
    if method == "ppo_learned_agent_time_cf":
        return influence
    if method == "ppo_learned_agent_shapley":
        return critic_sampled_agent_shapley(critic, ep.joint_obs, ep.actions, probs, samples=8, rng=rng)
    if method == "ppo_learned_acct":
        td_residuals = np.zeros(ep.actions.shape[0], dtype=np.float32)
        td_residuals[-1] = advantage
        return acct_advantages(influence, td_residuals, gamma=0.99, lam=0.90, residual_weight=0.5)
    if method == "ppo_learned_directional_acct":
        td_residuals = np.zeros(ep.actions.shape[0], dtype=np.float32)
        td_residuals[-1] = advantage
        return acct_advantages(
            influence,
            td_residuals,
            gamma=0.99,
            lam=0.90,
            residual_weight=0.5,
            transport_mode="directional",
        )
    raise ValueError(method)


def critic_sampled_agent_shapley(
    critic: CentralQ,
    joint_obs: np.ndarray,
    actions: np.ndarray,
    action_probs: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate per-time Shapley marginal credit using the learned central Q."""

    horizon, n_agents = actions.shape
    n_actions = action_probs.shape[-1]
    credits = np.zeros((horizon, n_agents), dtype=np.float32)
    with torch.no_grad():
        joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32)
        for t in range(horizon):
            for _ in range(samples):
                reference = np.zeros(n_agents, dtype=np.int64)
                for i in range(n_agents):
                    reference[i] = rng.choice(n_actions, p=action_probs[t, i])
                current = reference.copy()
                current_value = critic(
                    joint_obs_t[t : t + 1],
                    torch.as_tensor(current.reshape(1, -1), dtype=torch.long),
                ).item()
                for i in rng.permutation(n_agents):
                    if current[i] == actions[t, i]:
                        continue
                    previous_value = current_value
                    current[i] = actions[t, i]
                    current_value = critic(
                        joint_obs_t[t : t + 1],
                        torch.as_tensor(current.reshape(1, -1), dtype=torch.long),
                    ).item()
                    credits[t, i] += float(current_value - previous_value)
    return credits / float(samples)


def ppo_update(
    actor: Actor,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    credits: list[np.ndarray],
    clip_eps: float,
    epochs: int,
) -> None:
    obs_parts = []
    action_parts = []
    old_log_parts = []
    adv_parts = []
    for ep, credit in zip(episodes, credits):
        obs, actions, old_log_probs = flatten_episode(ep)
        obs_parts.append(obs)
        action_parts.append(actions)
        old_log_parts.append(old_log_probs)
        adv_parts.append(torch.as_tensor(credit.reshape(-1), dtype=torch.float32))

    obs_t = torch.cat(obs_parts, dim=0)
    actions_t = torch.cat(action_parts, dim=0)
    old_log_t = torch.cat(old_log_parts, dim=0)
    adv_t = torch.cat(adv_parts, dim=0)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-6)

    for _ in range(epochs):
        dist = torch.distributions.Categorical(logits=actor(obs_t))
        new_log_t = dist.log_prob(actions_t)
        ratio = torch.exp(new_log_t - old_log_t)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        loss = -torch.minimum(unclipped, clipped).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
        optimizer.step()


def train_method(
    method: Method,
    seed: int,
    episodes: int,
    max_cycles: int,
    batch_size: int,
    eval_every: int,
    ppo_epochs: int,
) -> list[dict[str, float | int | str]]:
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
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    replay: deque[Episode] = deque(maxlen=96)
    baseline = 0.0
    rows: list[dict[str, float | int | str]] = []
    episode_idx = 0

    while episode_idx < episodes:
        batch: list[Episode] = []
        for _ in range(min(batch_size, episodes - episode_idx)):
            episode_idx += 1
            env = make_env(max_cycles=max_cycles)
            ep = collect_episode(
                env,
                actor,
                rng,
                seed=seed * 1000 + episode_idx,
                greedy_eval_episodes=0,
            )
            env.close()
            batch.append(ep)
            replay.append(ep)
            baseline = 0.97 * baseline + 0.03 * ep.terminal_return

        train_critic(critic, critic_opt, replay, epochs=6)
        credits = [compute_credits(method, ep, actor, critic, baseline, rng) for ep in batch]
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
                    "env": "delayed_simple_spread",
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
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/mpe_ppo_results.csv"))
    args = parser.parse_args()

    methods: tuple[Method, ...] = (
        "ppo_shared",
        "ppo_learned_agent_time_cf",
        "ppo_learned_agent_shapley",
        "ppo_learned_acct",
        "ppo_learned_directional_acct",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env", "method", "seed", "episode", "sample_return", "greedy_return", "baseline"],
            lineterminator="\n",
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
                )
                writer.writerows(rows)
                f.flush()
                row_count += len(rows)
                print(f"finished {method} seed={seed} rows={len(rows)}", flush=True)
    print(f"wrote {row_count} rows to {args.out}")


if __name__ == "__main__":
    main()
