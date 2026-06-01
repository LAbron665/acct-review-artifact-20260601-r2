"""Diagnose learned counterfactual signal quality in delayed MPE2.

The diagnostic trains the compact MAPPO-style shared-GAE baseline, then
evaluates the learned centralized V and Q critics on held-out trajectories.  It
does not provide ground-truth counterfactual labels; instead it reports whether
the learned Q head fits observed return-to-go targets and whether its
counterfactual influence is aligned with the centralized GAE signal.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch

from acct.credit import acct_advantages
from experiments.run_mpe_delayed_experiment import Actor, CentralQ, critic_influence, make_env
from experiments.run_mpe_mappo_experiment import (
    CentralV,
    Trajectory,
    collect_delayed_episode,
    discounted_returns,
    train_q,
    train_value,
    value_td_and_gae,
)
from experiments.run_mpe_ppo_experiment import ppo_update


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = float(np.sum((target - target.mean()) ** 2))
    if denom <= 1e-8:
        return float("nan")
    return float(1.0 - np.sum((target - pred) ** 2) / denom)


def counterfactual_action_spread(critic: CentralQ, joint_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    horizon, n_agents = actions.shape
    n_actions = critic.n_actions
    spreads = np.zeros((horizon, n_agents), dtype=np.float32)
    with torch.no_grad():
        joint_obs_t = torch.as_tensor(joint_obs, dtype=torch.float32)
        for t in range(horizon):
            for i in range(n_agents):
                values = []
                for action in range(n_actions):
                    replaced = actions.copy()
                    replaced[t, i] = action
                    value = critic(
                        joint_obs_t[t : t + 1],
                        torch.as_tensor(replaced[t : t + 1], dtype=torch.long),
                    ).item()
                    values.append(value)
                spreads[t, i] = float(np.std(values))
    return spreads


def train_shared_gae_model(
    seed: int,
    episodes: int,
    max_cycles: int,
    batch_size: int,
    ppo_epochs: int,
    gamma: float,
    gae_lam: float,
) -> tuple[Actor, CentralV, CentralQ, int, int, int]:
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
        credits = []
        for ep in batch:
            _, shared_gae = value_td_and_gae(value_critic, ep, gamma=gamma, gae_lam=gae_lam)
            credits.append(np.repeat(shared_gae[:, None], ep.actions.shape[1], axis=1))
        ppo_update(actor, actor_opt, batch, credits, clip_eps=0.2, epochs=ppo_epochs)

    return actor, value_critic, q_critic, obs_dim, n_agents, n_actions


def evaluate_seed(
    seed: int,
    episodes: int,
    holdout: int,
    max_cycles: int,
    batch_size: int,
    ppo_epochs: int,
    gamma: float,
    gae_lam: float,
) -> dict[str, float | int | str]:
    actor, value_critic, q_critic, _, n_agents, n_actions = train_shared_gae_model(
        seed=seed,
        episodes=episodes,
        max_cycles=max_cycles,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        gamma=gamma,
        gae_lam=gae_lam,
    )

    q_targets = []
    q_preds = []
    v_targets = []
    v_preds = []
    abs_influence = []
    action_spreads = []
    time_abs_influence = []
    time_abs_gae = []
    signed_influence = []
    repeated_gae = []
    acct_credit_values = []

    for idx in range(holdout):
        env = make_env(max_cycles=max_cycles)
        ep = collect_delayed_episode(env, actor, seed=seed * 10_000 + idx)
        env.close()
        returns = discounted_returns(ep.delayed_rewards, gamma=gamma)
        td, shared_gae = value_td_and_gae(value_critic, ep, gamma=gamma, gae_lam=gae_lam)

        with torch.no_grad():
            joint_obs_t = torch.as_tensor(ep.joint_obs, dtype=torch.float32)
            actions_t = torch.as_tensor(ep.actions, dtype=torch.long)
            q_pred = q_critic(joint_obs_t, actions_t).cpu().numpy()
            v_pred = value_critic(joint_obs_t).cpu().numpy()

        probs = np.full((ep.actions.shape[0], n_agents, n_actions), 1.0 / float(n_actions), dtype=np.float32)
        influence = critic_influence(q_critic, ep.joint_obs, ep.actions, probs)
        acct_credit = acct_advantages(influence, td, gamma=gamma, lam=0.90, residual_weight=0.5)
        spread = counterfactual_action_spread(q_critic, ep.joint_obs, ep.actions)

        q_targets.extend(returns.tolist())
        q_preds.extend(q_pred.tolist())
        v_targets.extend(returns.tolist())
        v_preds.extend(v_pred.tolist())
        abs_influence.extend(np.abs(influence).reshape(-1).tolist())
        action_spreads.extend(spread.reshape(-1).tolist())
        time_abs_influence.extend(np.sum(np.abs(influence), axis=1).tolist())
        time_abs_gae.extend(np.abs(shared_gae).tolist())
        signed_influence.extend(influence.reshape(-1).tolist())
        repeated_gae.extend(np.repeat(shared_gae[:, None], n_agents, axis=1).reshape(-1).tolist())
        acct_credit_values.extend(acct_credit.reshape(-1).tolist())

    q_targets_a = np.asarray(q_targets, dtype=np.float64)
    q_preds_a = np.asarray(q_preds, dtype=np.float64)
    v_targets_a = np.asarray(v_targets, dtype=np.float64)
    v_preds_a = np.asarray(v_preds, dtype=np.float64)
    abs_influence_a = np.asarray(abs_influence, dtype=np.float64)
    action_spreads_a = np.asarray(action_spreads, dtype=np.float64)

    return {
        "env": "delayed_simple_spread_mappo",
        "seed": seed,
        "episodes": episodes,
        "holdout_episodes": holdout,
        "q_mse": float(np.mean((q_preds_a - q_targets_a) ** 2)),
        "q_r2": r2_score(q_targets_a, q_preds_a),
        "v_mse": float(np.mean((v_preds_a - v_targets_a) ** 2)),
        "v_r2": r2_score(v_targets_a, v_preds_a),
        "mean_abs_influence": float(np.mean(abs_influence_a)),
        "p95_abs_influence": float(np.quantile(abs_influence_a, 0.95)),
        "mean_action_spread": float(np.mean(action_spreads_a)),
        "time_abs_influence_abs_gae_corr": corrcoef(np.asarray(time_abs_influence), np.asarray(time_abs_gae)),
        "signed_influence_gae_corr": corrcoef(np.asarray(signed_influence), np.asarray(repeated_gae)),
        "acct_credit_gae_corr": corrcoef(np.asarray(acct_credit_values), np.asarray(repeated_gae)),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    metrics = [
        "q_mse",
        "q_r2",
        "v_mse",
        "v_r2",
        "mean_abs_influence",
        "p95_abs_influence",
        "mean_action_spread",
        "time_abs_influence_abs_gae_corr",
        "signed_influence_gae_corr",
        "acct_credit_gae_corr",
    ]
    summary = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "seeds": len(values),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=16)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lam", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=Path("results/mpe_counterfactual_quality.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/mpe_counterfactual_quality_summary.csv"))
    args = parser.parse_args()

    rows = [
        evaluate_seed(
            seed=seed,
            episodes=args.episodes,
            holdout=args.holdout,
            max_cycles=args.max_cycles,
            batch_size=args.batch_size,
            ppo_epochs=args.ppo_epochs,
            gamma=args.gamma,
            gae_lam=args.gae_lam,
        )
        for seed in range(args.seeds)
    ]
    write_csv(args.out, rows)
    write_csv(args.summary_out, summarize(rows))
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
