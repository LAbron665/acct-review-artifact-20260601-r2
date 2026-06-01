"""Tabular policy-gradient harness for fast local ACCT diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .credit import (
    acct_advantages,
    counterfactual_influence,
    return_equivalent_transport,
    sampled_agent_time_shapley,
    temporal_credit_transport,
    top_fraction_salience,
)
from .critics import fit_quadratic_reward_model


Method = Literal[
    "shared",
    "final_step_cf",
    "uniform_transport",
    "cf_transport",
    "return_eq_cf_transport",
    "agent_time_cf",
    "sampled_agent_time_shapley",
    "acct",
    "directional_acct",
    "learned_cf_transport",
    "learned_return_eq_cf_transport",
    "learned_agent_time_cf",
    "learned_acct",
    "learned_directional_acct",
    "learned_pruned_acct",
]


@dataclass
class TabularPolicy:
    horizon: int
    n_agents: int
    n_actions: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.logits = np.zeros((self.horizon, self.n_agents, self.n_actions), dtype=np.float64)

    def probs(self) -> np.ndarray:
        shifted = self.logits - self.logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def sample(self) -> np.ndarray:
        probs = self.probs()
        actions = np.zeros((self.horizon, self.n_agents), dtype=np.int64)
        for t in range(self.horizon):
            for i in range(self.n_agents):
                actions[t, i] = self.rng.choice(self.n_actions, p=probs[t, i])
        return actions

    def greedy_actions(self) -> np.ndarray:
        return self.probs().argmax(axis=-1)

    def update(self, actions: np.ndarray, credits: np.ndarray, lr: float) -> None:
        probs = self.probs()
        if credits.shape != actions.shape:
            raise ValueError("credits must match action matrix shape")
        for t in range(self.horizon):
            for i in range(self.n_agents):
                grad = -probs[t, i].copy()
                grad[actions[t, i]] += 1.0
                self.logits[t, i] += lr * credits[t, i] * grad
        np.clip(self.logits, -12.0, 12.0, out=self.logits)


def _credits_for_method(
    env,
    policy: TabularPolicy,
    actions: np.ndarray,
    reward: float,
    baseline: float,
    method: Method,
    critic_reward_fn=None,
    transport_gamma: float = 0.99,
    transport_lam: float = 0.90,
    residual_weight: float = 0.75,
    salience_fraction: float = 0.20,
) -> np.ndarray:
    advantage = reward - baseline
    if method == "shared":
        return np.full(actions.shape, advantage, dtype=np.float64)
    if method == "sampled_agent_time_shapley":
        return sampled_agent_time_shapley(env.reward, actions, policy.probs(), samples=16, rng=policy.rng)
    if method == "uniform_transport":
        td_residuals = np.zeros(env.horizon, dtype=np.float64)
        td_residuals[-1] = advantage
        salience = np.ones(actions.shape, dtype=np.float64)
        return temporal_credit_transport(td_residuals, salience, gamma=transport_gamma, lam=transport_lam)

    reward_fn = critic_reward_fn if method.startswith("learned_") else env.reward
    if reward_fn is None:
        raise ValueError(f"{method} requires a critic_reward_fn")
    influence = counterfactual_influence(reward_fn, actions, policy.probs())
    if method == "learned_pruned_acct":
        influence = top_fraction_salience(influence, salience_fraction)
    if method == "final_step_cf":
        final_step = np.zeros_like(influence)
        final_step[-1] = influence[-1]
        return final_step
    if method in ("cf_transport", "learned_cf_transport"):
        td_residuals = np.zeros(env.horizon, dtype=np.float64)
        td_residuals[-1] = advantage
        return temporal_credit_transport(
            td_residuals,
            salience=np.abs(influence),
            gamma=transport_gamma,
            lam=transport_lam,
        )
    if method in ("return_eq_cf_transport", "learned_return_eq_cf_transport"):
        return return_equivalent_transport(
            reward,
            salience=np.abs(influence),
            gamma=transport_gamma,
            lam=transport_lam,
        )
    if method in ("agent_time_cf", "learned_agent_time_cf"):
        return influence
    if method in ("acct", "learned_acct", "directional_acct", "learned_directional_acct", "learned_pruned_acct"):
        td_residuals = np.zeros(env.horizon, dtype=np.float64)
        td_residuals[-1] = advantage
        return acct_advantages(
            influence,
            td_residuals,
            gamma=transport_gamma,
            lam=transport_lam,
            residual_weight=residual_weight,
            transport_mode="directional" if "directional" in method else "absolute",
        )
    raise ValueError(f"unknown method: {method}")


def evaluate(policy: TabularPolicy, reward_fn, episodes: int = 128) -> tuple[float, float]:
    rewards = []
    for _ in range(episodes):
        rewards.append(float(reward_fn(policy.sample())))
    greedy = float(reward_fn(policy.greedy_actions()))
    return float(np.mean(rewards)), greedy


def run_training(
    env,
    method: Method,
    seed: int,
    episodes: int = 800,
    lr: float = 0.08,
    eval_every: int = 20,
    critic_samples: int = 2048,
    salience_fraction: float = 0.20,
) -> list[dict[str, float | int | str]]:
    policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=seed)
    critic_reward_fn = None
    if method.startswith("learned_"):
        critic_reward_fn = fit_quadratic_reward_model(env, seed=seed, samples=critic_samples).predict
    baseline = 0.0
    rows: list[dict[str, float | int | str]] = []

    for episode in range(1, episodes + 1):
        actions = policy.sample()
        reward = float(env.reward(actions))
        baseline = 0.98 * baseline + 0.02 * reward
        credits = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline,
            method,
            critic_reward_fn=critic_reward_fn,
            salience_fraction=salience_fraction,
        )
        policy.update(actions, credits, lr=lr)

        if episode % eval_every == 0 or episode == 1:
            sample_return, greedy_return = evaluate(policy, env.reward, episodes=64)
            rows.append(
                {
                    "env": env.name,
                    "method": method,
                    "seed": seed,
                    "episode": episode,
                    "sample_return": sample_return,
                    "greedy_return": greedy_return,
                    "baseline": baseline,
                }
            )
    return rows
