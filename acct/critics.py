"""Lightweight reward critics for local ACCT diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass
class QuadraticRewardModel:
    """Ridge-fitted quadratic reward predictor over binary action trajectories."""

    horizon: int
    n_agents: int
    weights: Array
    pair_i: Array
    pair_j: Array

    def features(self, actions: Array) -> Array:
        bits = np.asarray(actions, dtype=np.float64).reshape(-1)
        if bits.shape[0] != self.horizon * self.n_agents:
            raise ValueError("actions have wrong flattened size")
        pairs = bits[self.pair_i] * bits[self.pair_j]
        return np.concatenate(([1.0], bits, pairs))

    def predict(self, actions: Array) -> float:
        return float(self.features(actions) @ self.weights)


def fit_quadratic_reward_model(env, seed: int, samples: int = 2048, ridge: float = 1e-3) -> QuadraticRewardModel:
    """Fit a small reward model from uniformly sampled trajectories."""

    rng = np.random.default_rng(seed + 10_000)
    n_bits = env.horizon * env.n_agents
    pair_i, pair_j = np.triu_indices(n_bits, k=1)
    feature_dim = 1 + n_bits + len(pair_i)
    design = np.zeros((samples, feature_dim), dtype=np.float64)
    targets = np.zeros(samples, dtype=np.float64)

    probe = QuadraticRewardModel(env.horizon, env.n_agents, np.zeros(feature_dim), pair_i, pair_j)
    for row in range(samples):
        actions = rng.integers(env.n_actions, size=(env.horizon, env.n_agents), dtype=np.int64)
        design[row] = probe.features(actions)
        targets[row] = float(env.reward(actions))

    regularizer = np.sqrt(ridge) * np.eye(feature_dim, dtype=np.float64)
    augmented_design = np.vstack([design, regularizer])
    augmented_targets = np.concatenate([targets, np.zeros(feature_dim, dtype=np.float64)])
    weights, *_ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)
    return QuadraticRewardModel(env.horizon, env.n_agents, weights, pair_i, pair_j)
