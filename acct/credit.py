"""Credit assignment utilities for ACCT.

The code is deliberately framework-agnostic.  A PyMARL/EPyMARL/MAPPO
integration can use the same tensors before the actor update: compute a
counterfactual influence table, transport TD residuals backward over agent-time
pairs, and feed the resulting credits as per-agent advantages or auxiliary value
targets.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


Array = np.ndarray
TransportMode = Literal["absolute", "directional"]


def counterfactual_influence(
    reward_fn: Callable[[Array], float],
    actions: Array,
    action_probs: Array,
) -> Array:
    """Compute per-agent per-time counterfactual influence.

    Args:
        reward_fn: Callable mapping an integer action matrix ``[T, N]`` to the
            scalar team reward or a critic prediction.
        actions: Integer action matrix with shape ``[T, N]``.
        action_probs: Behavior probabilities with shape ``[T, N, A]``.

    Returns:
        Influence matrix ``I[t, i] = R(a) - E_{a_i'~pi_i} R(a_{-i}, a_i')``.
    """

    actions = np.asarray(actions, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError("actions must have shape [T, N]")
    if action_probs.ndim != 3:
        raise ValueError("action_probs must have shape [T, N, A]")
    if action_probs.shape[:2] != actions.shape:
        raise ValueError("action_probs and actions disagree on [T, N]")

    horizon, n_agents = actions.shape
    n_actions = action_probs.shape[-1]
    actual_reward = float(reward_fn(actions))
    influence = np.zeros((horizon, n_agents), dtype=np.float64)

    for t in range(horizon):
        for i in range(n_agents):
            expected = 0.0
            for a in range(n_actions):
                replaced = actions.copy()
                replaced[t, i] = a
                expected += action_probs[t, i, a] * float(reward_fn(replaced))
            influence[t, i] = actual_reward - expected
    return influence


def sampled_agent_time_shapley(
    reward_fn: Callable[[Array], float],
    actions: Array,
    action_probs: Array,
    samples: int = 16,
    rng: np.random.Generator | None = None,
) -> Array:
    """Estimate Shapley-style marginal credit over agent-time decision units.

    Each sample draws a reference trajectory from the behavior policy, then
    reveals the actual trajectory along a random permutation of agent-time
    units. The marginal reward increments are assigned to the revealed units.
    This is a diagnostic baseline inspired by Shapley credit assignment; it is
    intentionally local and much cheaper than a full learned STAS-style model.
    """

    actions = np.asarray(actions, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError("actions must have shape [T, N]")
    if action_probs.ndim != 3:
        raise ValueError("action_probs must have shape [T, N, A]")
    if action_probs.shape[:2] != actions.shape:
        raise ValueError("action_probs and actions disagree on [T, N]")
    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng() if rng is None else rng
    horizon, n_agents = actions.shape
    n_actions = action_probs.shape[-1]
    units = np.arange(horizon * n_agents)
    credits = np.zeros((horizon, n_agents), dtype=np.float64)

    for _ in range(samples):
        reference = np.zeros_like(actions)
        for t in range(horizon):
            for i in range(n_agents):
                reference[t, i] = rng.choice(n_actions, p=action_probs[t, i])

        current = reference.copy()
        current_reward = float(reward_fn(current))
        for unit in rng.permutation(units):
            t, i = divmod(int(unit), n_agents)
            if current[t, i] == actions[t, i]:
                continue
            previous_reward = current_reward
            current[t, i] = actions[t, i]
            current_reward = float(reward_fn(current))
            credits[t, i] += current_reward - previous_reward

    return credits / float(samples)


def temporal_credit_transport(
    td_residuals: Array,
    salience: Array,
    gamma: float = 0.99,
    lam: float = 0.95,
    eps: float = 1e-8,
) -> Array:
    """Transport global TD residuals backward over agent-time pairs.

    For each residual at time ``k``, ACCT distributes the signed residual over
    all earlier pairs ``(t, i), t <= k`` using nonnegative counterfactual
    salience and an eligibility decay.  The residual transport component is
    conservative: its entries sum to the sum of the TD residuals.
    """

    td_residuals = np.asarray(td_residuals, dtype=np.float64)
    salience = np.asarray(salience, dtype=np.float64)
    if td_residuals.ndim != 1:
        raise ValueError("td_residuals must have shape [T]")
    if salience.ndim != 2:
        raise ValueError("salience must have shape [T, N]")
    if salience.shape[0] != td_residuals.shape[0]:
        raise ValueError("td_residuals and salience disagree on horizon")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError("gamma and lam must lie in [0, 1]")

    horizon, n_agents = salience.shape
    credits = np.zeros((horizon, n_agents), dtype=np.float64)
    nonnegative_salience = np.maximum(salience, 0.0)

    for k, delta in enumerate(td_residuals):
        if abs(delta) <= eps:
            continue
        weights = np.zeros((k + 1, n_agents), dtype=np.float64)
        for t in range(k + 1):
            weights[t] = ((gamma * lam) ** (k - t)) * nonnegative_salience[t]
        denom = float(weights.sum())
        if denom <= eps:
            weights.fill(1.0)
            denom = float(weights.sum())
        credits[: k + 1] += delta * weights / denom
    return credits


def return_equivalent_transport(
    team_return: float,
    salience: Array,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Array:
    """Redistribute one observed team return over agent-time salience.

    This is a small local diagnostic proxy for return-equivalent
    agent-temporal redistribution: the generated credit entries sum to the
    observed team return, without adding ACCT's instantaneous signed
    counterfactual influence term.
    """

    salience = np.asarray(salience, dtype=np.float64)
    if salience.ndim != 2:
        raise ValueError("salience must have shape [T, N]")
    residuals = np.zeros(salience.shape[0], dtype=np.float64)
    residuals[-1] = float(team_return)
    return temporal_credit_transport(residuals, salience=salience, gamma=gamma, lam=lam)


def top_fraction_salience(salience: Array, fraction: float) -> Array:
    """Keep only the largest salience entries by absolute magnitude.

    This diagnostic helper probes whether a learned counterfactual head ranks
    reward-relevant agent-time pairs ahead of distractors even when its raw
    influence is dense. It preserves the original entry values for the retained
    pairs and zeros the rest.
    """

    salience = np.asarray(salience, dtype=np.float64)
    if salience.ndim != 2:
        raise ValueError("salience must have shape [T, N]")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must lie in (0, 1]")
    if fraction >= 1.0:
        return salience.copy()

    flat = salience.reshape(-1)
    keep = max(1, int(np.ceil(fraction * flat.size)))
    indices = np.argpartition(np.abs(flat), -keep)[-keep:]
    pruned = np.zeros_like(flat)
    pruned[indices] = flat[indices]
    return pruned.reshape(salience.shape)


def directional_temporal_credit_transport(
    td_residuals: Array,
    influence: Array,
    gamma: float = 0.99,
    lam: float = 0.95,
    eps: float = 1e-8,
) -> Array:
    """Transport residuals only through sign-compatible influence when possible.

    Positive residuals are assigned to earlier positive counterfactual
    influences, while negative residuals are assigned to earlier negative
    influences. If no sign-compatible influence exists for a residual, the
    rule falls back to absolute salience and then to uniform transport. The
    fallback preserves the same conservation guarantee as
    ``temporal_credit_transport`` while avoiding the common case where a
    harmful action receives positive transported reward.
    """

    td_residuals = np.asarray(td_residuals, dtype=np.float64)
    influence = np.asarray(influence, dtype=np.float64)
    if td_residuals.ndim != 1:
        raise ValueError("td_residuals must have shape [T]")
    if influence.ndim != 2:
        raise ValueError("influence must have shape [T, N]")
    if influence.shape[0] != td_residuals.shape[0]:
        raise ValueError("td_residuals and influence disagree on horizon")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError("gamma and lam must lie in [0, 1]")

    horizon, n_agents = influence.shape
    credits = np.zeros((horizon, n_agents), dtype=np.float64)

    for k, delta in enumerate(td_residuals):
        if abs(delta) <= eps:
            continue
        if delta > 0:
            salience = np.maximum(influence[: k + 1], 0.0)
        else:
            salience = np.maximum(-influence[: k + 1], 0.0)
        if float(salience.sum()) <= eps:
            salience = np.abs(influence[: k + 1])

        weights = np.zeros((k + 1, n_agents), dtype=np.float64)
        for t in range(k + 1):
            weights[t] = ((gamma * lam) ** (k - t)) * salience[t]
        denom = float(weights.sum())
        if denom <= eps:
            weights.fill(1.0)
            denom = float(weights.sum())
        credits[: k + 1] += delta * weights / denom
    return credits


def acct_advantages(
    influence: Array,
    td_residuals: Array,
    gamma: float = 0.99,
    lam: float = 0.95,
    residual_weight: float = 1.0,
    transport_mode: TransportMode = "absolute",
) -> Array:
    """Combine instantaneous counterfactual influence and residual transport."""

    influence = np.asarray(influence, dtype=np.float64)
    if transport_mode == "absolute":
        transported = temporal_credit_transport(
            td_residuals,
            salience=np.abs(influence),
            gamma=gamma,
            lam=lam,
        )
    elif transport_mode == "directional":
        transported = directional_temporal_credit_transport(
            td_residuals,
            influence=influence,
            gamma=gamma,
            lam=lam,
        )
    else:
        raise ValueError(f"unknown transport_mode: {transport_mode}")
    return influence + residual_weight * transported
