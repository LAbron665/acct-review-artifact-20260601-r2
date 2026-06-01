"""Shape-safe ACCT adapter for EPyMARL-style PPO/actor-critic learners.

This module does not depend on EPyMARL internals.  It operates on the tensor
shapes used by EPyMARL learners:

- influence: ``[batch, time, agents]``
- td_residuals: ``[batch, time, agents]`` or ``[batch, time, 1]``
- mask: ``[batch, time, agents]`` or ``[batch, time, 1]``

The returned tensor has shape ``[batch, time, agents]`` and can replace or
augment the ``advantages`` tensor immediately before the actor loss in
``PPOLearner.train`` or ``ActorCriticLearner.train``.
"""

from __future__ import annotations

from typing import Literal

import torch


TransportMode = Literal["absolute", "directional"]
StandardizeMode = Literal["none", "masked"]


def expand_agent_dim(tensor: torch.Tensor, n_agents: int, name: str) -> torch.Tensor:
    """Expand an EPyMARL common-reward shaped tensor to per-agent shape."""

    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, N] or [B, T, 1]")
    if tensor.shape[-1] == n_agents:
        return tensor
    if tensor.shape[-1] == 1:
        return tensor.expand(-1, -1, n_agents)
    raise ValueError(f"{name} has incompatible agent dimension {tensor.shape[-1]}; expected 1 or {n_agents}")


def masked_standardize(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize credits over valid entries while preserving invalid zeros."""

    denom = mask.sum().clamp_min(1.0)
    mean = (values * mask).sum() / denom
    centered = (values - mean) * mask
    var = (centered.square().sum() / denom).clamp_min(eps)
    return centered / torch.sqrt(var)


def _transport_one(
    td_residuals: torch.Tensor,
    influence: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
    transport_mode: TransportMode,
    eps: float,
) -> torch.Tensor:
    """Transport one episode's residuals over the agent-time lattice."""

    time, n_agents = influence.shape
    credits = torch.zeros_like(influence)
    valid = mask > 0

    for k in range(time):
        residual = td_residuals[k]
        residual_mass = residual.mean()
        if residual_mass.abs() <= eps:
            continue

        eligible = valid[: k + 1]
        if transport_mode == "directional" and residual_mass > 0:
            salience = torch.clamp(influence[: k + 1], min=0.0)
        elif transport_mode == "directional" and residual_mass < 0:
            salience = torch.clamp(-influence[: k + 1], min=0.0)
        elif transport_mode == "absolute":
            salience = influence[: k + 1].abs()
        else:
            raise ValueError(f"unknown transport_mode: {transport_mode}")

        salience = salience * eligible
        if salience.sum() <= eps:
            salience = influence[: k + 1].abs() * eligible
        if salience.sum() <= eps:
            salience = eligible.to(influence.dtype)

        decay = torch.tensor(
            [(gamma * lam) ** (k - t) for t in range(k + 1)],
            dtype=influence.dtype,
            device=influence.device,
        ).view(k + 1, 1)
        weights = salience * decay
        denom = weights.sum()
        if denom <= eps:
            continue
        credits[: k + 1] += residual_mass * weights / denom

    return credits * mask


def acct_epymarl_advantages(
    influence: torch.Tensor,
    td_residuals: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.90,
    residual_weight: float = 0.5,
    transport_mode: TransportMode = "absolute",
    standardize: StandardizeMode = "masked",
) -> torch.Tensor:
    """Return EPyMARL-shaped ACCT actor advantages.

    ``influence`` is the signed counterfactual influence for each agent-time
    pair.  ``td_residuals`` can be a common residual with final dimension one or
    an already expanded per-agent residual tensor.  The transport component is
    computed per episode and conserves the masked residual mass.
    """

    if influence.ndim != 3:
        raise ValueError("influence must have shape [B, T, N]")
    n_agents = influence.shape[-1]
    td_residuals = expand_agent_dim(td_residuals.to(influence.dtype), n_agents, "td_residuals")
    mask = expand_agent_dim(mask.to(influence.dtype), n_agents, "mask")
    if td_residuals.shape[:2] != influence.shape[:2] or mask.shape[:2] != influence.shape[:2]:
        raise ValueError("influence, td_residuals, and mask disagree on [B, T]")

    transported = torch.stack(
        [
            _transport_one(
                td_residuals[b],
                influence[b],
                mask[b],
                gamma=gamma,
                lam=lam,
                transport_mode=transport_mode,
                eps=1e-8,
            )
            for b in range(influence.shape[0])
        ],
        dim=0,
    )
    credits = (influence + residual_weight * transported) * mask
    if standardize == "masked":
        return masked_standardize(credits, mask)
    if standardize == "none":
        return credits
    raise ValueError(f"unknown standardize mode: {standardize}")


def counterfactual_influence_from_replacements(
    q_taken: torch.Tensor,
    q_replacements: torch.Tensor,
    action_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute influence from a counterfactual Q replacement head.

    Args:
        q_taken: value of the actually executed joint action, shape
            ``[batch, time, agents]`` or ``[batch, time, 1]``.
        q_replacements: replacement values for each agent action,
            shape ``[batch, time, agents, actions]``.
        action_probs: decentralized action probabilities with the same last
            two dimensions as ``q_replacements``.
    """

    if q_replacements.shape != action_probs.shape:
        raise ValueError("q_replacements and action_probs must have the same shape")
    if q_replacements.ndim != 4:
        raise ValueError("q_replacements must have shape [B, T, N, A]")
    n_agents = q_replacements.shape[-2]
    q_taken = expand_agent_dim(q_taken.to(q_replacements.dtype), n_agents, "q_taken")
    expected = (q_replacements * action_probs).sum(dim=-1)
    return q_taken - expected
