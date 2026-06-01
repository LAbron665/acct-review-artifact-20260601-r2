"""Reference implementation for Agent-Time Counterfactual Credit Transport."""

from .credit import acct_advantages, counterfactual_influence, temporal_credit_transport
from .critics import QuadraticRewardModel, fit_quadratic_reward_model
from .envs import DelayedLeverEnv, PairGateEnv
from .learners import TabularPolicy, run_training

__all__ = [
    "acct_advantages",
    "counterfactual_influence",
    "temporal_credit_transport",
    "QuadraticRewardModel",
    "fit_quadratic_reward_model",
    "DelayedLeverEnv",
    "PairGateEnv",
    "TabularPolicy",
    "run_training",
]
