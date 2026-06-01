"""Analyze whether credit mass lands on reward-relevant agent-time pairs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acct.envs import DelayedLeverEnv, PairGateEnv, make_envs
from acct.learners import Method, TabularPolicy, _credits_for_method
from acct.critics import fit_quadratic_reward_model


METHODS: tuple[Method, ...] = (
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
)


def relevant_mask(env) -> np.ndarray:
    mask = np.zeros((env.horizon, env.n_agents), dtype=bool)
    if isinstance(env, DelayedLeverEnv):
        for t, i, _ in env.important:
            mask[t, i] = True
    elif isinstance(env, PairGateEnv):
        for gate in env.gates:
            for t, i, _ in gate:
                mask[t, i] = True
    else:
        raise TypeError(f"unknown env type: {type(env)!r}")
    return mask


def analyze_env(env, seeds: int, samples: int, critic_samples: int) -> list[dict[str, str]]:
    mask = relevant_mask(env)
    irrelevant = ~mask
    rows = []
    for method in METHODS:
        rel_mass_values = []
        rel_coverage_values = []
        irr_active_values = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed + 123)
            policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=seed)
            critic_reward_fn = None
            if method.startswith("learned_"):
                critic_reward_fn = fit_quadratic_reward_model(env, seed=seed, samples=critic_samples).predict
            baseline = 0.0
            for _ in range(samples):
                actions = rng.integers(env.n_actions, size=(env.horizon, env.n_agents), dtype=np.int64)
                reward = float(env.reward(actions))
                baseline = 0.95 * baseline + 0.05 * reward
                credits = _credits_for_method(
                    env,
                    policy,
                    actions,
                    reward,
                    baseline,
                    method=method,
                    critic_reward_fn=critic_reward_fn,
                )
                abs_credit = np.abs(credits)
                total = float(abs_credit.sum())
                rel_mass_values.append(float(abs_credit[mask].sum() / total) if total > 1e-10 else 0.0)
                rel_coverage_values.append(float(np.mean(abs_credit[mask] > 1e-8)))
                irr_active_values.append(float(np.mean(abs_credit[irrelevant] > 1e-8)))
        rows.append(
            {
                "env": env.name,
                "method": method,
                "relevant_mass": f"{np.mean(rel_mass_values):.4f}",
                "relevant_coverage": f"{np.mean(rel_coverage_values):.4f}",
                "irrelevant_active": f"{np.mean(irr_active_values):.4f}",
                "seeds": str(seeds),
                "samples_per_seed": str(samples),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--critic-samples", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/credit_localization.csv"))
    args = parser.parse_args()

    rows = []
    for env in make_envs():
        rows.extend(analyze_env(env, seeds=args.seeds, samples=args.samples, critic_samples=args.critic_samples))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "method",
                "relevant_mass",
                "relevant_coverage",
                "irrelevant_active",
                "seeds",
                "samples_per_seed",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote localization metrics to {args.out}")


if __name__ == "__main__":
    main()
