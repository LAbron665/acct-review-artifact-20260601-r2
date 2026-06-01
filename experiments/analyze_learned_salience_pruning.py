"""Diagnose whether dense learned salience still ranks relevant pairs highly."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acct.credit import counterfactual_influence, top_fraction_salience
from acct.critics import fit_quadratic_reward_model
from acct.envs import DelayedLeverEnv, PairGateEnv, make_envs
from acct.learners import TabularPolicy


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


def summarize_pruned_salience(
    env,
    fractions: list[float],
    seeds: int,
    samples: int,
    critic_samples: int,
) -> list[dict[str, str]]:
    mask = relevant_mask(env)
    irrelevant = ~mask
    rows: list[dict[str, str]] = []

    for fraction in fractions:
        rel_mass_values: list[float] = []
        rel_coverage_values: list[float] = []
        irr_active_values: list[float] = []
        kept_values: list[int] = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed + 123)
            policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=seed)
            reward_model = fit_quadratic_reward_model(env, seed=seed, samples=critic_samples)
            for _ in range(samples):
                actions = rng.integers(env.n_actions, size=(env.horizon, env.n_agents), dtype=np.int64)
                influence = counterfactual_influence(reward_model.predict, actions, policy.probs())
                salience = top_fraction_salience(np.abs(influence), fraction)
                total = float(salience.sum())
                rel_mass_values.append(float(salience[mask].sum() / total) if total > 1e-10 else 0.0)
                rel_coverage_values.append(float(np.mean(salience[mask] > 1e-8)))
                irr_active_values.append(float(np.mean(salience[irrelevant] > 1e-8)))
                kept_values.append(int(np.count_nonzero(salience)))

        rows.append(
            {
                "env": env.name,
                "top_fraction": f"{fraction:.2f}",
                "kept_pairs_mean": f"{np.mean(kept_values):.3f}",
                "relevant_mass": f"{np.mean(rel_mass_values):.4f}",
                "relevant_coverage": f"{np.mean(rel_coverage_values):.4f}",
                "irrelevant_active": f"{np.mean(irr_active_values):.4f}",
                "seeds": str(seeds),
                "samples_per_seed": str(samples),
                "critic_samples": str(critic_samples),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.50, 1.00])
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--critic-samples", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/learned_salience_pruning.csv"))
    args = parser.parse_args()

    rows = []
    for env in make_envs():
        rows.extend(
            summarize_pruned_salience(
                env,
                fractions=args.fractions,
                seeds=args.seeds,
                samples=args.samples,
                critic_samples=args.critic_samples,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "top_fraction",
                "kept_pairs_mean",
                "relevant_mass",
                "relevant_coverage",
                "irrelevant_active",
                "seeds",
                "samples_per_seed",
                "critic_samples",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote learned salience pruning diagnostics to {args.out}")


if __name__ == "__main__":
    main()
