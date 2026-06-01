"""Influence perturbation diagnostics for ACCT.

This script tests a narrow robustness question: when a counterfactual head gets
the relative agent-time pattern mostly right but its magnitude or noise changes,
does residual transport behave differently from raw agent-time counterfactual
credit? The diagnostic is intentionally local and uses the same small tasks as
the main tabular experiments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acct.credit import acct_advantages, counterfactual_influence
from acct.envs import make_envs
from acct.learners import TabularPolicy, evaluate


@dataclass(frozen=True)
class Perturbation:
    name: str
    scale: float
    noise_std: float


PERTURBATIONS = (
    Perturbation("scale_0.25", 0.25, 0.0),
    Perturbation("scale_1.00", 1.0, 0.0),
    Perturbation("scale_4.00", 4.0, 0.0),
    Perturbation("noise_0.10", 1.0, 0.10),
    Perturbation("noise_0.25", 1.0, 0.25),
)

METHODS = ("perturbed_agent_time_cf", "perturbed_acct")


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def perturbed_influence(
    env,
    policy: TabularPolicy,
    actions: np.ndarray,
    perturbation: Perturbation,
    rng: np.random.Generator,
) -> np.ndarray:
    influence = counterfactual_influence(env.reward, actions, policy.probs())
    if perturbation.noise_std > 0.0:
        influence = influence + rng.normal(0.0, perturbation.noise_std, size=influence.shape)
    return perturbation.scale * influence


def run_training(
    env,
    method: str,
    perturbation: Perturbation,
    seed: int,
    episodes: int,
    lr: float,
    eval_every: int,
    residual_weight: float,
) -> list[dict[str, float | int | str]]:
    policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=seed)
    stable_offset = sum(ord(ch) for ch in f"{env.name}:{method}:{perturbation.name}")
    noise_rng = np.random.default_rng(10_000_000 + 1_003 * seed + stable_offset)
    baseline = 0.0
    rows: list[dict[str, float | int | str]] = []
    for episode in range(1, episodes + 1):
        actions = policy.sample()
        reward = float(env.reward(actions))
        baseline = 0.98 * baseline + 0.02 * reward
        influence = perturbed_influence(env, policy, actions, perturbation, noise_rng)
        if method == "perturbed_agent_time_cf":
            credits = influence
        elif method == "perturbed_acct":
            td_residuals = np.zeros(env.horizon, dtype=np.float64)
            td_residuals[-1] = reward - baseline
            credits = acct_advantages(
                influence,
                td_residuals,
                gamma=0.99,
                lam=0.90,
                residual_weight=residual_weight,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        policy.update(actions, credits, lr=lr)

        if episode % eval_every == 0 or episode == 1:
            sample_return, greedy_return = evaluate(policy, env.reward, episodes=64)
            rows.append(
                {
                    "env": env.name,
                    "perturbation": perturbation.name,
                    "method": method,
                    "seed": seed,
                    "episode": episode,
                    "sample_return": sample_return,
                    "greedy_return": greedy_return,
                    "baseline": baseline,
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str]], final_episode: int, comparison_episode: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    values: dict[tuple[str, str, str, int, int], tuple[float, float]] = {}
    for row in rows:
        values[
            (
                str(row["env"]),
                str(row["perturbation"]),
                str(row["method"]),
                int(row["seed"]),
                int(row["episode"]),
            )
        ] = (float(row["sample_return"]), float(row["greedy_return"]))

    summary_rows = []
    keys = sorted({(env, pert, method) for env, pert, method, _, _ in values})
    for env, perturbation, method in keys:
        seeds = sorted({seed for e, p, m, seed, ep in values if e == env and p == perturbation and m == method and ep == final_episode})
        sample_200 = np.asarray([values[(env, perturbation, method, seed, comparison_episode)][0] for seed in seeds], dtype=np.float64)
        greedy_final = np.asarray([values[(env, perturbation, method, seed, final_episode)][1] for seed in seeds], dtype=np.float64)
        summary_rows.append(
            {
                "env": env,
                "perturbation": perturbation,
                "method": method,
                "sample_200_mean": f"{sample_200.mean():.4f}",
                "sample_200_std": f"{sample_200.std(ddof=0):.4f}",
                "greedy_final_mean": f"{greedy_final.mean():.4f}",
                "greedy_final_std": f"{greedy_final.std(ddof=0):.4f}",
                "seeds": str(len(seeds)),
            }
        )

    rng = np.random.default_rng(20260601)
    stat_rows = []
    for env, perturbation in sorted({(env, pert) for env, pert, _, _, _ in values}):
        seeds = sorted(
            seed
            for e, p, m, seed, ep in values
            if e == env and p == perturbation and m == "perturbed_acct" and ep == comparison_episode
            and (env, perturbation, "perturbed_agent_time_cf", seed, comparison_episode) in values
        )
        diffs = np.asarray(
            [
                values[(env, perturbation, "perturbed_acct", seed, comparison_episode)][0]
                - values[(env, perturbation, "perturbed_agent_time_cf", seed, comparison_episode)][0]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=10_000)
        stat_rows.append(
            {
                "env": env,
                "perturbation": perturbation,
                "episode": str(comparison_episode),
                "comparison": "perturbed ACCT - perturbed AT-CF",
                "mean_diff": f"{diffs.mean():.4f}",
                "ci95_low": f"{ci_low:.4f}",
                "ci95_high": f"{ci_high:.4f}",
                "win_rate": f"{np.mean(diffs > 0):.4f}",
                "paired_seeds": str(len(seeds)),
            }
        )
    return summary_rows, stat_rows


def write_csv(path: Path, rows: list[dict[str, str | float | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--residual-weight", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=Path("results/influence_perturbation.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/influence_perturbation_summary.csv"))
    parser.add_argument("--stats-out", type=Path, default=Path("results/influence_perturbation_stat_tests.csv"))
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for env in make_envs():
        for perturbation in PERTURBATIONS:
            for method in METHODS:
                for seed in range(args.seeds):
                    rows.extend(
                        run_training(
                            env,
                            method=method,
                            perturbation=perturbation,
                            seed=seed,
                            episodes=args.episodes,
                            lr=args.lr,
                            eval_every=args.eval_every,
                            residual_weight=args.residual_weight,
                        )
                    )

    raw_fields = ["env", "perturbation", "method", "seed", "episode", "sample_return", "greedy_return", "baseline"]
    write_csv(args.out, rows, raw_fields)
    summary_rows, stat_rows = summarize(rows, final_episode=args.episodes, comparison_episode=200)
    summary_fields = ["env", "perturbation", "method", "sample_200_mean", "sample_200_std", "greedy_final_mean", "greedy_final_std", "seeds"]
    stat_fields = ["env", "perturbation", "episode", "comparison", "mean_diff", "ci95_low", "ci95_high", "win_rate", "paired_seeds"]
    write_csv(args.summary_out, summary_rows, summary_fields)
    write_csv(args.stats_out, stat_rows, stat_fields)
    print(f"wrote raw perturbation diagnostics to {args.out}")
    print(f"wrote perturbation summary to {args.summary_out}")
    print(f"wrote perturbation statistics to {args.stats_out}")


if __name__ == "__main__":
    main()
