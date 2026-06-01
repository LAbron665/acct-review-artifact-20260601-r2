"""Stress local credit methods as irrelevant agent-time decisions increase."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acct.envs import DelayedLeverEnv
from acct.learners import Method, run_training


METHODS: tuple[Method, ...] = (
    "shared",
    "uniform_transport",
    "agent_time_cf",
    "acct",
)

COMPARISONS = (
    ("acct", "shared", "ACCT - shared"),
    ("acct", "uniform_transport", "ACCT - uniform transport"),
    ("acct", "agent_time_cf", "ACCT - agent-time CF"),
)


def make_stress_envs() -> tuple[DelayedLeverEnv, ...]:
    important = (
        (0, 0, 1),
        (1, 1, 1),
        (2, 2, 0),
        (3, 3, 1),
    )
    configs = (
        ("lever_20_pairs", 5, 4),
        ("lever_32_pairs", 8, 4),
        ("lever_48_pairs", 12, 4),
        ("lever_72_pairs", 12, 6),
    )
    return tuple(
        DelayedLeverEnv(horizon=horizon, n_agents=n_agents, important=important, name=name)
        for name, horizon, n_agents in configs
    )


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(rows: list[dict[str, float | int | str]], episode: int) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for env in sorted({str(row["env"]) for row in rows}):
        env_rows = [row for row in rows if row["env"] == env]
        total_pairs = str(env_rows[0]["total_pairs"])
        relevant_pairs = str(env_rows[0]["relevant_pairs"])
        irrelevant_pairs = str(env_rows[0]["irrelevant_pairs"])
        for method in METHODS:
            selected = [
                row for row in env_rows
                if row["method"] == method and int(row["episode"]) == episode
            ]
            sample = np.asarray([float(row["sample_return"]) for row in selected], dtype=np.float64)
            greedy = np.asarray([float(row["greedy_return"]) for row in selected], dtype=np.float64)
            summary.append(
                {
                    "env": env,
                    "total_pairs": total_pairs,
                    "relevant_pairs": relevant_pairs,
                    "irrelevant_pairs": irrelevant_pairs,
                    "method": method,
                    "episode": str(episode),
                    "sample_mean": f"{sample.mean():.4f}",
                    "sample_std": f"{sample.std(ddof=1):.4f}",
                    "greedy_mean": f"{greedy.mean():.4f}",
                    "greedy_std": f"{greedy.std(ddof=1):.4f}",
                    "seeds": str(len(selected)),
                }
            )
    return summary


def paired_stats(
    rows: list[dict[str, float | int | str]],
    episode: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, str]]:
    values: dict[tuple[str, str, int], float] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        env = str(row["env"])
        metadata[env] = (str(row["total_pairs"]), str(row["relevant_pairs"]), str(row["irrelevant_pairs"]))
        if int(row["episode"]) == episode:
            values[(env, str(row["method"]), int(row["seed"]))] = float(row["sample_return"])

    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, str]] = []
    for env in sorted(metadata):
        total_pairs, relevant_pairs, irrelevant_pairs = metadata[env]
        for left, right, label in COMPARISONS:
            seeds = sorted(
                run_seed for row_env, method, run_seed in values
                if row_env == env and method == left and (env, right, run_seed) in values
            )
            diffs = np.asarray([values[(env, left, run_seed)] - values[(env, right, run_seed)] for run_seed in seeds])
            ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=bootstrap_samples)
            out_rows.append(
                {
                    "env": env,
                    "total_pairs": total_pairs,
                    "relevant_pairs": relevant_pairs,
                    "irrelevant_pairs": irrelevant_pairs,
                    "episode": str(episode),
                    "comparison": label,
                    "mean_diff": f"{diffs.mean():.4f}",
                    "ci95_low": f"{ci_low:.4f}",
                    "ci95_high": f"{ci_high:.4f}",
                    "win_rate": f"{np.mean(diffs > 0):.4f}",
                    "paired_seeds": str(len(seeds)),
                }
            )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--episode", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/distractor_scaling_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/distractor_scaling_summary.csv"))
    parser.add_argument("--stats", type=Path, default=Path("results/distractor_scaling_stat_tests.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260601)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for env in make_stress_envs():
        total_pairs = env.horizon * env.n_agents
        relevant_pairs = len(env.important)
        irrelevant_pairs = total_pairs - relevant_pairs
        for method in METHODS:
            for seed in range(args.seeds):
                run_rows = run_training(env, method=method, seed=seed, episodes=args.episodes)
                for row in run_rows:
                    row["total_pairs"] = total_pairs
                    row["relevant_pairs"] = relevant_pairs
                    row["irrelevant_pairs"] = irrelevant_pairs
                rows.extend(run_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "total_pairs",
                "relevant_pairs",
                "irrelevant_pairs",
                "method",
                "seed",
                "episode",
                "sample_return",
                "greedy_return",
                "baseline",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows, episode=args.episode)
    with args.summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "total_pairs",
                "relevant_pairs",
                "irrelevant_pairs",
                "method",
                "episode",
                "sample_mean",
                "sample_std",
                "greedy_mean",
                "greedy_std",
                "seeds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    stats_rows = paired_stats(
        rows,
        episode=args.episode,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    with args.stats.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "total_pairs",
                "relevant_pairs",
                "irrelevant_pairs",
                "episode",
                "comparison",
                "mean_diff",
                "ci95_low",
                "ci95_high",
                "win_rate",
                "paired_seeds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"wrote distractor-scaling results to {args.out}")
    print(f"wrote distractor-scaling summary to {args.summary}")
    print(f"wrote distractor-scaling statistics to {args.stats}")


if __name__ == "__main__":
    main()
