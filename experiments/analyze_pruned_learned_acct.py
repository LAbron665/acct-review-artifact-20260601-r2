"""Run a local learned-pruned-ACCT diagnostic on the toy tasks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acct.envs import make_envs
from acct.learners import Method, run_training


METHODS: tuple[Method, ...] = (
    "learned_agent_time_cf",
    "learned_acct",
    "learned_pruned_acct",
)

COMPARISONS = (
    ("learned_pruned_acct", "learned_acct", "Pruned learned ACCT - learned ACCT"),
    ("learned_pruned_acct", "learned_agent_time_cf", "Pruned learned ACCT - learned AT-CF"),
)


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(rows: list[dict[str, float | int | str]], episode: int) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for env in sorted({str(row["env"]) for row in rows}):
        for method in METHODS:
            selected = [
                row for row in rows
                if row["env"] == env and row["method"] == method and int(row["episode"]) == episode
            ]
            sample = np.array([float(row["sample_return"]) for row in selected], dtype=np.float64)
            greedy = np.array([float(row["greedy_return"]) for row in selected], dtype=np.float64)
            summary.append(
                {
                    "env": env,
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
    for row in rows:
        if int(row["episode"]) == episode:
            values[(str(row["env"]), str(row["method"]), int(row["seed"]))] = float(row["sample_return"])

    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, str]] = []
    envs = sorted({env for env, _, _ in values})
    for env in envs:
        for left, right, label in COMPARISONS:
            seeds = sorted(
                seed for row_env, method, seed in values
                if row_env == env and method == left and (env, right, seed) in values
            )
            diffs = np.asarray([values[(env, left, s)] - values[(env, right, s)] for s in seeds])
            ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=bootstrap_samples)
            out_rows.append(
                {
                    "env": env,
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


def read_rows(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--episode", type=int, default=200)
    parser.add_argument("--critic-samples", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/pruned_learned_acct_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/pruned_learned_acct_summary.csv"))
    parser.add_argument("--stats", type=Path, default=Path("results/pruned_learned_acct_stat_tests.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260601)
    parser.add_argument("--reuse-existing", action="store_true", help="summarize the existing --out CSV without rerunning training")
    args = parser.parse_args()

    if args.reuse_existing:
        rows = read_rows(args.out)
    else:
        rows = []
        for env in make_envs():
            for method in METHODS:
                for seed in range(args.seeds):
                    rows.extend(
                        run_training(
                            env,
                            method=method,
                            seed=seed,
                            episodes=args.episodes,
                            critic_samples=args.critic_samples,
                        )
                    )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "env",
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

    print(f"wrote raw pruned learned ACCT diagnostics to {args.out}")
    print(f"wrote summary to {args.summary}")
    print(f"wrote paired statistics to {args.stats}")


if __name__ == "__main__":
    main()
