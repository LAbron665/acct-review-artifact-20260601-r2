"""Full-curve sample-efficiency summaries for local ACCT diagnostics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("acct", "uniform_transport", "ACCT - uniform transport"),
    ("acct", "cf_transport", "ACCT - CF transport-only"),
    ("acct", "return_eq_cf_transport", "ACCT - return-eq CF transport"),
    ("acct", "agent_time_cf", "ACCT - agent-time CF"),
    ("acct", "sampled_agent_time_shapley", "ACCT - sampled Shapley"),
    ("learned_acct", "learned_cf_transport", "Learned ACCT - learned CF transport"),
    ("learned_acct", "learned_return_eq_cf_transport", "Learned ACCT - learned return-eq CF transport"),
    ("learned_acct", "learned_agent_time_cf", "Learned ACCT - learned AT-CF"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def first_episode_at_or_above(episodes: np.ndarray, values: np.ndarray, threshold: float) -> float:
    hits = np.flatnonzero(values >= threshold)
    if hits.size == 0:
        return float("nan")
    return float(episodes[int(hits[0])])


def curve_average(episodes: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(episodes)
    episodes = episodes[order]
    values = values[order]
    if len(episodes) == 1 or episodes[-1] == episodes[0]:
        return float(values[-1])
    return float(np.trapezoid(values, episodes) / (episodes[-1] - episodes[0]))


def summarize_metric(values: list[float]) -> tuple[str, str]:
    arr = np.asarray(values, dtype=np.float64)
    return f"{np.nanmean(arr):.4f}", f"{np.nanstd(arr):.4f}"


def summarize_threshold(values: list[float]) -> tuple[str, str]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return "nan", "0.0000"
    return f"{float(np.median(finite)):.1f}", f"{len(finite) / len(arr):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/local_experiments.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/local_efficiency_summary.csv"))
    parser.add_argument("--stats-out", type=Path, default=Path("results/local_efficiency_stat_tests.csv"))
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    rows = read_rows(args.input)
    grouped: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["method"], int(row["seed"]))].append(
            (int(row["episode"]), float(row["sample_return"]), float(row["greedy_return"]))
        )

    per_seed_rows = []
    for (env, method, seed), points in sorted(grouped.items()):
        episodes = np.asarray([point[0] for point in points], dtype=np.float64)
        sample_values = np.asarray([point[1] for point in points], dtype=np.float64)
        greedy_values = np.asarray([point[2] for point in points], dtype=np.float64)
        per_seed_rows.append(
            {
                "env": env,
                "method": method,
                "seed": seed,
                "sample_auc": curve_average(episodes, sample_values),
                "greedy_auc": curve_average(episodes, greedy_values),
                "first_sample_ge_threshold": first_episode_at_or_above(episodes, sample_values, args.threshold),
                "first_greedy_ge_threshold": first_episode_at_or_above(episodes, greedy_values, args.threshold),
            }
        )

    summary_rows = []
    by_env_method: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed_rows:
        by_env_method[(str(row["env"]), str(row["method"]))].append(row)
    for (env, method), method_rows in sorted(by_env_method.items()):
        sample_auc_mean, sample_auc_std = summarize_metric([float(row["sample_auc"]) for row in method_rows])
        greedy_auc_mean, greedy_auc_std = summarize_metric([float(row["greedy_auc"]) for row in method_rows])
        sample_threshold_median, sample_threshold_success = summarize_threshold(
            [float(row["first_sample_ge_threshold"]) for row in method_rows]
        )
        greedy_threshold_median, greedy_threshold_success = summarize_threshold(
            [float(row["first_greedy_ge_threshold"]) for row in method_rows]
        )
        summary_rows.append(
            {
                "env": env,
                "method": method,
                "sample_auc_mean": sample_auc_mean,
                "sample_auc_std": sample_auc_std,
                "greedy_auc_mean": greedy_auc_mean,
                "greedy_auc_std": greedy_auc_std,
                "threshold": f"{args.threshold:.3f}",
                "first_sample_ge_threshold_median": sample_threshold_median,
                "first_sample_ge_threshold_success": sample_threshold_success,
                "first_greedy_ge_threshold_median": greedy_threshold_median,
                "first_greedy_ge_threshold_success": greedy_threshold_success,
                "seeds": str(len(method_rows)),
            }
        )

    value_lookup = {
        (str(row["env"]), str(row["method"]), int(row["seed"])): row for row in per_seed_rows
    }
    envs = sorted({str(row["env"]) for row in per_seed_rows})
    rng = np.random.default_rng(args.seed)
    stats_rows = []
    for env in envs:
        for left, right, label in COMPARISONS:
            seeds = sorted(
                seed
                for e, m, seed in value_lookup
                if e == env and m == left and (env, right, seed) in value_lookup
            )
            if not seeds:
                continue
            for metric, higher_is_better in (
                ("sample_auc", True),
                ("greedy_auc", True),
                ("first_sample_ge_threshold", False),
            ):
                left_values = np.asarray([float(value_lookup[(env, left, seed)][metric]) for seed in seeds])
                right_values = np.asarray([float(value_lookup[(env, right, seed)][metric]) for seed in seeds])
                mask = np.isfinite(left_values) & np.isfinite(right_values)
                if not mask.any():
                    continue
                diffs = left_values[mask] - right_values[mask]
                ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=args.bootstrap_samples)
                win_rate = np.mean(diffs > 0) if higher_is_better else np.mean(diffs < 0)
                stats_rows.append(
                    {
                        "env": env,
                        "comparison": label,
                        "metric": metric,
                        "mean_diff": f"{diffs.mean():.4f}",
                        "ci95_low": f"{ci_low:.4f}",
                        "ci95_high": f"{ci_high:.4f}",
                        "win_rate": f"{win_rate:.4f}",
                        "paired_seeds": str(int(mask.sum())),
                    }
                )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "method",
                "sample_auc_mean",
                "sample_auc_std",
                "greedy_auc_mean",
                "greedy_auc_std",
                "threshold",
                "first_sample_ge_threshold_median",
                "first_sample_ge_threshold_success",
                "first_greedy_ge_threshold_median",
                "first_greedy_ge_threshold_success",
                "seeds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with args.stats_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "comparison",
                "metric",
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

    print(f"wrote efficiency summary to {args.summary_out}")
    print(f"wrote efficiency statistics to {args.stats_out}")


if __name__ == "__main__":
    main()
