"""Paired statistical summaries for local ACCT diagnostics."""

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


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/local_experiments.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/local_stat_tests.csv"))
    parser.add_argument("--episode", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    with args.input.open() as f:
        rows = list(csv.DictReader(f))

    values: dict[tuple[str, str, int], float] = {}
    envs = sorted({row["env"] for row in rows})
    for row in rows:
        if int(row["episode"]) == args.episode:
            values[(row["env"], row["method"], int(row["seed"]))] = float(row["sample_return"])

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for env in envs:
        seeds_by_method: dict[str, set[int]] = defaultdict(set)
        for row_env, method, seed in values:
            if row_env == env:
                seeds_by_method[method].add(seed)
        for left, right, label in COMPARISONS:
            seeds = sorted(seeds_by_method[left] & seeds_by_method[right])
            if not seeds:
                continue
            diffs = np.asarray([values[(env, left, seed)] - values[(env, right, seed)] for seed in seeds])
            ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=args.bootstrap_samples)
            out_rows.append(
                {
                    "env": env,
                    "episode": str(args.episode),
                    "comparison": label,
                    "mean_diff": f"{diffs.mean():.4f}",
                    "ci95_low": f"{ci_low:.4f}",
                    "ci95_high": f"{ci_high:.4f}",
                    "win_rate": f"{np.mean(diffs > 0):.4f}",
                    "paired_seeds": str(len(seeds)),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
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
        writer.writerows(out_rows)
    print(f"wrote paired statistics to {args.out}")


if __name__ == "__main__":
    main()
