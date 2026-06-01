"""Paired bootstrap statistics for MPE/LBF diagnostic CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--episode", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    with args.input.open() as f:
        rows = [row for row in csv.DictReader(f) if int(row["episode"]) == args.episode]

    all_methods = sorted({row["method"] for row in rows})
    methods = args.methods or [method for method in all_methods if method != args.baseline_method]
    values: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        values[(row["method"], int(row["seed"]), row["env"])] = {
            "sample_return": float(row["sample_return"]),
            "greedy_return": float(row["greedy_return"]),
        }

    envs = sorted({row["env"] for row in rows})
    rng = np.random.default_rng(args.seed)
    out_rows = []
    for env in envs:
        baseline_seeds = {seed for method, seed, row_env in values if method == args.baseline_method and row_env == env}
        for method in methods:
            method_seeds = {seed for row_method, seed, row_env in values if row_method == method and row_env == env}
            paired_seeds = sorted(baseline_seeds & method_seeds)
            if not paired_seeds:
                continue
            for metric in ("sample_return", "greedy_return"):
                diffs = np.asarray(
                    [
                        values[(method, seed, env)][metric] - values[(args.baseline_method, seed, env)][metric]
                        for seed in paired_seeds
                    ],
                    dtype=np.float64,
                )
                ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=args.bootstrap_samples)
                out_rows.append(
                    {
                        "env": env,
                        "episode": str(args.episode),
                        "metric": metric,
                        "comparison": f"{method} - {args.baseline_method}",
                        "mean_diff": f"{diffs.mean():.4f}",
                        "ci95_low": f"{ci_low:.4f}",
                        "ci95_high": f"{ci_high:.4f}",
                        "win_rate": f"{np.mean(diffs > 0):.4f}",
                        "paired_seeds": str(len(paired_seeds)),
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "episode",
                "metric",
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
