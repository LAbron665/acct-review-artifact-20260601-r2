"""Check whether learned-pruned ACCT depends sharply on the pruning fraction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acct.envs import make_envs
from acct.learners import run_training


def summarize(rows: list[dict[str, float | int | str]], episode: int) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    envs = sorted({str(row["env"]) for row in rows})
    fractions = sorted({float(row["top_fraction"]) for row in rows})
    for env in envs:
        for fraction in fractions:
            selected = [
                row for row in rows
                if row["env"] == env
                and abs(float(row["top_fraction"]) - fraction) < 1e-12
                and int(row["episode"]) == episode
            ]
            sample = np.asarray([float(row["sample_return"]) for row in selected], dtype=np.float64)
            greedy = np.asarray([float(row["greedy_return"]) for row in selected], dtype=np.float64)
            out_rows.append(
                {
                    "env": env,
                    "top_fraction": f"{fraction:.2f}",
                    "episode": str(episode),
                    "sample_mean": f"{sample.mean():.4f}",
                    "sample_std": f"{sample.std(ddof=1):.4f}",
                    "greedy_mean": f"{greedy.mean():.4f}",
                    "greedy_std": f"{greedy.std(ddof=1):.4f}",
                    "seeds": str(len(selected)),
                }
            )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--episode", type=int, default=200)
    parser.add_argument("--critic-samples", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("results/pruned_fraction_sensitivity_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/pruned_fraction_sensitivity_summary.csv"))
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for env in make_envs():
        for fraction in args.fractions:
            for seed in range(args.seeds):
                run_rows = run_training(
                    env,
                    method="learned_pruned_acct",
                    seed=seed,
                    episodes=args.episodes,
                    critic_samples=args.critic_samples,
                    salience_fraction=fraction,
                )
                for row in run_rows:
                    row["top_fraction"] = f"{fraction:.2f}"
                rows.extend(run_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "method",
                "top_fraction",
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
                "top_fraction",
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

    print(f"wrote pruned-fraction sensitivity results to {args.out}")
    print(f"wrote pruned-fraction sensitivity summary to {args.summary}")


if __name__ == "__main__":
    main()
