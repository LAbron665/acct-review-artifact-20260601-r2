"""Summarize delayed environment diagnostic CSV results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/mpe_delayed_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/mpe_delayed_summary.csv"))
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["shared", "learned_agent_time_cf", "learned_acct", "learned_directional_acct"],
    )
    args = parser.parse_args()

    with args.input.open() as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    final_episode = max(int(row["episode"]) for row in rows)
    envs = sorted({row["env"] for row in rows})
    for row in rows:
        if int(row["episode"]) == final_episode:
            grouped[(row["env"], row["method"], "greedy_final")].append(float(row["greedy_return"]))
            grouped[(row["env"], row["method"], "sample_final")].append(float(row["sample_return"]))

    summary_rows = []
    for env in envs:
        for method in args.methods:
            for metric in ("greedy_final", "sample_final"):
                values = np.asarray(grouped[(env, method, metric)], dtype=np.float64)
                summary_rows.append(
                    {
                        "env": env,
                        "method": method,
                        "metric": metric,
                        "mean": f"{values.mean():.4f}",
                        "std": f"{values.std(ddof=0):.4f}",
                        "seeds": str(len(values)),
                        "final_episode": str(final_episode),
                    }
                )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env", "method", "metric", "mean", "std", "seeds", "final_episode"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
