"""Run local ACCT diagnostics and save raw CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from acct.envs import make_envs
from acct.learners import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/local_experiments.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = (
        "shared",
        "final_step_cf",
        "uniform_transport",
        "cf_transport",
        "return_eq_cf_transport",
        "agent_time_cf",
        "sampled_agent_time_shapley",
        "acct",
        "learned_cf_transport",
        "learned_return_eq_cf_transport",
        "learned_agent_time_cf",
        "learned_acct",
    )
    for env in make_envs():
        for method in methods:
            for seed in range(args.seeds):
                rows.extend(run_training(env, method=method, seed=seed, episodes=args.episodes))

    with args.out.open("w", newline="") as f:
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
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
