"""Plot ACCT local diagnostics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "shared": "Shared return",
    "final_step_cf": "Final-step CF",
    "uniform_transport": "Uniform transport",
    "cf_transport": "CF transport-only",
    "return_eq_cf_transport": "Return-eq CF tr.",
    "agent_time_cf": "Agent-time CF",
    "sampled_agent_time_shapley": "Sampled Shapley",
    "acct": "ACCT",
    "learned_cf_transport": "Learned CF transport",
    "learned_return_eq_cf_transport": "Learned return-eq CF tr.",
    "learned_agent_time_cf": "Learned AT-CF",
    "learned_acct": "Learned ACCT",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/local_experiments.csv"))
    parser.add_argument("--figure", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/Figure/acct_local_results.pdf"))
    parser.add_argument("--summary", type=Path, default=Path("results/local_summary.csv"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    greedy_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    sample_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["env"], row["method"], int(row["episode"]))
        greedy_grouped[key].append(float(row["greedy_return"]))
        sample_grouped[key].append(float(row["sample_return"]))

    envs = sorted({row["env"] for row in rows})
    methods = [
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
    ]
    fig, axes = plt.subplots(2, len(envs), figsize=(7.3, 4.35), sharex=True, sharey=False)
    if len(envs) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    summary_rows = []
    legend_handles = None
    legend_labels = None
    for col, env in enumerate(envs):
        sample_ax = axes[0, col]
        greedy_ax = axes[1, col]
        for method in methods:
            episodes = sorted(ep for e, m, ep in greedy_grouped if e == env and m == method)
            sample_means = []
            sample_stderrs = []
            greedy_means = []
            greedy_stderrs = []
            for episode in episodes:
                sample_values = np.asarray(sample_grouped[(env, method, episode)], dtype=np.float64)
                greedy_values = np.asarray(greedy_grouped[(env, method, episode)], dtype=np.float64)
                sample_means.append(sample_values.mean())
                sample_stderrs.append(sample_values.std(ddof=0) / np.sqrt(max(len(sample_values), 1)))
                greedy_means.append(greedy_values.mean())
                greedy_stderrs.append(greedy_values.std(ddof=0) / np.sqrt(max(len(greedy_values), 1)))
            sample_line = sample_ax.plot(episodes, sample_means, label=METHOD_LABELS[method], linewidth=1.45)[0]
            sample_ax.fill_between(
                episodes,
                np.asarray(sample_means) - sample_stderrs,
                np.asarray(sample_means) + sample_stderrs,
                alpha=0.12,
            )
            greedy_ax.plot(episodes, greedy_means, color=sample_line.get_color(), linewidth=1.45)
            greedy_ax.fill_between(
                episodes,
                np.asarray(greedy_means) - greedy_stderrs,
                np.asarray(greedy_means) + greedy_stderrs,
                color=sample_line.get_color(),
                alpha=0.12,
            )
            sample_200 = np.asarray(sample_grouped[(env, method, 200)], dtype=np.float64)
            greedy_800 = np.asarray(greedy_grouped[(env, method, episodes[-1])], dtype=np.float64)
            summary_rows.append(
                {
                    "env": env,
                    "method": method,
                    "sample_200_mean": f"{sample_200.mean():.4f}",
                    "sample_200_std": f"{sample_200.std(ddof=0):.4f}",
                    "greedy_800_mean": f"{greedy_800.mean():.4f}",
                    "greedy_800_std": f"{greedy_800.std(ddof=0):.4f}",
                    "seeds": str(len(greedy_800)),
                }
            )
        sample_ax.set_title(env.replace("_", " ").title(), fontsize=9)
        sample_ax.set_ylabel("Sampled return")
        greedy_ax.set_ylabel("Greedy return")
        greedy_ax.set_xlabel("Episode")
        sample_ax.grid(alpha=0.25, linewidth=0.6)
        greedy_ax.grid(alpha=0.25, linewidth=0.6)
        if legend_handles is None:
            legend_handles, legend_labels = sample_ax.get_legend_handles_labels()
    fig.legend(legend_handles, legend_labels, frameon=False, fontsize=7.1, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"wrote figure to {args.figure}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "method",
                "sample_200_mean",
                "sample_200_std",
                "greedy_800_mean",
                "greedy_800_std",
                "seeds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
