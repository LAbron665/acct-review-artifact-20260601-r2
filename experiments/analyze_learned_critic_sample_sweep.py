"""Sample-size sweep for learned local counterfactual reward models.

This diagnostic asks whether the local learned-credit result depends on the
amount of uniformly sampled data used to fit the quadratic reward model.
It reports both held-out reward-model fit and downstream policy-gradient
returns for learned AT-CF and learned ACCT.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from acct.critics import fit_quadratic_reward_model
from acct.envs import make_envs
from acct.learners import run_training


METHODS = ("learned_agent_time_cf", "learned_acct")


def heldout_fit(env, seed: int, critic_samples: int, holdout: int) -> dict[str, str]:
    model = fit_quadratic_reward_model(env, seed=seed, samples=critic_samples)
    rng = np.random.default_rng(seed + 500_000 + critic_samples)
    y_true = np.zeros(holdout, dtype=np.float64)
    y_pred = np.zeros(holdout, dtype=np.float64)
    for idx in range(holdout):
        actions = rng.integers(env.n_actions, size=(env.horizon, env.n_agents), dtype=np.int64)
        y_true[idx] = float(env.reward(actions))
        y_pred[idx] = float(model.predict(actions))
    mse = float(np.mean((y_true - y_pred) ** 2))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom if denom > 0 else 0.0
    return {
        "env": env.name,
        "critic_samples": str(critic_samples),
        "seed": str(seed),
        "holdout": str(holdout),
        "mse": f"{mse:.6f}",
        "r2": f"{r2:.6f}",
    }


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def summarize_runs(run_rows: list[dict[str, str]], quality_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    final_episode = max(int(row["episode"]) for row in run_rows)
    grouped_returns: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in run_rows:
        if int(row["episode"]) == 200:
            grouped_returns[(row["env"], row["critic_samples"], row["method"], "sample_200")].append(
                float(row["sample_return"])
            )
        if int(row["episode"]) == final_episode:
            grouped_returns[(row["env"], row["critic_samples"], row["method"], "greedy_final")].append(
                float(row["greedy_return"])
            )

    grouped_quality: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in quality_rows:
        grouped_quality[(row["env"], row["critic_samples"], "mse")].append(float(row["mse"]))
        grouped_quality[(row["env"], row["critic_samples"], "r2")].append(float(row["r2"]))

    envs = sorted({row["env"] for row in run_rows})
    sample_sizes = sorted({row["critic_samples"] for row in run_rows}, key=int)
    out_rows: list[dict[str, str]] = []
    for env in envs:
        for sample_size in sample_sizes:
            mse_mean, mse_std = mean_std(grouped_quality[(env, sample_size, "mse")])
            r2_mean, r2_std = mean_std(grouped_quality[(env, sample_size, "r2")])
            for method in METHODS:
                sample_mean, sample_std = mean_std(grouped_returns[(env, sample_size, method, "sample_200")])
                greedy_mean, greedy_std = mean_std(grouped_returns[(env, sample_size, method, "greedy_final")])
                out_rows.append(
                    {
                        "env": env,
                        "critic_samples": sample_size,
                        "method": method,
                        "sample_200_mean": f"{sample_mean:.4f}",
                        "sample_200_std": f"{sample_std:.4f}",
                        "greedy_final_mean": f"{greedy_mean:.4f}",
                        "greedy_final_std": f"{greedy_std:.4f}",
                        "critic_mse_mean": f"{mse_mean:.6f}",
                        "critic_mse_std": f"{mse_std:.6f}",
                        "critic_r2_mean": f"{r2_mean:.6f}",
                        "critic_r2_std": f"{r2_std:.6f}",
                        "seeds": str(len(grouped_returns[(env, sample_size, method, "sample_200")])),
                        "final_episode": str(final_episode),
                    }
                )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--critic-samples", nargs="+", type=int, default=[128, 512, 2048, 8192])
    parser.add_argument("--holdout", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("results/learned_critic_sample_sweep.csv"))
    parser.add_argument(
        "--quality-out",
        type=Path,
        default=Path("results/learned_critic_sample_sweep_quality.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/learned_critic_sample_sweep_summary.csv"),
    )
    args = parser.parse_args()

    run_rows: list[dict[str, str]] = []
    quality_rows: list[dict[str, str]] = []
    for env in make_envs():
        for critic_samples in args.critic_samples:
            for seed in range(args.seeds):
                quality_rows.append(heldout_fit(env, seed=seed, critic_samples=critic_samples, holdout=args.holdout))
                for method in METHODS:
                    rows = run_training(
                        env,
                        method=method,
                        seed=seed,
                        episodes=args.episodes,
                        critic_samples=critic_samples,
                    )
                    for row in rows:
                        row = {key: str(value) for key, value in row.items()}
                        row["critic_samples"] = str(critic_samples)
                        run_rows.append(row)

    write_rows(
        args.out,
        run_rows,
        ["env", "method", "seed", "episode", "sample_return", "greedy_return", "baseline", "critic_samples"],
    )
    write_rows(args.quality_out, quality_rows, ["env", "critic_samples", "seed", "holdout", "mse", "r2"])
    summary_rows = summarize_runs(run_rows, quality_rows)
    write_rows(
        args.summary_out,
        summary_rows,
        [
            "env",
            "critic_samples",
            "method",
            "sample_200_mean",
            "sample_200_std",
            "greedy_final_mean",
            "greedy_final_std",
            "critic_mse_mean",
            "critic_mse_std",
            "critic_r2_mean",
            "critic_r2_std",
            "seeds",
            "final_episode",
        ],
    )
    print(f"wrote raw rows to {args.out}")
    print(f"wrote quality rows to {args.quality_out}")
    print(f"wrote summary rows to {args.summary_out}")


if __name__ == "__main__":
    main()
