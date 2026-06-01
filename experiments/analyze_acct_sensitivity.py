"""Run a small ACCT transport hyperparameter sensitivity diagnostic."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from acct.envs import DelayedLeverEnv, PairGateEnv
from acct.learners import TabularPolicy, _credits_for_method, evaluate


VARIANTS = (
    ("abs_lam0.00_beta0.75", "acct", 0.00, 0.75),
    ("abs_lam0.50_beta0.75", "acct", 0.50, 0.75),
    ("abs_lam0.90_beta0.25", "acct", 0.90, 0.25),
    ("abs_lam0.90_beta0.75", "acct", 0.90, 0.75),
    ("abs_lam0.90_beta1.25", "acct", 0.90, 1.25),
    ("abs_lam0.99_beta0.75", "acct", 0.99, 0.75),
    ("dir_lam0.90_beta0.75", "directional_acct", 0.90, 0.75),
    ("dir_lam0.90_beta1.25", "directional_acct", 0.90, 1.25),
)


def run_variant(env, label: str, method: str, lam: float, beta: float, seed: int, episodes: int, eval_every: int, lr: float):
    policy = TabularPolicy(env.horizon, env.n_agents, env.n_actions, seed=seed)
    baseline = 0.0
    rows = []
    for episode in range(1, episodes + 1):
        actions = policy.sample()
        reward = float(env.reward(actions))
        baseline = 0.98 * baseline + 0.02 * reward
        credits = _credits_for_method(
            env,
            policy,
            actions,
            reward,
            baseline,
            method,
            transport_lam=lam,
            residual_weight=beta,
        )
        policy.update(actions, credits, lr=lr)
        if episode % eval_every == 0 or episode == 1:
            sample_return, greedy_return = evaluate(policy, env.reward, episodes=64)
            rows.append(
                {
                    "env": env.name,
                    "variant": label,
                    "method": method,
                    "lambda": lam,
                    "beta": beta,
                    "seed": seed,
                    "episode": episode,
                    "sample_return": sample_return,
                    "greedy_return": greedy_return,
                    "baseline": baseline,
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]], episode: int, final_episode: int):
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        if int(row["episode"]) in {episode, final_episode}:
            grouped.setdefault((str(row["env"]), str(row["variant"])), []).append(row)

    summary_rows = []
    for (env_name, variant), items in sorted(grouped.items()):
        sample_values = [float(row["sample_return"]) for row in items if int(row["episode"]) == episode]
        greedy_values = [float(row["greedy_return"]) for row in items if int(row["episode"]) == final_episode]
        if not sample_values or not greedy_values:
            continue
        lambda_value = float(items[0]["lambda"])
        beta_value = float(items[0]["beta"])
        method = str(items[0]["method"])
        summary_rows.append(
            {
                "env": env_name,
                "variant": variant,
                "method": method,
                "lambda": lambda_value,
                "beta": beta_value,
                f"sample_{episode}_mean": float(np.mean(sample_values)),
                f"sample_{episode}_std": float(np.std(sample_values, ddof=0)),
                f"greedy_{final_episode}_mean": float(np.mean(greedy_values)),
                f"greedy_{final_episode}_std": float(np.std(greedy_values, ddof=0)),
                "seeds": len(sample_values),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--sample-episode", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/acct_sensitivity.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/acct_sensitivity_summary.csv"))
    args = parser.parse_args()

    rows = []
    for env in (DelayedLeverEnv(), PairGateEnv()):
        for label, method, lam, beta in VARIANTS:
            for seed in range(args.seeds):
                rows.extend(run_variant(env, label, method, lam, beta, seed, args.episodes, args.eval_every, args.lr))

    write_rows(
        args.out,
        rows,
        fieldnames=[
            "env",
            "variant",
            "method",
            "lambda",
            "beta",
            "seed",
            "episode",
            "sample_return",
            "greedy_return",
            "baseline",
        ],
    )
    summary_rows = summarize(rows, episode=args.sample_episode, final_episode=args.episodes)
    write_rows(args.summary_out, summary_rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote {len(summary_rows)} rows to {args.summary_out}")


if __name__ == "__main__":
    main()
