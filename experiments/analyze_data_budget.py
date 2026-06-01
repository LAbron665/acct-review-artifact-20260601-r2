"""Summarize data budgets for the reported ACCT diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LOCAL_EXACT_METHODS = {
    "shared",
    "final_step_cf",
    "uniform_transport",
    "cf_transport",
    "return_eq_cf_transport",
    "agent_time_cf",
    "sampled_agent_time_shapley",
    "acct",
}
LOCAL_LEARNED_METHODS = {
    "learned_cf_transport",
    "learned_return_eq_cf_transport",
    "learned_agent_time_cf",
    "learned_acct",
}


FIELDNAMES = [
    "evidence_family",
    "sources",
    "environments",
    "methods",
    "seeds",
    "train_runs",
    "train_episodes_per_run",
    "train_trajectories",
    "eval_rows",
    "sample_eval_episodes",
    "greedy_eval_episodes",
    "extra_model_fit_trajectories",
    "heldout_model_eval_trajectories",
    "note",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_rows_if_present(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_rows(path)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_rows(rows: list[dict[str, str]], methods: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row["method"] in methods]


def join_sorted(values: set[str]) -> str:
    return "+".join(sorted(values))


def run_keys(rows: list[dict[str, str]], extra: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    return {
        tuple([row.get("env", ""), row["method"], row["seed"], *[row[key] for key in extra]])
        for row in rows
    }


def max_episode(rows: list[dict[str, str]]) -> int:
    return max(int(row["episode"]) for row in rows)


def policy_budget_row(
    *,
    evidence_family: str,
    sources: str,
    rows: list[dict[str, str]],
    train_episodes_per_run: int | str | None = None,
    sample_eval_per_row: int = 0,
    greedy_eval_per_row: int = 0,
    extra_model_fit_trajectories: int = 0,
    heldout_model_eval_trajectories: int = 0,
    key_extra: tuple[str, ...] = (),
    note: str,
) -> dict[str, str]:
    runs = run_keys(rows, extra=key_extra)
    episodes = train_episodes_per_run if train_episodes_per_run is not None else max_episode(rows)
    train_trajectories = "" if not isinstance(episodes, int) else len(runs) * episodes
    seed_values = {row["seed"] for row in rows}
    env_values = {row.get("env", "") for row in rows}
    method_values = {row["method"] for row in rows}
    return {
        "evidence_family": evidence_family,
        "sources": sources,
        "environments": str(len(env_values)),
        "methods": str(len(method_values)),
        "seeds": str(len(seed_values)),
        "train_runs": str(len(runs)),
        "train_episodes_per_run": str(episodes),
        "train_trajectories": str(train_trajectories),
        "eval_rows": str(len(rows)),
        "sample_eval_episodes": str(len(rows) * sample_eval_per_row),
        "greedy_eval_episodes": str(len(rows) * greedy_eval_per_row),
        "extra_model_fit_trajectories": str(extra_model_fit_trajectories),
        "heldout_model_eval_trajectories": str(heldout_model_eval_trajectories),
        "note": note,
    }


def critic_sweep_extra_fit(rows: list[dict[str, str]], quality_rows: list[dict[str, str]]) -> tuple[int, int]:
    policy_fit_keys = run_keys(rows, extra=("critic_samples",))
    policy_fit = 0
    for env, method, seed, critic_samples in policy_fit_keys:
        _ = (env, method, seed)
        policy_fit += int(critic_samples)

    quality_fit_keys = {
        (row["env"], row["seed"], row["critic_samples"], row["holdout"])
        for row in quality_rows
    }
    quality_fit = sum(int(critic_samples) for _, _, critic_samples, _ in quality_fit_keys)
    holdout = sum(int(holdout) for _, _, _, holdout in quality_fit_keys)
    return policy_fit + quality_fit, holdout


def combined_row(
    evidence_family: str,
    sources: str,
    rows_by_source: list[tuple[list[dict[str, str]], int, int, int | None]],
    note: str,
) -> dict[str, str]:
    all_rows: list[dict[str, str]] = []
    train_runs = 0
    train_trajectories = 0
    sample_eval_episodes = 0
    greedy_eval_episodes = 0
    episode_values: set[str] = set()
    for rows, train_episodes, sample_eval_per_row, greedy_eval_per_row in rows_by_source:
        all_rows.extend(rows)
        runs = run_keys(rows)
        train_runs += len(runs)
        train_trajectories += len(runs) * train_episodes
        sample_eval_episodes += len(rows) * sample_eval_per_row
        greedy_eval_episodes += len(rows) * greedy_eval_per_row
        episode_values.add(str(train_episodes))

    seed_values = {row["seed"] for row in all_rows}
    env_values = {row.get("env", "") for row in all_rows}
    method_values = {row["method"] for row in all_rows}
    return {
        "evidence_family": evidence_family,
        "sources": sources,
        "environments": str(len(env_values)),
        "methods": str(len(method_values)),
        "seeds": str(len(seed_values)),
        "train_runs": str(train_runs),
        "train_episodes_per_run": join_sorted(episode_values),
        "train_trajectories": str(train_trajectories),
        "eval_rows": str(len(all_rows)),
        "sample_eval_episodes": str(sample_eval_episodes),
        "greedy_eval_episodes": str(greedy_eval_episodes),
        "extra_model_fit_trajectories": "0",
        "heldout_model_eval_trajectories": "0",
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/data_budget.csv"))
    args = parser.parse_args()

    results = args.results
    local = read_rows(results / "local_experiments.csv")
    local_exact = method_rows(local, LOCAL_EXACT_METHODS)
    local_learned = method_rows(local, LOCAL_LEARNED_METHODS)
    learned_sweep = read_rows(results / "learned_critic_sample_sweep.csv")
    learned_sweep_quality = read_rows(results / "learned_critic_sample_sweep_quality.csv")
    sweep_fit, sweep_holdout = critic_sweep_extra_fit(learned_sweep, learned_sweep_quality)
    pruned_learned = read_rows(results / "pruned_learned_acct_results.csv")
    pruned_fraction = read_rows(results / "pruned_fraction_sensitivity_results.csv")
    distractor_scaling = read_rows(results / "distractor_scaling_results.csv")
    perturbation = read_rows(results / "influence_perturbation.csv")

    mpe_delayed = read_rows(results / "mpe_delayed_results.csv")
    mpe_ppo = read_rows(results / "mpe_ppo_results.csv")
    mpe_mappo = read_rows(results / "mpe_mappo_results.csv")
    lbf = read_rows(results / "lbf_ppo_results.csv")
    rware = read_rows(results / "rware_ppo_results.csv")
    medium_mpe_ppo = read_rows_if_present(results / "medium_mpe_ppo_results.csv")
    medium_mpe_mappo = read_rows_if_present(results / "medium_mpe_mappo_results.csv")
    medium_lbf = read_rows_if_present(results / "medium_lbf_ppo_results.csv")
    medium_rware = read_rows_if_present(results / "medium_rware_ppo_results.csv")
    medium_blocks = []
    if medium_mpe_ppo:
        medium_blocks.append((medium_mpe_ppo, 600, 0, 5))
    if medium_mpe_mappo:
        medium_blocks.append((medium_mpe_mappo, 600, 0, 5))
    if medium_lbf:
        medium_blocks.append((medium_lbf, 1000, 0, 8))
    if medium_rware:
        medium_blocks.append((medium_rware, 1000, 0, 8))

    rows = [
        policy_budget_row(
            evidence_family="Local exact/oracle diagnostics",
            sources="local_experiments.csv",
            rows=local_exact,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            note="Exact reward or critic queries are accounted separately in query_budget.csv.",
        ),
        policy_budget_row(
            evidence_family="Local learned diagnostics",
            sources="local_experiments.csv",
            rows=local_learned,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            extra_model_fit_trajectories=len(run_keys(local_learned)) * 2048,
            note="Each learned method fits a quadratic reward model from 2048 uniform trajectories.",
        ),
        policy_budget_row(
            evidence_family="Learned critic sample sweep",
            sources="learned_critic_sample_sweep*.csv",
            rows=learned_sweep,
            train_episodes_per_run=400,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            extra_model_fit_trajectories=sweep_fit,
            heldout_model_eval_trajectories=sweep_holdout,
            key_extra=("critic_samples",),
            note="Includes policy-run critic fits plus separate held-out model fits.",
        ),
        policy_budget_row(
            evidence_family="Pruned learned ACCT diagnostic",
            sources="pruned_learned_acct*.csv",
            rows=pruned_learned,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            extra_model_fit_trajectories=len(run_keys(pruned_learned)) * 2048,
            note="Local learned-credit diagnostic comparing top-fraction pruned ACCT against learned ACCT and learned AT-CF.",
        ),
        policy_budget_row(
            evidence_family="Pruned fraction sensitivity",
            sources="pruned_fraction_sensitivity*.csv",
            rows=pruned_fraction,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            key_extra=("top_fraction",),
            extra_model_fit_trajectories=len(run_keys(pruned_fraction, extra=("top_fraction",))) * 2048,
            note="Local sensitivity diagnostic for learned-pruned ACCT top-fraction choices.",
        ),
        policy_budget_row(
            evidence_family="Distractor scaling stress test",
            sources="distractor_scaling*.csv",
            rows=distractor_scaling,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            note="Local stress diagnostic that increases irrelevant agent-time pairs while holding relevant pairs fixed.",
        ),
        policy_budget_row(
            evidence_family="Influence perturbation diagnostics",
            sources="influence_perturbation*.csv",
            rows=perturbation,
            train_episodes_per_run=400,
            sample_eval_per_row=64,
            greedy_eval_per_row=1,
            key_extra=("perturbation",),
            note="Exact local diagnostic perturbing counterfactual-influence scale and noise.",
        ),
        combined_row(
            evidence_family="MPE2 smoke diagnostics",
            sources="mpe_delayed_results.csv+mpe_ppo_results.csv+mpe_mappo_results.csv",
            rows_by_source=[
                (mpe_delayed, 300, 0, 3),
                (mpe_ppo, 300, 0, 5),
                (mpe_mappo, 300, 0, 5),
            ],
            note="Sample returns are from training episodes or batches; greedy evaluation uses extra environment episodes.",
        ),
        policy_budget_row(
            evidence_family="LBF stress test",
            sources="lbf_ppo_results.csv",
            rows=lbf,
            train_episodes_per_run=500,
            greedy_eval_per_row=8,
            note="Tiny external-environment stress test, not tuned benchmark evidence.",
        ),
        policy_budget_row(
            evidence_family="RWARE stress test",
            sources="rware_ppo_results.csv",
            rows=rware,
            train_episodes_per_run=300,
            greedy_eval_per_row=8,
            note="Tiny external-environment stress test with zero-return outcome.",
        ),
        *(
            [
                combined_row(
                    evidence_family="Medium benchmark suite",
                    sources="medium_mpe_ppo_results.csv+medium_mpe_mappo_results.csv+medium_lbf_ppo_results.csv+medium_rware_ppo_results.csv",
                    rows_by_source=medium_blocks,
                    note="Medium local suite requested for paper polish: MPE2 at 600 episodes and LBF/RWARE at 1000 episodes, all with 16 seeds when present.",
                )
            ]
            if medium_blocks
            else []
        ),
    ]

    write_rows(args.out, rows)
    print(f"wrote data budget to {args.out}")


if __name__ == "__main__":
    main()
