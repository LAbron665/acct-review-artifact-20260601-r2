"""Per-trajectory query-budget summaries for local credit rules."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from acct.envs import make_envs


SHAPLEY_SAMPLES = 16


METHODS = (
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


METHOD_LABELS = {
    "shared": "Shared return",
    "final_step_cf": "Final-step CF",
    "uniform_transport": "Uniform transport",
    "cf_transport": "CF transport-only",
    "return_eq_cf_transport": "Return-equivalent CF transport",
    "agent_time_cf": "Agent-time CF",
    "sampled_agent_time_shapley": "Sampled Shapley",
    "acct": "ACCT",
    "learned_cf_transport": "Learned CF transport-only",
    "learned_return_eq_cf_transport": "Learned return-equivalent CF transport",
    "learned_agent_time_cf": "Learned AT-CF",
    "learned_acct": "Learned ACCT",
}


def budget_for(method: str, horizon: int, n_agents: int, n_actions: int) -> dict[str, int | str]:
    agent_time_pairs = horizon * n_agents
    full_counterfactual = agent_time_pairs * n_actions
    final_step_counterfactual = n_agents * n_actions
    transport_entries = n_agents * horizon * (horizon + 1) // 2
    shapley_upper_bound = SHAPLEY_SAMPLES * (1 + agent_time_pairs)

    if method == "shared":
        return {
            "credit_model_queries": 0,
            "replacement_queries": 0,
            "transport_weight_entries": 0,
            "shapley_permutations": 0,
            "note": "no credit-specific critic or reward-model query",
        }
    if method == "uniform_transport":
        return {
            "credit_model_queries": 0,
            "replacement_queries": 0,
            "transport_weight_entries": transport_entries,
            "shapley_permutations": 0,
            "note": "transport uses only observed TD residual",
        }
    if method == "final_step_cf":
        return {
            "credit_model_queries": 1 + final_step_counterfactual,
            "replacement_queries": final_step_counterfactual,
            "transport_weight_entries": 0,
            "shapley_permutations": 0,
            "note": "minimal terminal-step counterfactual budget",
        }
    if method == "sampled_agent_time_shapley":
        return {
            "credit_model_queries": shapley_upper_bound,
            "replacement_queries": SHAPLEY_SAMPLES * agent_time_pairs,
            "transport_weight_entries": 0,
            "shapley_permutations": SHAPLEY_SAMPLES,
            "note": "upper bound; skips replacement calls when reference equals actual",
        }
    if method in {
        "cf_transport",
        "return_eq_cf_transport",
        "agent_time_cf",
        "acct",
        "learned_cf_transport",
        "learned_return_eq_cf_transport",
        "learned_agent_time_cf",
        "learned_acct",
    }:
        transport_entries_for_method = transport_entries if "transport" in method or "acct" in method else 0
        return {
            "credit_model_queries": 1 + full_counterfactual,
            "replacement_queries": full_counterfactual,
            "transport_weight_entries": transport_entries_for_method,
            "shapley_permutations": 0,
            "note": "one actual prediction plus one replacement per agent-time-action",
        }
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/query_budget.csv"))
    args = parser.parse_args()

    rows = []
    for env in make_envs():
        for method in METHODS:
            budget = budget_for(method, env.horizon, env.n_agents, env.n_actions)
            rows.append(
                {
                    "env": env.name,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "horizon": env.horizon,
                    "n_agents": env.n_agents,
                    "n_actions": env.n_actions,
                    **budget,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "method",
                "label",
                "horizon",
                "n_agents",
                "n_actions",
                "credit_model_queries",
                "replacement_queries",
                "transport_weight_entries",
                "shapley_permutations",
                "note",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote query budget to {args.out}")


if __name__ == "__main__":
    main()
