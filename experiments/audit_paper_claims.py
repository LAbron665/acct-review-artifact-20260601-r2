"""Check that key numeric claims in the paper match artifact CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def q(value: str | float, places: int = 3) -> str:
    quant = Decimal("1").scaleb(-places)
    return str(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def mean_pm(row: dict[str, str], places: int = 3) -> str:
    return f"{q(row['mean'], places)} \\pm {q(row['std'], places)}"


def mean_pm_fields(row: dict[str, str], mean_key: str, std_key: str, places: int = 3) -> str:
    return f"{q(row[mean_key], places)} \\pm {q(row[std_key], places)}"


def interval(row: dict[str, str], places: int = 3) -> str:
    return f"[{q(row['ci95_low'], places)},{q(row['ci95_high'], places)}]"


def latex_int(value: str | int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def row_by(rows: list[dict[str, str]], **query: str) -> dict[str, str]:
    matches = [row for row in rows if all(row[key] == value for key, value in query.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {query}, found {len(matches)}")
    return matches[0]


def add_expectation(expectations: list[dict[str, str]], label: str, source: str, snippet: str) -> None:
    expectations.append({"label": label, "source": source, "snippet": snippet})


def add_any_expectation(expectations: list[dict[str, object]], label: str, source: str, snippets: list[str]) -> None:
    expectations.append({"label": label, "source": source, "snippet": " || ".join(snippets), "snippets": snippets})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("../Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex"),
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=Path("../Formatting_Instructions_For_NeurIPS_2026/checklist.tex"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/paper_claim_audit.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    paper_path = args.paper if args.paper.is_absolute() else root / args.paper
    checklist_path = args.checklist if args.checklist.is_absolute() else root / args.checklist
    paper_text = paper_path.read_text()
    if checklist_path.exists():
        paper_text += "\n" + checklist_path.read_text()
    results = root / "results"
    expectations: list[dict[str, object]] = []

    add_expectation(
        expectations,
        "reporting mean std convention",
        "reporting convention",
        "mean $\\pm$ standard deviation",
    )
    add_expectation(
        expectations,
        "reporting bootstrap convention",
        "reporting convention",
        "10{,}000 bootstrap resamples",
    )
    add_expectation(
        expectations,
        "reporting descriptive intervals",
        "reporting convention",
        "descriptive uncertainty summaries",
    )
    add_expectation(
        expectations,
        "reporting full curve metrics",
        "reporting convention",
        "full-curve learning-efficiency",
    )
    add_expectation(
        expectations,
        "reporting auc metric",
        "reporting convention",
        "sampled-return AUC",
    )
    add_expectation(
        expectations,
        "reporting threshold metric",
        "reporting convention",
        "first episode at which each method reaches sampled return 1.5",
    )
    add_expectation(
        expectations,
        "query budget script",
        "query_budget.csv",
        "code/results/query\\_budget.csv",
    )
    add_expectation(
        expectations,
        "data budget script",
        "data_budget.csv",
        "code/results/data\\_budget.csv",
    )
    add_expectation(
        expectations,
        "data budget analyzer",
        "data_budget.csv",
        "analyze\\_data\\_budget.py",
    )
    add_expectation(
        expectations,
        "data budget critic fit caveat",
        "data_budget.csv",
        "auxiliary critic-fit data budget",
    )
    for label, snippet in [
        ("query delayed acct queries", "41 critic/reward-model queries"),
        ("query delayed acct transport", "60 transport-weight entries"),
        ("query delayed shapley queries", "upper bound of 336 queries"),
        ("query pair acct queries", "49 critic/reward-model queries"),
        ("query pair acct transport", "84 transport-weight entries"),
        ("query pair shapley queries", "upper bound of 400 queries"),
        ("query fairness conclusion", "not obtained by giving \\acct{} a larger counterfactual query budget"),
    ]:
        add_expectation(expectations, label, "query budget reporting", snippet)
    add_any_expectation(
        expectations,
        "artifact verified zip or final url status",
        "experimental protocol",
        [
            "verified local ZIP rather than a public anonymous repository",
            "The code artifact is available for anonymous review at",
        ],
    )
    add_any_expectation(
        expectations,
        "artifact final url replacement state",
        "experimental protocol",
        [
            "replace this sentence with the review URL",
            "The code and data are available at",
        ],
    )
    add_any_expectation(
        expectations,
        "checklist public url state",
        "experimental protocol",
        [
            "does not yet provide a public anonymous repository URL",
            "The code and data are available at",
        ],
    )
    for label, snippet in [
        ("artifact local package status", "identity-scrubbed local artifact package"),
        ("artifact anonymous upload status", "anonymous review URL"),
        ("checklist llm methodology status", "No methodologically important, original, or non-standard LLM or agent component"),
        ("checklist llm audit status", "all claims, citations, and numerical results are checked against the manuscript and artifact audits"),
        ("objective status not return equivalent", "training-credit transformation rather than a return-equivalent reward-redistribution method"),
        ("objective status credit sum", "not constrained to sum to the environment return"),
        ("objective status not unbiased", "not generally an unbiased policy-gradient advantage"),
        ("objective status policy preservation", "rather than a universal policy-preservation theorem"),
        ("transport scale invariance", "Transport-scale invariance"),
        ("transport scale component caveat", "This proposition concerns $C^{\\mathrm{tr}}$ only"),
        ("transport full credit scale caveat", "full \\acct{} credit remains scale-sensitive"),
        ("transport learned calibration boundary", "raw agent-time CF directly inherits the learned head's magnitude calibration"),
        ("evidence scope map", "Evidence-scope map"),
        ("evidence mechanistic claim", "The headline empirical claim is mechanistic"),
        ("evidence perturbation scope", "Influence perturbation diagnostics"),
        ("evidence local unsupported", "Performance dominance on high-dimensional MARL benchmarks"),
        ("evidence mpe unsupported", "A reliable performance advantage over MAPPO"),
        ("evidence independent reproduction unsupported", "Independent third-party reproduction"),
        ("positioning appendix", "Positioning Against Agent-Temporal Credit Methods"),
        ("positioning table", "Positioning of \\acct{} relative to closely related temporal and agent-temporal credit-assignment methods"),
        ("positioning maca related", "multi-level coordination credit on StarCraft tasks"),
        ("positioning maca boundary", "coalition-level spatial credit across agent subsets"),
        ("positioning tar2 shaping", "potential-based shaping connection"),
        ("positioning tar2 benchmarks", "SMACLite and Google Research Football"),
        ("positioning cast", "CAST imposes causal structure"),
        ("positioning magic", "MAGIC estimates multi-step inter-agent causal influence"),
        ("positioning design boundary", "not a complete reward-redistribution architecture"),
        ("positioning acct not causal discovery", "causal-discovery model"),
        ("positioning acct not intrinsic motivation", "intrinsic-motivation system"),
        ("positioning asynchronous method", "asynchronous-action credit assignment is orthogonal"),
        ("positioning asynchronous boundary", "synchronized trajectory lattice"),
        ("limitations synchronous action", "synchronous-action local diagnostics"),
        ("artifact makefile smoke", "\\texttt{artifact-smoke}"),
        ("artifact anonymity audit", "anonymity audit"),
        ("artifact anonymity audit file", "results/anonymity\\_audit.csv"),
        ("artifact makefile release", "\\texttt{release-check}"),
        ("artifact release repo target", "\\texttt{release-repo}"),
        ("artifact dependency license audit", "dependency-license metadata"),
        ("artifact third party notes", "THIRD\\_PARTY.md"),
        ("checklist dependency license audit", "dependency-license metadata audit generated from installed Python package metadata"),
        ("artifact bibliography audit", "bibliography consistency"),
        ("artifact release repo metadata", "clean local anonymous repository"),
        ("artifact anonymous mirror route", "anonymous.4open.science"),
        ("artifact latex submission audit", "\\LaTeX{} submission audit"),
        ("artifact manifest hashes", "validate manifest SHA-256 hashes"),
        ("artifact zip verifier", "anonymous-ZIP verifier"),
        ("artifact manifest scope", "manifest hashes"),
        ("local learning rate", "learning rate $0.08$"),
        ("local ema baseline", "update weight $0.02$"),
        ("local diagnostic seeds", "The local diagnostics use 32 seeds"),
        ("local figure seeds", "Local diagnostic results over 32 seeds"),
        ("local figure sampled greedy panels", "sampled-policy evaluation and bottom panels show greedy-policy evaluation"),
        ("local transport lambda beta", "default $\\lambda=0.90$, and default residual weight $\\beta=0.75$"),
        ("local return equivalent proxy", "Return-eq CF transport"),
        ("support preservation proposition", "Support preservation under nonzero salience mass"),
        ("support preservation condition", "some eligible nonzero salience must remain for each nonzero residual"),
        ("influence perturbation summary", "influence\\_perturbation\\_summary.csv"),
        ("influence perturbation statistics", "influence\\_perturbation\\_stat\\_tests.csv"),
        ("influence perturbation protocol", "16 seeds and 400 training episodes"),
        ("influence perturbation caveat", "does not imply robustness to arbitrary critic misspecification"),
        ("influence perturbation scale caveat", "full \\acct{} credit remains scale-sensitive through its instantaneous term"),
        ("learned salience pruning file", "learned\\_salience\\_pruning.csv"),
        ("learned salience ranking caveat", "not be read as a new trained variant of \\acct{}"),
        ("learned salience calibration caveat", "would need confidence calibration or pruning"),
        ("pruned learned acct summary file", "Pruned learned \\acct{}"),
        ("pruned learned acct diagnostic caveat", "not used as the headline method"),
        ("pruned fraction sensitivity table", "Pruned learned-\\acct{} pruning-fraction sensitivity"),
        ("pruned fraction sensitivity caveat", "not identical across tasks"),
        ("distractor scaling table", "Distractor-scaling stress diagnostic"),
        ("distractor scaling caveat", "not a claim about SMAC-scale generalization"),
        ("learned local critic samples", "2048 uniformly sampled trajectories"),
        ("mpe actor critic learning rates", "actor learning rate $2\\cdot10^{-3}$, critic learning rate $3\\cdot10^{-3}$"),
        ("mpe ppo protocol", "minibatches of eight episodes, three PPO epochs"),
        ("mpe acct protocol", "residual weight $0.5$; hybrid MAPPO uses learned-credit weight $0.05$"),
        ("lbf protocol", "500 training episodes, 25-step episodes"),
        ("rware protocol", "300 training episodes, eight seeds, 100-step episodes"),
        ("method schematic figure file", "Figure/acct_method_schematic.pdf"),
        ("polished local figure file", "Figure/acct_local_results_polished.pdf"),
        ("medium benchmark figure file", "Figure/acct_medium_benchmarks.pdf"),
        ("diagnostics appendix figure file", "Figure/acct_diagnostics_appendix.pdf"),
        ("medium benchmark summary file", "medium\\_benchmark\\_summary.csv"),
        ("medium benchmark stats file", "medium\\_benchmark\\_stat\\_tests.csv"),
        ("medium benchmark protocol", "600 training episodes with 16 seeds"),
        ("medium lbf rware protocol", "1000 training episodes with 16 seeds"),
        ("medium benchmark metrics", "final sampled return, final greedy return, sampled-return AUC, greedy-return AUC, seed-wise win rate"),
    ]:
        add_expectation(expectations, label, "experimental protocol", snippet)
    add_any_expectation(
        expectations,
        "artifact final url helper or applied URL",
        "experimental protocol",
        ["finalize\\_anonymous\\_url.py", "The code and data are available at"],
    )

    local_summary = read_rows(results / "local_summary.csv")
    for row in local_summary:
        add_expectation(
            expectations,
            f"local summary {row['env']} {row['method']} sample 200",
            "local_summary.csv",
            mean_pm_fields(row, "sample_200_mean", "sample_200_std"),
        )

    salience_pruning = read_rows(results / "learned_salience_pruning.csv")
    for env, top_fraction in [
        ("delayed_lever", "0.20"),
        ("delayed_lever", "1.00"),
        ("pair_gate", "0.20"),
        ("pair_gate", "1.00"),
    ]:
        row = row_by(salience_pruning, env=env, top_fraction=top_fraction)
        label = f"learned salience pruning {env} top {top_fraction}"
        add_expectation(expectations, f"{label} kept pairs", "learned_salience_pruning.csv", q(row["kept_pairs_mean"]))
        add_expectation(expectations, f"{label} relevant mass", "learned_salience_pruning.csv", q(row["relevant_mass"]))
        add_expectation(expectations, f"{label} irrelevant active", "learned_salience_pruning.csv", q(row["irrelevant_active"]))

    pruned_summary = read_rows(results / "pruned_learned_acct_summary.csv")
    pruned_stats = read_rows(results / "pruned_learned_acct_stat_tests.csv")
    for env in ("delayed_lever", "pair_gate"):
        row = row_by(pruned_summary, env=env, method="learned_pruned_acct")
        add_expectation(
            expectations,
            f"pruned learned acct {env} sample 200",
            "pruned_learned_acct_summary.csv",
            mean_pm_fields(row, "sample_mean", "sample_std"),
        )
        diff_row = row_by(pruned_stats, env=env, comparison="Pruned learned ACCT - learned ACCT")
        label = f"pruned learned acct {env} vs learned acct"
        add_expectation(expectations, label, "pruned_learned_acct_stat_tests.csv", q(diff_row["mean_diff"]))
        add_expectation(expectations, f"{label} interval", "pruned_learned_acct_stat_tests.csv", interval(diff_row))

    pruned_fraction = read_rows(results / "pruned_fraction_sensitivity_summary.csv")
    for env in ("delayed_lever", "pair_gate"):
        for top_fraction in ("0.10", "0.20", "0.30"):
            row = row_by(pruned_fraction, env=env, top_fraction=top_fraction)
            add_expectation(
                expectations,
                f"pruned fraction {env} top {top_fraction}",
                "pruned_fraction_sensitivity_summary.csv",
                mean_pm_fields(row, "sample_mean", "sample_std"),
            )

    distractor_scaling = read_rows(results / "distractor_scaling_summary.csv")
    for total_pairs in ("20", "32", "48", "72"):
        for method in ("shared", "agent_time_cf", "acct"):
            row = row_by(distractor_scaling, total_pairs=total_pairs, method=method)
            add_expectation(
                expectations,
                f"distractor scaling {total_pairs} pairs {method}",
                "distractor_scaling_summary.csv",
                mean_pm_fields(row, "sample_mean", "sample_std"),
            )

    local_stats = read_rows(results / "local_stat_tests.csv")
    local_specs = [
        ("delayed_lever", "ACCT - uniform transport"),
        ("pair_gate", "ACCT - uniform transport"),
        ("delayed_lever", "ACCT - CF transport-only"),
        ("pair_gate", "ACCT - CF transport-only"),
        ("delayed_lever", "ACCT - return-eq CF transport"),
        ("pair_gate", "ACCT - return-eq CF transport"),
        ("delayed_lever", "ACCT - agent-time CF"),
        ("pair_gate", "ACCT - agent-time CF"),
        ("delayed_lever", "ACCT - sampled Shapley"),
        ("pair_gate", "ACCT - sampled Shapley"),
        ("delayed_lever", "Learned ACCT - learned CF transport"),
        ("pair_gate", "Learned ACCT - learned CF transport"),
        ("delayed_lever", "Learned ACCT - learned return-eq CF transport"),
        ("pair_gate", "Learned ACCT - learned return-eq CF transport"),
        ("delayed_lever", "Learned ACCT - learned AT-CF"),
        ("pair_gate", "Learned ACCT - learned AT-CF"),
    ]
    for env, comparison in local_specs:
        row = row_by(local_stats, env=env, comparison=comparison)
        label = f"local {env} {comparison}"
        add_expectation(expectations, label, "local_stat_tests.csv", q(row["mean_diff"]))
        add_expectation(expectations, f"{label} interval", "local_stat_tests.csv", interval(row))

    query_budget = read_rows(results / "query_budget.csv")
    query_specs = [
        ("delayed_lever", "final_step_cf"),
        ("delayed_lever", "agent_time_cf"),
        ("delayed_lever", "return_eq_cf_transport"),
        ("delayed_lever", "acct"),
        ("delayed_lever", "sampled_agent_time_shapley"),
        ("pair_gate", "final_step_cf"),
        ("pair_gate", "agent_time_cf"),
        ("pair_gate", "return_eq_cf_transport"),
        ("pair_gate", "acct"),
        ("pair_gate", "sampled_agent_time_shapley"),
    ]
    for env, method in query_specs:
        row = row_by(query_budget, env=env, method=method)
        label = f"query budget {env} {method}"
        add_expectation(expectations, f"{label} queries", "query_budget.csv", row["credit_model_queries"])
        add_expectation(expectations, f"{label} replacements", "query_budget.csv", row["replacement_queries"])
        add_expectation(expectations, f"{label} transport", "query_budget.csv", row["transport_weight_entries"])

    data_budget = read_rows(results / "data_budget.csv")
    for family in (
        "Local exact/oracle diagnostics",
        "Local learned diagnostics",
        "Learned critic sample sweep",
        "Pruned learned ACCT diagnostic",
        "Pruned fraction sensitivity",
        "Distractor scaling stress test",
        "Influence perturbation diagnostics",
        "MPE2 smoke diagnostics",
        "LBF stress test",
        "RWARE stress test",
    ):
        row = row_by(data_budget, evidence_family=family)
        policy_eval = int(row["sample_eval_episodes"]) + int(row["greedy_eval_episodes"])
        snippet = (
            f"{family} & ${row['train_runs']}$ & ${latex_int(row['train_trajectories'])}$ "
            f"& ${latex_int(row['extra_model_fit_trajectories'])}$ & ${latex_int(policy_eval)}$ "
            f"& ${latex_int(row['heldout_model_eval_trajectories'])}$"
        )
        add_expectation(expectations, f"data budget {family}", "data_budget.csv", snippet)

    efficiency_summary = read_rows(results / "local_efficiency_summary.csv")
    efficiency_methods = (
        "shared",
        "cf_transport",
        "agent_time_cf",
        "sampled_agent_time_shapley",
        "acct",
        "learned_agent_time_cf",
        "learned_acct",
    )
    for env in ("delayed_lever", "pair_gate"):
        for method in efficiency_methods:
            row = row_by(efficiency_summary, env=env, method=method)
            add_expectation(
                expectations,
                f"efficiency {env} {method} sample auc",
                "local_efficiency_summary.csv",
                mean_pm_fields(row, "sample_auc_mean", "sample_auc_std"),
            )
            threshold = row["first_sample_ge_threshold_median"]
            if threshold != "nan":
                add_expectation(
                    expectations,
                    f"efficiency {env} {method} threshold",
                    "local_efficiency_summary.csv",
                    str(int(float(threshold))),
                )

    efficiency_stats = read_rows(results / "local_efficiency_stat_tests.csv")
    efficiency_specs = [
        ("delayed_lever", "ACCT - agent-time CF", "sample_auc"),
        ("pair_gate", "ACCT - agent-time CF", "sample_auc"),
        ("delayed_lever", "ACCT - sampled Shapley", "sample_auc"),
        ("pair_gate", "ACCT - sampled Shapley", "sample_auc"),
        ("delayed_lever", "Learned ACCT - learned AT-CF", "sample_auc"),
        ("pair_gate", "Learned ACCT - learned AT-CF", "sample_auc"),
    ]
    for env, comparison, metric in efficiency_specs:
        row = row_by(efficiency_stats, env=env, comparison=comparison, metric=metric)
        label = f"efficiency {env} {comparison} {metric}"
        add_expectation(expectations, label, "local_efficiency_stat_tests.csv", q(row["mean_diff"]))
        add_expectation(expectations, f"{label} interval", "local_efficiency_stat_tests.csv", interval(row))

    learned_sweep = read_rows(results / "learned_critic_sample_sweep_summary.csv")
    for env in ("delayed_lever", "pair_gate"):
        for sample_size in ("128", "512", "2048", "8192"):
            acct_row = row_by(learned_sweep, env=env, critic_samples=sample_size, method="learned_acct")
            atcf_row = row_by(learned_sweep, env=env, critic_samples=sample_size, method="learned_agent_time_cf")
            label = f"learned sweep {env} {sample_size}"
            add_expectation(
                expectations,
                f"{label} critic r2",
                "learned_critic_sample_sweep_summary.csv",
                mean_pm_fields(acct_row, "critic_r2_mean", "critic_r2_std"),
            )
            add_expectation(
                expectations,
                f"{label} atcf sample 200",
                "learned_critic_sample_sweep_summary.csv",
                mean_pm_fields(atcf_row, "sample_200_mean", "sample_200_std"),
            )
            add_expectation(
                expectations,
                f"{label} acct sample 200",
                "learned_critic_sample_sweep_summary.csv",
                mean_pm_fields(acct_row, "sample_200_mean", "sample_200_std"),
            )
            add_expectation(
                expectations,
                f"{label} greedy final",
                "learned_critic_sample_sweep_summary.csv",
                mean_pm_fields(acct_row, "greedy_final_mean", "greedy_final_std"),
            )

    perturbation_summary = read_rows(results / "influence_perturbation_summary.csv")
    perturbation_table_specs = [
        ("delayed_lever", "scale_0.25", "perturbed_agent_time_cf"),
        ("delayed_lever", "scale_0.25", "perturbed_acct"),
        ("pair_gate", "scale_0.25", "perturbed_agent_time_cf"),
        ("pair_gate", "scale_0.25", "perturbed_acct"),
        ("delayed_lever", "scale_1.00", "perturbed_agent_time_cf"),
        ("delayed_lever", "scale_1.00", "perturbed_acct"),
        ("pair_gate", "scale_1.00", "perturbed_agent_time_cf"),
        ("pair_gate", "scale_1.00", "perturbed_acct"),
        ("delayed_lever", "scale_4.00", "perturbed_agent_time_cf"),
        ("delayed_lever", "scale_4.00", "perturbed_acct"),
        ("pair_gate", "scale_4.00", "perturbed_agent_time_cf"),
        ("pair_gate", "scale_4.00", "perturbed_acct"),
        ("delayed_lever", "noise_0.10", "perturbed_agent_time_cf"),
        ("delayed_lever", "noise_0.10", "perturbed_acct"),
        ("pair_gate", "noise_0.10", "perturbed_agent_time_cf"),
        ("pair_gate", "noise_0.10", "perturbed_acct"),
        ("delayed_lever", "noise_0.25", "perturbed_agent_time_cf"),
        ("delayed_lever", "noise_0.25", "perturbed_acct"),
        ("pair_gate", "noise_0.25", "perturbed_agent_time_cf"),
        ("pair_gate", "noise_0.25", "perturbed_acct"),
    ]
    for env, perturbation, method in perturbation_table_specs:
        row = row_by(perturbation_summary, env=env, perturbation=perturbation, method=method)
        add_expectation(
            expectations,
            f"perturbation {env} {perturbation} {method}",
            "influence_perturbation_summary.csv",
            mean_pm_fields(row, "sample_200_mean", "sample_200_std"),
        )

    perturbation_stats = read_rows(results / "influence_perturbation_stat_tests.csv")
    for env, perturbation in (
        ("delayed_lever", "scale_0.25"),
        ("pair_gate", "scale_0.25"),
        ("delayed_lever", "scale_4.00"),
        ("pair_gate", "scale_4.00"),
        ("delayed_lever", "noise_0.10"),
        ("pair_gate", "noise_0.10"),
    ):
        row = row_by(
            perturbation_stats,
            env=env,
            perturbation=perturbation,
            comparison="perturbed ACCT - perturbed AT-CF",
        )
        label = f"perturbation stat {env} {perturbation}"
        add_expectation(expectations, label, "influence_perturbation_stat_tests.csv", q(row["mean_diff"]))
        add_expectation(expectations, f"{label} interval", "influence_perturbation_stat_tests.csv", interval(row))

    sensitivity = read_rows(results / "acct_sensitivity_summary.csv")
    for row in sensitivity:
        label = f"sensitivity {row['env']} {row['variant']}"
        add_expectation(
            expectations,
            label,
            "acct_sensitivity_summary.csv",
            f"{q(row['sample_200_mean'])} \\pm {q(row['sample_200_std'])}",
        )

    for source, rows, specs in [
        (
            "mpe_delayed_summary.csv",
            read_rows(results / "mpe_delayed_summary.csv"),
            [
                ("shared", "greedy_final"),
                ("shared", "sample_final"),
                ("learned_agent_time_cf", "greedy_final"),
                ("learned_agent_time_cf", "sample_final"),
                ("learned_acct", "greedy_final"),
                ("learned_acct", "sample_final"),
            ],
        ),
        (
            "mpe_ppo_summary.csv",
            read_rows(results / "mpe_ppo_summary.csv"),
            [
                ("ppo_shared", "greedy_final"),
                ("ppo_shared", "sample_final"),
                ("ppo_learned_agent_time_cf", "greedy_final"),
                ("ppo_learned_agent_time_cf", "sample_final"),
                ("ppo_learned_agent_shapley", "greedy_final"),
                ("ppo_learned_agent_shapley", "sample_final"),
                ("ppo_learned_acct", "greedy_final"),
                ("ppo_learned_acct", "sample_final"),
                ("ppo_learned_directional_acct", "greedy_final"),
                ("ppo_learned_directional_acct", "sample_final"),
            ],
        ),
        (
            "mpe_mappo_summary.csv",
            read_rows(results / "mpe_mappo_summary.csv"),
            [
                ("mappo_shared_gae", "greedy_final"),
                ("mappo_shared_gae", "sample_final"),
                ("mappo_learned_agent_time_cf", "greedy_final"),
                ("mappo_learned_agent_time_cf", "sample_final"),
                ("mappo_learned_acct", "greedy_final"),
                ("mappo_learned_acct", "sample_final"),
                ("mappo_gae_plus_agent_time_cf", "greedy_final"),
                ("mappo_gae_plus_agent_time_cf", "sample_final"),
                ("mappo_gae_plus_acct", "greedy_final"),
                ("mappo_gae_plus_acct", "sample_final"),
            ],
        ),
    ]:
        for method, metric in specs:
            row = row_by(rows, method=method, metric=metric)
            add_expectation(expectations, f"{source} {method} {metric}", source, mean_pm(row))

    mpe_stats_specs = [
        ("mpe_delayed_stat_tests.csv", "learned_acct - shared", "sample_return"),
        ("mpe_delayed_stat_tests.csv", "learned_acct - shared", "greedy_return"),
        ("mpe_ppo_stat_tests.csv", "ppo_learned_acct - ppo_shared", "sample_return"),
        ("mpe_ppo_stat_tests.csv", "ppo_learned_acct - ppo_shared", "greedy_return"),
        ("mpe_ppo_stat_tests.csv", "ppo_learned_directional_acct - ppo_shared", "sample_return"),
        ("mpe_ppo_stat_tests.csv", "ppo_learned_directional_acct - ppo_shared", "greedy_return"),
        ("mpe_mappo_stat_tests.csv", "mappo_learned_acct - mappo_shared_gae", "greedy_return"),
        ("mpe_mappo_stat_tests.csv", "mappo_gae_plus_acct - mappo_shared_gae", "greedy_return"),
        ("mpe_mappo_stat_tests.csv", "mappo_gae_plus_acct - mappo_shared_gae", "sample_return"),
    ]
    for source, comparison, metric in mpe_stats_specs:
        row = row_by(read_rows(results / source), comparison=comparison, metric=metric)
        label = f"{source} {comparison} {metric}"
        add_expectation(expectations, label, source, q(row["mean_diff"]))
        add_expectation(expectations, f"{label} interval", source, interval(row))

    for source, rows, specs in [
        (
            "mpe_counterfactual_quality_summary.csv",
            read_rows(results / "mpe_counterfactual_quality_summary.csv"),
            ["q_r2", "q_mse", "v_r2", "v_mse", "mean_abs_influence"],
        ),
        (
            "mpe_counterfactual_quality_600_summary.csv",
            read_rows(results / "mpe_counterfactual_quality_600_summary.csv"),
            ["q_r2", "v_r2", "time_abs_influence_abs_gae_corr", "signed_influence_gae_corr", "acct_credit_gae_corr"],
        ),
    ]:
        for metric in specs:
            row = row_by(rows, metric=metric)
            places = 4 if metric == "mean_abs_influence" else 3
            add_expectation(
                expectations,
                f"{source} {metric}",
                source,
                f"{q(row['mean'], places)} \\pm {q(row['std'], places)}",
            )

    localization = read_rows(results / "credit_localization.csv")
    localization_methods = sorted({row["method"] for row in localization})
    for method in localization_methods:
        rows_for_method = [row for row in localization if row["method"] == method]
        for metric in ("relevant_mass", "relevant_coverage", "irrelevant_active"):
            mean_value = sum(float(row[metric]) for row in rows_for_method) / len(rows_for_method)
            add_expectation(
                expectations,
                f"localization {method} {metric}",
                "credit_localization.csv",
                q(mean_value),
            )

    for source, rows, specs in [
        (
            "lbf_ppo_summary.csv",
            read_rows(results / "lbf_ppo_summary.csv"),
            [
                ("lbf_ppo_shared", "greedy_final"),
                ("lbf_ppo_learned_agent_time_cf", "greedy_final"),
                ("lbf_ppo_learned_acct", "greedy_final"),
            ],
        ),
        (
            "rware_ppo_summary.csv",
            read_rows(results / "rware_ppo_summary.csv"),
            [
                ("rware_ppo_shared", "greedy_final"),
                ("rware_ppo_shared", "sample_final"),
                ("rware_ppo_learned_agent_time_cf", "greedy_final"),
                ("rware_ppo_learned_acct", "greedy_final"),
            ],
        ),
    ]:
        for method, metric in specs:
            row = row_by(rows, method=method, metric=metric)
            add_expectation(expectations, f"{source} {method} {metric}", source, mean_pm(row))

    epymarl_smoke = json.loads((results / "epymarl_smoke.json").read_text())
    add_expectation(
        expectations,
        "epymarl smoke commit",
        "epymarl_smoke.json",
        epymarl_smoke["epymarl_commit"],
    )
    add_expectation(
        expectations,
        "epymarl smoke env key",
        "epymarl_smoke.json",
        epymarl_smoke["env_key"],
    )
    add_expectation(
        expectations,
        "epymarl smoke time limit",
        "epymarl_smoke.json",
        f"time\\_limit={epymarl_smoke['time_limit']}",
    )
    add_expectation(
        expectations,
        "epymarl smoke t max",
        "epymarl_smoke.json",
        f"t\\_max={epymarl_smoke['t_max']}",
    )
    epymarl_acct_smoke = json.loads((results / "epymarl_acct_smoke.json").read_text())
    add_expectation(
        expectations,
        "epymarl acct smoke mode",
        "epymarl_acct_smoke.json",
        epymarl_acct_smoke["acct_smoke_mode"].replace("_", "-"),
    )
    add_expectation(
        expectations,
        "epymarl acct smoke use flag",
        "epymarl_acct_smoke.json",
        f"use\\_acct={epymarl_acct_smoke['use_acct']}",
    )
    epymarl_qhead_smoke = json.loads((results / "epymarl_acct_qhead_smoke.json").read_text())
    add_expectation(
        expectations,
        "epymarl learned qhead smoke mode",
        "epymarl_acct_qhead_smoke.json",
        epymarl_qhead_smoke["acct_smoke_mode"].replace("_", "-"),
    )
    add_expectation(
        expectations,
        "epymarl learned qhead standardise rewards",
        "epymarl_acct_qhead_smoke.json",
        f"standardise\\_rewards={epymarl_qhead_smoke['standardise_rewards']}",
    )
    add_expectation(
        expectations,
        "epymarl learned qhead loss",
        "epymarl_acct_qhead_smoke.json",
        f"acct\\_q\\_loss={q(epymarl_qhead_smoke['final_metrics']['acct_q_loss']['value'])}",
    )
    epymarl_compare = read_rows(results / "epymarl_lbf_comparison_summary.csv")
    mappo_compare = row_by(epymarl_compare, method="epymarl_mappo")
    acct_compare = row_by(epymarl_compare, method="epymarl_acct_learned_qhead")
    add_expectation(
        expectations,
        "epymarl comparison mappo sample return",
        "epymarl_lbf_comparison_summary.csv",
        mean_pm_fields(mappo_compare, "sample_return_mean", "sample_return_std"),
    )
    add_expectation(
        expectations,
        "epymarl comparison acct sample return",
        "epymarl_lbf_comparison_summary.csv",
        mean_pm_fields(acct_compare, "sample_return_mean", "sample_return_std"),
    )
    add_expectation(
        expectations,
        "epymarl comparison acct greedy return",
        "epymarl_lbf_comparison_summary.csv",
        mean_pm_fields(acct_compare, "greedy_return_mean", "greedy_return_std"),
    )
    epymarl_compare_stats = read_rows(results / "epymarl_lbf_comparison_stat_tests.csv")
    sample_diff = row_by(
        epymarl_compare_stats,
        metric="sample_return",
        comparison="epymarl_acct_learned_qhead - epymarl_mappo",
    )
    add_expectation(
        expectations,
        "epymarl comparison sample diff",
        "epymarl_lbf_comparison_stat_tests.csv",
        q(sample_diff["mean_diff"]),
    )
    add_expectation(
        expectations,
        "epymarl comparison sample diff interval",
        "epymarl_lbf_comparison_stat_tests.csv",
        interval(sample_diff),
    )

    rows_out = []
    failed = 0
    for expectation in expectations:
        snippets = expectation.get("snippets", [expectation["snippet"]])
        ok = any(str(snippet) in paper_text for snippet in snippets)
        failed += int(not ok)
        rows_out.append(
            {
                "label": str(expectation["label"]),
                "source": str(expectation["source"]),
                "snippet": str(expectation["snippet"]),
                "status": "PASS" if ok else "FAIL",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "source", "snippet", "status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"checked {len(rows_out)} paper claims and reporting conventions")
    print(f"failed: {failed}")
    print(f"wrote {args.out}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
