"""Generate polished ACCT manuscript figures from artifact CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_medium_benchmark_suite import BENCHMARKS, summarize_benchmark, write_csv


FIG_DIR = Path("../Formatting_Instructions_For_NeurIPS_2026/Figure")

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "purple": "#7B3294",
    "gray": "#6C6C6C",
    "lightgray": "#D0D0D0",
}

METHOD_LABELS = {
    "shared": "Shared",
    "final_step_cf": "Final-step CF",
    "uniform_transport": "Uniform transport",
    "cf_transport": "CF transport",
    "return_eq_cf_transport": "Return-eq tr.",
    "agent_time_cf": "AT-CF",
    "sampled_agent_time_shapley": "Shapley",
    "acct": "ACCT",
    "learned_cf_transport": "Learned CF tr.",
    "learned_return_eq_cf_transport": "Learned ret.-eq tr.",
    "learned_agent_time_cf": "Learned AT-CF",
    "learned_acct": "Learned ACCT",
    "ppo_shared": "Shared PPO",
    "ppo_learned_agent_time_cf": "AT-CF",
    "ppo_learned_agent_shapley": "Shapley",
    "ppo_learned_acct": "ACCT",
    "ppo_learned_directional_acct": "Dir. ACCT",
    "mappo_shared_gae": "Shared GAE",
    "mappo_learned_agent_time_cf": "AT-CF",
    "mappo_learned_acct": "ACCT",
    "mappo_gae_plus_agent_time_cf": "GAE+AT-CF",
    "mappo_gae_plus_acct": "GAE+ACCT",
    "lbf_ppo_shared": "Shared PPO",
    "lbf_ppo_learned_agent_time_cf": "AT-CF",
    "lbf_ppo_learned_acct": "ACCT",
    "rware_ppo_shared": "Shared PPO",
    "rware_ppo_learned_agent_time_cf": "AT-CF",
    "rware_ppo_learned_acct": "ACCT",
}

BENCHMARK_LABELS = {
    "mpe-ppo": "MPE2 PPO",
    "mpe-mappo": "MPE2 MAPPO",
    "lbf": "LBF",
    "rware": "RWARE",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_stderr(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0) / math.sqrt(max(len(arr), 1)))


def safe_name(text: str) -> str:
    return text.replace("_", " ").replace("-", " ").title()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_method_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.35))
    ax.set_xlim(-0.45, 5.65)
    ax.set_ylim(-0.3, 3.2)
    ax.axis("off")

    times = range(5)
    agents = range(3)
    for t in times:
        ax.text(t, 2.95, f"$t={t}$", ha="center", va="center", fontsize=8)
    for i in agents:
        ax.text(-0.25, 2.35 - i * 0.8, f"agent {i+1}", ha="right", va="center", fontsize=8)

    relevant = {(0, 0), (1, 2), (2, 1)}
    for t in times:
        for i in agents:
            y = 2.35 - i * 0.8
            color = OKABE_ITO["orange"] if (t, i) in relevant else "#F7F7F7"
            edge = OKABE_ITO["orange"] if (t, i) in relevant else OKABE_ITO["lightgray"]
            ax.add_patch(Circle((t, y), 0.13, facecolor=color, edgecolor=edge, lw=1.0))
            if (t, i) in relevant:
                ax.text(t, y, "$I$", ha="center", va="center", fontsize=7, color="white")

    residual_x = 4.85
    ax.add_patch(Rectangle((residual_x - 0.25, 0.35), 0.5, 2.1, facecolor="#EEF3FB", edgecolor=OKABE_ITO["blue"], lw=1.0))
    ax.text(residual_x, 2.62, "future TD\nresiduals", ha="center", va="center", fontsize=8, color=OKABE_ITO["blue"])
    for idx, y in enumerate([2.2, 1.55, 0.9]):
        ax.text(residual_x, y, f"$\\delta_{{{idx+2}}}$", ha="center", va="center", fontsize=8, color=OKABE_ITO["blue"])

    arrow_specs = [
        ((residual_x - 0.25, 2.2), (0, 2.35), 0.95),
        ((residual_x - 0.25, 2.2), (1, 0.75), 0.58),
        ((residual_x - 0.25, 1.55), (2, 1.55), 0.78),
        ((residual_x - 0.25, 0.9), (1, 0.75), 0.72),
        ((residual_x - 0.25, 0.9), (0, 2.35), 0.36),
    ]
    for start, end, alpha in arrow_specs:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.3,
            color=OKABE_ITO["blue"],
            alpha=alpha,
            connectionstyle="arc3,rad=0.08",
        )
        ax.add_patch(arrow)

    ax.text(
        2.15,
        0.05,
        "ACCT normalizes counterfactual salience over eligible earlier agent-time pairs, then transports each residual backward.",
        ha="center",
        va="center",
        fontsize=8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_local_results(input_path: Path, out: Path, summary_path: Path) -> None:
    rows = read_rows(input_path)
    if not rows:
        raise FileNotFoundError(input_path)

    selected_methods = [
        "shared",
        "final_step_cf",
        "uniform_transport",
        "agent_time_cf",
        "sampled_agent_time_shapley",
        "acct",
        "learned_agent_time_cf",
        "learned_acct",
    ]
    colors = {
        "shared": OKABE_ITO["gray"],
        "final_step_cf": "#B8B8B8",
        "uniform_transport": OKABE_ITO["sky"],
        "agent_time_cf": OKABE_ITO["green"],
        "sampled_agent_time_shapley": OKABE_ITO["purple"],
        "acct": OKABE_ITO["orange"],
        "learned_agent_time_cf": "#4DBD8F",
        "learned_acct": OKABE_ITO["vermillion"],
    }
    linestyles = {
        "learned_agent_time_cf": "--",
        "learned_acct": "--",
        "final_step_cf": ":",
        "uniform_transport": "-.",
    }

    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: {"sample": [], "greedy": []})
    for row in rows:
        key = (row["env"], row["method"], int(row["episode"]))
        grouped[key]["sample"].append(float(row["sample_return"]))
        grouped[key]["greedy"].append(float(row["greedy_return"]))

    envs = sorted({row["env"] for row in rows})
    fig, axes = plt.subplots(2, len(envs), figsize=(7.05, 3.9), sharex=True, constrained_layout=True)
    if len(envs) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    legend_handles = []
    legend_labels = []
    summary_rows = []
    for col, env in enumerate(envs):
        for row_idx, metric in enumerate(("sample", "greedy")):
            ax = axes[row_idx, col]
            for method in selected_methods:
                episodes = sorted(ep for e, m, ep in grouped if e == env and m == method)
                if not episodes:
                    continue
                means = []
                stderrs = []
                for episode in episodes:
                    mean, stderr = mean_stderr(grouped[(env, method, episode)][metric])
                    means.append(mean)
                    stderrs.append(stderr)
                line = ax.plot(
                    episodes,
                    means,
                    color=colors[method],
                    lw=2.0 if "acct" in method else 1.35,
                    ls=linestyles.get(method, "-"),
                    label=METHOD_LABELS[method],
                )[0]
                ax.fill_between(
                    episodes,
                    np.asarray(means) - np.asarray(stderrs),
                    np.asarray(means) + np.asarray(stderrs),
                    color=colors[method],
                    alpha=0.12 if "acct" in method else 0.07,
                    linewidth=0,
                )
                if row_idx == 0 and col == 0:
                    legend_handles.append(line)
                    legend_labels.append(METHOD_LABELS[method])
                if row_idx == 0:
                    values_200 = grouped.get((env, method, 200), {"sample": []})["sample"]
                    final_values = grouped[(env, method, episodes[-1])]["greedy"]
                    if values_200 and final_values:
                        summary_rows.append(
                            {
                                "env": env,
                                "method": method,
                                "sample_200_mean": f"{np.mean(values_200):.4f}",
                                "sample_200_std": f"{np.std(values_200, ddof=0):.4f}",
                                "greedy_final_mean": f"{np.mean(final_values):.4f}",
                                "greedy_final_std": f"{np.std(final_values, ddof=0):.4f}",
                                "seeds": str(len(final_values)),
                                "final_episode": str(episodes[-1]),
                            }
                        )

            ax.grid(alpha=0.18, linewidth=0.6)
            if row_idx == 0:
                ax.set_title(safe_name(env))
            if col == 0:
                ax.set_ylabel("Sampled return" if metric == "sample" else "Greedy return")
            if row_idx == 1:
                ax.set_xlabel("Training episode")
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        columnspacing=1.0,
        handlelength=1.8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    write_csv(
        summary_path,
        summary_rows,
        ["env", "method", "sample_200_mean", "sample_200_std", "greedy_final_mean", "greedy_final_std", "seeds", "final_episode"],
    )


def ensure_medium_summary(summary_path: Path, stats_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_rows = read_rows(summary_path)
    stat_rows = read_rows(stats_path)
    if summary_rows and stat_rows:
        return summary_rows, stat_rows

    all_summary: list[dict[str, str]] = []
    all_stats: list[dict[str, str]] = []
    for benchmark in BENCHMARKS.values():
        if not benchmark.results.exists():
            continue
        s_rows, st_rows = summarize_benchmark(benchmark, benchmark.results, bootstrap_samples=10_000, seed=20260601)
        all_summary.extend(s_rows)
        all_stats.extend(st_rows)
    if all_summary:
        write_csv(
            summary_path,
            all_summary,
            ["benchmark", "env", "method", "metric", "mean", "std", "seeds", "final_episode", "threshold", "first_threshold_median"],
        )
        write_csv(
            stats_path,
            all_stats,
            ["benchmark", "env", "metric", "comparison", "mean_diff", "ci95_low", "ci95_high", "win_rate", "paired_seeds", "final_episode"],
        )
    return all_summary, all_stats


def row_lookup(rows: list[dict[str, str]], **query: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in query.items()):
            return row
    return None


def plot_medium_benchmarks(summary_path: Path, stats_path: Path, out: Path) -> None:
    summary_rows, stat_rows = ensure_medium_summary(summary_path, stats_path)
    if not summary_rows:
        raise FileNotFoundError("medium benchmark summary is missing; run run_medium_benchmark_suite.py first")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.95), constrained_layout=True)

    def color_for(method: str) -> str:
        if "shared" in method:
            return OKABE_ITO["gray"]
        if "shapley" in method:
            return OKABE_ITO["purple"]
        if "directional" in method:
            return OKABE_ITO["vermillion"]
        if "gae_plus_acct" in method:
            return OKABE_ITO["blue"]
        if "acct" in method:
            return OKABE_ITO["orange"]
        if "agent_time" in method:
            return OKABE_ITO["green"]
        return OKABE_ITO["sky"]

    def metric_value(benchmark_name: str, method: str, metric: str) -> tuple[float, float, int]:
        row = row_lookup(summary_rows, benchmark=benchmark_name, method=method, metric=metric)
        if row is None:
            return float("nan"), float("nan"), 0
        return float(row["mean"]), float(row["std"]), int(row["final_episode"])

    def win_text(benchmark_name: str, method: str) -> str:
        final_stat = row_lookup(stat_rows, benchmark=benchmark_name, metric="final_sample", comparison=f"{method} - {baseline_for(benchmark_name)}")
        auc_stat = row_lookup(stat_rows, benchmark=benchmark_name, metric="sample_auc", comparison=f"{method} - {baseline_for(benchmark_name)}")
        if final_stat is None and auc_stat is None:
            return "base"
        if final_stat is None or auc_stat is None:
            return "n/a"
        return f"{float(final_stat['win_rate']):.2f}/{float(auc_stat['win_rate']):.2f}"

    def baseline_for(benchmark_name: str) -> str:
        for row in stat_rows:
            if row["benchmark"] == benchmark_name and " - " in row["comparison"]:
                return row["comparison"].split(" - ", maxsplit=1)[1]
        rows = [row for row in summary_rows if row["benchmark"] == benchmark_name and row["metric"] == "final_sample"]
        return rows[0]["method"] if rows else ""

    for ax, benchmark_name in zip(axes.flat, BENCHMARK_LABELS):
        rows_for_benchmark = [row for row in summary_rows if row["benchmark"] == benchmark_name and row["metric"] == "final_sample"]
        if not rows_for_benchmark:
            ax.axis("off")
            continue
        methods = [row["method"] for row in rows_for_benchmark]
        y = np.arange(len(methods))[::-1]

        all_values: list[float] = []
        final_episode = 0
        for yi, method in zip(y, methods):
            final_mean, final_std, episode = metric_value(benchmark_name, method, "final_sample")
            auc_mean, auc_std, _ = metric_value(benchmark_name, method, "sample_auc")
            final_episode = max(final_episode, episode)
            all_values.extend([final_mean - final_std, final_mean + final_std, auc_mean - auc_std, auc_mean + auc_std])
            color = color_for(method)
            ax.barh(
                yi,
                final_mean,
                xerr=final_std,
                height=0.48,
                color=color,
                edgecolor="#303030" if "acct" in method else "white",
                linewidth=0.35 if "acct" in method else 0.0,
                alpha=0.88,
                capsize=1.5,
                zorder=2,
            )
            ax.scatter(
                auc_mean,
                yi,
                marker="D",
                s=23,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                zorder=4,
            )
            ax.text(
                1.015,
                yi,
                win_text(benchmark_name, method),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=6.5,
                color="#333333",
                clip_on=False,
            )

        finite_values = [value for value in all_values if math.isfinite(value)]
        if finite_values:
            lo, hi = min(finite_values), max(finite_values)
            pad = max((hi - lo) * 0.12, 0.02 if hi <= 0 else 0.003)
            ax.set_xlim(lo - pad, hi + pad)
        ax.axvline(0.0, color="#B0B0B0", lw=0.8, zorder=1)
        ax.grid(axis="x", alpha=0.18, linewidth=0.6)
        ax.set_title(f"{BENCHMARK_LABELS[benchmark_name]} ({final_episode} eps)")
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods])
        ax.text(1.015, 1.02, "win F/A", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color="#555555")
        if ax in axes[1, :]:
            ax.set_xlabel("Return (higher is better)")

    out.parent.mkdir(parents=True, exist_ok=True)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=OKABE_ITO["gray"], edgecolor="none", alpha=0.88, label="Final sampled return"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#404040", markersize=4.5, label="Sample-return AUC"),
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=2, columnspacing=1.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_medium_summary_table(summary_path: Path, stats_path: Path, out: Path) -> None:
    summary_rows, stat_rows = ensure_medium_summary(summary_path, stats_path)
    if not summary_rows:
        raise FileNotFoundError("medium benchmark summary is missing; run run_medium_benchmark_suite.py first")

    def baseline_for(benchmark_name: str) -> str:
        for row in stat_rows:
            if row["benchmark"] == benchmark_name and " - " in row["comparison"]:
                return row["comparison"].split(" - ", maxsplit=1)[1]
        rows = [row for row in summary_rows if row["benchmark"] == benchmark_name and row["metric"] == "final_sample"]
        return rows[0]["method"] if rows else ""

    def metric_cell(benchmark_name: str, method: str, metric: str) -> str:
        row = row_lookup(summary_rows, benchmark=benchmark_name, method=method, metric=metric)
        if row is None:
            return "--"
        return f"${float(row['mean']):.3f} \\pm {float(row['std']):.3f}$"

    def win_cell(benchmark_name: str, method: str) -> str:
        baseline = baseline_for(benchmark_name)
        if method == baseline:
            return "base"
        final_stat = row_lookup(stat_rows, benchmark=benchmark_name, metric="final_sample", comparison=f"{method} - {baseline}")
        auc_stat = row_lookup(stat_rows, benchmark=benchmark_name, metric="sample_auc", comparison=f"{method} - {baseline}")
        if final_stat is None or auc_stat is None:
            return "--"
        return f"{float(final_stat['win_rate']):.2f}/{float(auc_stat['win_rate']):.2f}"

    lines = [
        "% Auto-generated by experiments/plot_paper_figures.py; do not edit by hand.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Medium benchmark summary generated from \\texttt{medium\\_benchmark\\_summary.csv} and \\texttt{medium\\_benchmark\\_stat\\_tests.csv}. Values are mean $\\pm$ standard deviation over 16 seeds; Win F/A reports seed-wise win rate for final sampled return and sampled-return AUC against the shared baseline.}",
        "\\label{tab:medium-summary}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Benchmark & Method & Final sampled & Final greedy & Sample AUC & Win F/A \\\\",
        "\\midrule",
    ]
    first_group = True
    for benchmark_name in BENCHMARK_LABELS:
        rows_for_benchmark = [row for row in summary_rows if row["benchmark"] == benchmark_name and row["metric"] == "final_sample"]
        if not rows_for_benchmark:
            continue
        if not first_group:
            lines.append("\\addlinespace[2pt]")
        first_group = False
        for idx, row in enumerate(rows_for_benchmark):
            method = row["method"]
            benchmark_label = BENCHMARK_LABELS[benchmark_name] if idx == 0 else ""
            method_label = METHOD_LABELS.get(method, method).replace("_", "\\_")
            lines.append(
                f"{benchmark_label} & {method_label} & "
                f"{metric_cell(benchmark_name, method, 'final_sample')} & "
                f"{metric_cell(benchmark_name, method, 'final_greedy')} & "
                f"{metric_cell(benchmark_name, method, 'sample_auc')} & "
                f"{win_cell(benchmark_name, method)} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def plot_diagnostics_appendix(out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65), constrained_layout=True)

    localization = read_rows(Path("results/credit_localization.csv"))
    selected = ["shared", "agent_time_cf", "sampled_agent_time_shapley", "acct", "learned_agent_time_cf", "learned_acct"]
    envs = sorted({row["env"] for row in localization})
    matrix = np.full((len(selected), len(envs)), np.nan)
    for r, method in enumerate(selected):
        for c, env in enumerate(envs):
            row = row_lookup(localization, env=env, method=method)
            if row:
                matrix[r, c] = float(row["relevant_mass"])
    ax = axes[0, 0]
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_title("Credit localization")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([safe_name(env) for env in envs])
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in selected])
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", color="white" if matrix[r, c] < 0.65 else "black", fontsize=6.5)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, label="relevant mass")

    sweep = read_rows(Path("results/learned_critic_sample_sweep_summary.csv"))
    ax = axes[0, 1]
    for env, marker in [("delayed_lever", "o"), ("pair_gate", "s")]:
        for method, color in [("learned_agent_time_cf", OKABE_ITO["green"]), ("learned_acct", OKABE_ITO["orange"])]:
            rows = sorted(
                [row for row in sweep if row["env"] == env and row["method"] == method],
                key=lambda row: int(row["critic_samples"]),
            )
            if rows:
                ax.plot(
                    [int(row["critic_samples"]) for row in rows],
                    [float(row["sample_200_mean"]) for row in rows],
                    marker=marker,
                    color=color,
                    lw=1.45,
                    label=f"{'Lever' if env == 'delayed_lever' else 'Gate'} {METHOD_LABELS[method].replace('Learned ', '')}",
                )
    ax.set_xscale("log", base=2)
    ax.set_title("Critic data sweep")
    ax.set_xlabel("uniform critic-fit samples")
    ax.set_ylabel("Sample@200")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower right")

    perturbation = read_rows(Path("results/influence_perturbation_summary.csv"))
    ax = axes[1, 0]
    order = ["scale_0.25", "scale_1.00", "scale_4.00", "noise_0.10", "noise_0.25"]
    for method, color in [("perturbed_agent_time_cf", OKABE_ITO["green"]), ("perturbed_acct", OKABE_ITO["orange"])]:
        values = []
        for perturb in order:
            rows = [row for row in perturbation if row["env"] == "pair_gate" and row["perturbation"] == perturb and row["method"] == method]
            values.append(float(rows[0]["sample_200_mean"]) if rows else np.nan)
        ax.plot(range(len(order)), values, marker="o", color=color, lw=1.6, label="AT-CF" if "agent" in method else "ACCT")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["scale .25", "scale 1", "scale 4", "noise .10", "noise .25"], rotation=20)
    ax.set_title("Influence perturbation")
    ax.set_ylabel("Pair Gate Sample@200")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False)

    distractor = read_rows(Path("results/distractor_scaling_summary.csv"))
    ax = axes[1, 1]
    for method, color in [("shared", OKABE_ITO["gray"]), ("agent_time_cf", OKABE_ITO["green"]), ("acct", OKABE_ITO["orange"])]:
        rows = sorted([row for row in distractor if row["method"] == method], key=lambda row: int(row["total_pairs"]))
        if rows:
            ax.errorbar(
                [int(row["total_pairs"]) for row in rows],
                [float(row["sample_mean"]) for row in rows],
                yerr=[float(row["sample_std"]) for row in rows],
                marker="o",
                lw=1.5,
                capsize=1.5,
                color=color,
                label=METHOD_LABELS[method],
            )
    ax.set_title("Distractor scaling")
    ax.set_xlabel("agent-time pairs")
    ax.set_ylabel("Sample@200")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, figure_paths: list[Path]) -> None:
    rows = [{"figure": figure.as_posix(), "exists": str(figure.exists())} for figure in figure_paths]
    write_csv(path, rows, ["figure", "exists"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--local-input", type=Path, default=Path("results/local_experiments.csv"))
    parser.add_argument("--local-summary", type=Path, default=Path("results/local_polished_summary.csv"))
    parser.add_argument("--medium-summary", type=Path, default=Path("results/medium_benchmark_summary.csv"))
    parser.add_argument("--medium-stats", type=Path, default=Path("results/medium_benchmark_stat_tests.csv"))
    parser.add_argument("--medium-table", type=Path, default=FIG_DIR / "acct_medium_summary_table.tex")
    parser.add_argument("--manifest", type=Path, default=Path("results/figure_manifest.csv"))
    args = parser.parse_args()

    configure_style()
    figure_paths = [
        args.figure_dir / "acct_method_schematic.pdf",
        args.figure_dir / "acct_local_results_polished.pdf",
        args.figure_dir / "acct_medium_benchmarks.pdf",
        args.figure_dir / "acct_diagnostics_appendix.pdf",
        args.medium_table,
    ]
    plot_method_schematic(figure_paths[0])
    plot_local_results(args.local_input, figure_paths[1], args.local_summary)
    plot_medium_benchmarks(args.medium_summary, args.medium_stats, figure_paths[2])
    plot_diagnostics_appendix(figure_paths[3])
    write_medium_summary_table(args.medium_summary, args.medium_stats, args.medium_table)
    write_manifest(args.manifest, figure_paths)
    for figure in figure_paths:
        print(f"wrote {figure}")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
