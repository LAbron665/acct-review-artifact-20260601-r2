"""Run the medium ACCT benchmark suite and write unified summaries.

The suite intentionally stays laptop-scale: MPE2 PPO/MAPPO at 600 episodes
and LBF/RWARE at 1000 episodes. Raw logs are written to separate ``medium_*``
CSV files so the earlier smoke-test evidence remains auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Benchmark:
    name: str
    script: str
    results: Path
    summary: Path
    stats: Path
    episodes: int
    seeds: int
    baseline: str
    methods: tuple[str, ...]
    args: tuple[str, ...]
    threshold: float
    greedy_eval_per_row: int


BENCHMARKS = {
    "mpe-ppo": Benchmark(
        name="mpe-ppo",
        script="experiments/run_mpe_ppo_experiment.py",
        results=Path("results/medium_mpe_ppo_results.csv"),
        summary=Path("results/medium_mpe_ppo_summary.csv"),
        stats=Path("results/medium_mpe_ppo_stat_tests.csv"),
        episodes=600,
        seeds=16,
        baseline="ppo_shared",
        methods=(
            "ppo_shared",
            "ppo_learned_agent_time_cf",
            "ppo_learned_agent_shapley",
            "ppo_learned_acct",
            "ppo_learned_directional_acct",
        ),
        args=("--max-cycles", "25", "--batch-size", "8", "--eval-every", "100", "--ppo-epochs", "3"),
        threshold=-0.95,
        greedy_eval_per_row=5,
    ),
    "mpe-mappo": Benchmark(
        name="mpe-mappo",
        script="experiments/run_mpe_mappo_experiment.py",
        results=Path("results/medium_mpe_mappo_results.csv"),
        summary=Path("results/medium_mpe_mappo_summary.csv"),
        stats=Path("results/medium_mpe_mappo_stat_tests.csv"),
        episodes=600,
        seeds=16,
        baseline="mappo_shared_gae",
        methods=(
            "mappo_shared_gae",
            "mappo_learned_agent_time_cf",
            "mappo_learned_acct",
            "mappo_gae_plus_agent_time_cf",
            "mappo_gae_plus_acct",
        ),
        args=(
            "--max-cycles",
            "25",
            "--batch-size",
            "8",
            "--eval-every",
            "100",
            "--ppo-epochs",
            "3",
            "--hybrid-weight",
            "0.05",
        ),
        threshold=-0.95,
        greedy_eval_per_row=5,
    ),
    "lbf": Benchmark(
        name="lbf",
        script="experiments/run_lbf_ppo_experiment.py",
        results=Path("results/medium_lbf_ppo_results.csv"),
        summary=Path("results/medium_lbf_ppo_summary.csv"),
        stats=Path("results/medium_lbf_ppo_stat_tests.csv"),
        episodes=1000,
        seeds=16,
        baseline="lbf_ppo_shared",
        methods=("lbf_ppo_shared", "lbf_ppo_learned_agent_time_cf", "lbf_ppo_learned_acct"),
        args=("--max-cycles", "25", "--batch-size", "8", "--eval-every", "100", "--ppo-epochs", "3"),
        threshold=0.10,
        greedy_eval_per_row=8,
    ),
    "rware": Benchmark(
        name="rware",
        script="experiments/run_rware_ppo_experiment.py",
        results=Path("results/medium_rware_ppo_results.csv"),
        summary=Path("results/medium_rware_ppo_summary.csv"),
        stats=Path("results/medium_rware_ppo_stat_tests.csv"),
        episodes=1000,
        seeds=16,
        baseline="rware_ppo_shared",
        methods=("rware_ppo_shared", "rware_ppo_learned_agent_time_cf", "rware_ppo_learned_acct"),
        args=(
            "--env-id",
            "rware-tiny-2ag-easy-v2",
            "--max-steps",
            "100",
            "--batch-size",
            "8",
            "--eval-every",
            "100",
            "--ppo-epochs",
            "3",
        ),
        threshold=0.01,
        greedy_eval_per_row=8,
    ),
}


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, object]:
    start = time.time()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 200:
            tail = tail[-200:]
    returncode = proc.wait()
    return {
        "command": " ".join(command),
        "returncode": returncode,
        "seconds": round(time.time() - start, 2),
        "output_tail": "".join(tail)[-4000:],
    }


def arg_value(args: tuple[str, ...], flag: str, default: str) -> str:
    if flag not in args:
        return default
    idx = args.index(flag)
    return args[idx + 1]


def train_one_seed(benchmark_name: str, method: str, seed: int, benchmark: Benchmark) -> list[dict[str, float | int | str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    if benchmark_name == "mpe-ppo":
        from experiments.run_mpe_ppo_experiment import train_method

        return train_method(
            method,  # type: ignore[arg-type]
            seed=seed,
            episodes=benchmark.episodes,
            max_cycles=int(arg_value(benchmark.args, "--max-cycles", "25")),
            batch_size=int(arg_value(benchmark.args, "--batch-size", "8")),
            eval_every=int(arg_value(benchmark.args, "--eval-every", "100")),
            ppo_epochs=int(arg_value(benchmark.args, "--ppo-epochs", "3")),
        )
    if benchmark_name == "mpe-mappo":
        from experiments.run_mpe_mappo_experiment import train_method

        return train_method(
            method,  # type: ignore[arg-type]
            seed=seed,
            episodes=benchmark.episodes,
            max_cycles=int(arg_value(benchmark.args, "--max-cycles", "25")),
            batch_size=int(arg_value(benchmark.args, "--batch-size", "8")),
            eval_every=int(arg_value(benchmark.args, "--eval-every", "100")),
            ppo_epochs=int(arg_value(benchmark.args, "--ppo-epochs", "3")),
            gamma=0.99,
            gae_lam=0.95,
            hybrid_weight=float(arg_value(benchmark.args, "--hybrid-weight", "0.05")),
        )
    if benchmark_name == "lbf":
        from experiments.run_lbf_ppo_experiment import train_method

        return train_method(
            method,  # type: ignore[arg-type]
            seed=seed,
            episodes=benchmark.episodes,
            max_cycles=int(arg_value(benchmark.args, "--max-cycles", "25")),
            batch_size=int(arg_value(benchmark.args, "--batch-size", "8")),
            eval_every=int(arg_value(benchmark.args, "--eval-every", "100")),
            ppo_epochs=int(arg_value(benchmark.args, "--ppo-epochs", "3")),
        )
    if benchmark_name == "rware":
        from experiments.run_rware_ppo_experiment import train_method

        return train_method(
            method,  # type: ignore[arg-type]
            seed=seed,
            env_id=arg_value(benchmark.args, "--env-id", "rware-tiny-2ag-easy-v2"),
            episodes=benchmark.episodes,
            max_steps=int(arg_value(benchmark.args, "--max-steps", "100")),
            batch_size=int(arg_value(benchmark.args, "--batch-size", "8")),
            eval_every=int(arg_value(benchmark.args, "--eval-every", "100")),
            ppo_epochs=int(arg_value(benchmark.args, "--ppo-epochs", "3")),
        )
    raise ValueError(benchmark_name)


def run_benchmark_parallel(benchmark: Benchmark, *, max_workers: int) -> dict[str, object]:
    start = time.time()
    fieldnames = ["env", "method", "seed", "episode", "sample_return", "greedy_return", "baseline"]
    benchmark.results.parent.mkdir(parents=True, exist_ok=True)
    commands = []
    failures: list[str] = []
    with benchmark.results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(train_one_seed, benchmark.name, method, seed, benchmark): (method, seed)
                for method in benchmark.methods
                for seed in range(benchmark.seeds)
            }
            for future in as_completed(future_to_job):
                method, seed = future_to_job[future]
                try:
                    rows = future.result()
                except Exception as exc:  # noqa: BLE001 - record subprocess-style failure detail.
                    failures.append(f"{method} seed={seed}: {exc}")
                    print(f"failed {benchmark.name} {method} seed={seed}: {exc}", flush=True)
                    continue
                writer.writerows(rows)
                f.flush()
                print(f"finished {benchmark.name} {method} seed={seed} rows={len(rows)}", flush=True)
                commands.append({"method": method, "seed": seed, "rows": len(rows)})
    return {
        "command": f"parallel {benchmark.name} workers={max_workers}",
        "returncode": 0 if not failures else 1,
        "seconds": round(time.time() - start, 2),
        "output_tail": "\n".join(failures[-20:]) if failures else f"wrote {benchmark.results}",
        "jobs": commands,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def existing_is_complete(path: Path, benchmark: Benchmark) -> bool:
    rows = read_rows(path)
    if not rows:
        return False
    methods = {row["method"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    max_episode = max(int(row["episode"]) for row in rows)
    return (
        set(benchmark.methods).issubset(methods)
        and len(seeds) >= benchmark.seeds
        and max_episode >= benchmark.episodes
    )


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if len(diffs) == 0:
        return float("nan"), float("nan")
    draws = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_benchmark(
    benchmark: Benchmark,
    results_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_rows(results_path)
    if not rows:
        return [], []

    envs = sorted({row["env"] for row in rows})
    methods = [method for method in benchmark.methods if method in {row["method"] for row in rows}]
    final_episode = max(int(row["episode"]) for row in rows)
    rng = np.random.default_rng(seed)

    by_series: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["env"], row["method"], int(row["seed"]))
        by_series.setdefault(key, []).append(row)

    summary_rows: list[dict[str, str]] = []
    per_seed: dict[tuple[str, str, int], dict[str, float]] = {}
    for env in envs:
        for method in methods:
            seed_ids = sorted(
                seed_id
                for series_env, series_method, seed_id in by_series
                if series_env == env and series_method == method
            )
            final_sample = []
            final_greedy = []
            sample_auc = []
            greedy_auc = []
            first_threshold = []
            for seed_id in seed_ids:
                series = sorted(by_series[(env, method, seed_id)], key=lambda row: int(row["episode"]))
                episodes = np.asarray([int(row["episode"]) for row in series], dtype=np.float64)
                sample = np.asarray([float(row["sample_return"]) for row in series], dtype=np.float64)
                greedy = np.asarray([float(row["greedy_return"]) for row in series], dtype=np.float64)
                final_sample.append(float(sample[-1]))
                final_greedy.append(float(greedy[-1]))
                if len(episodes) > 1:
                    denom = max(float(episodes[-1] - episodes[0]), 1.0)
                    sample_auc.append(float(np.trapezoid(sample, episodes) / denom))
                    greedy_auc.append(float(np.trapezoid(greedy, episodes) / denom))
                else:
                    sample_auc.append(float(sample[-1]))
                    greedy_auc.append(float(greedy[-1]))
                reached = episodes[sample >= benchmark.threshold]
                first_threshold.append(float(reached[0]) if len(reached) else float("nan"))
                per_seed[(env, method, seed_id)] = {
                    "final_sample": float(sample[-1]),
                    "final_greedy": float(greedy[-1]),
                    "sample_auc": float(sample_auc[-1]),
                    "greedy_auc": float(greedy_auc[-1]),
                }

            for metric_name, values in (
                ("final_sample", final_sample),
                ("final_greedy", final_greedy),
                ("sample_auc", sample_auc),
                ("greedy_auc", greedy_auc),
            ):
                arr = np.asarray(values, dtype=np.float64)
                summary_rows.append(
                    {
                        "benchmark": benchmark.name,
                        "env": env,
                        "method": method,
                        "metric": metric_name,
                        "mean": f"{arr.mean():.4f}",
                        "std": f"{arr.std(ddof=0):.4f}",
                        "seeds": str(len(arr)),
                        "final_episode": str(final_episode),
                        "threshold": f"{benchmark.threshold:.4f}",
                        "first_threshold_median": f"{np.nanmedian(first_threshold):.1f}"
                        if not np.all(np.isnan(first_threshold))
                        else "nan",
                    }
                )

    stat_rows: list[dict[str, str]] = []
    for env in envs:
        baseline_seeds = {
            seed_id
            for series_env, series_method, seed_id in by_series
            if series_env == env and series_method == benchmark.baseline
        }
        for method in methods:
            if method == benchmark.baseline:
                continue
            method_seeds = {
                seed_id
                for series_env, series_method, seed_id in by_series
                if series_env == env and series_method == method
            }
            paired_seeds = sorted(baseline_seeds & method_seeds)
            for metric in ("final_sample", "final_greedy", "sample_auc", "greedy_auc"):
                diffs = np.asarray(
                    [
                        per_seed[(env, method, seed_id)][metric]
                        - per_seed[(env, benchmark.baseline, seed_id)][metric]
                        for seed_id in paired_seeds
                    ],
                    dtype=np.float64,
                )
                ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng, samples=bootstrap_samples)
                stat_rows.append(
                    {
                        "benchmark": benchmark.name,
                        "env": env,
                        "metric": metric,
                        "comparison": f"{method} - {benchmark.baseline}",
                        "mean_diff": f"{diffs.mean():.4f}" if len(diffs) else "nan",
                        "ci95_low": f"{ci_low:.4f}",
                        "ci95_high": f"{ci_high:.4f}",
                        "win_rate": f"{np.mean(diffs > 0):.4f}" if len(diffs) else "nan",
                        "paired_seeds": str(len(paired_seeds)),
                        "final_episode": str(final_episode),
                    }
                )
    return summary_rows, stat_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(BENCHMARKS),
        default=["mpe-ppo", "mpe-mappo", "lbf", "rware"],
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--subprocess", action="store_true", help="run each benchmark through its standalone script")
    parser.add_argument("--report", type=Path, default=Path("results/medium_benchmark_suite_report.json"))
    parser.add_argument("--summary", type=Path, default=Path("results/medium_benchmark_summary.csv"))
    parser.add_argument("--stats", type=Path, default=Path("results/medium_benchmark_stat_tests.csv"))
    args = parser.parse_args()

    root = Path(".").resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    commands: list[dict[str, object]] = []
    selected = [BENCHMARKS[name] for name in args.only]
    for benchmark in selected:
        if args.skip_existing and existing_is_complete(benchmark.results, benchmark):
            commands.append(
                {
                    "benchmark": benchmark.name,
                    "command": "skip-existing",
                    "returncode": 0,
                    "seconds": 0.0,
                    "output_tail": f"{benchmark.results} already has {benchmark.episodes} episodes and {benchmark.seeds} seeds",
                }
            )
        else:
            if args.subprocess:
                command = [
                    str(args.python),
                    benchmark.script,
                    "--episodes",
                    str(benchmark.episodes),
                    "--seeds",
                    str(benchmark.seeds),
                    "--out",
                    str(benchmark.results),
                    *benchmark.args,
                ]
                result = run_command(command, cwd=root, env=env)
            else:
                result = run_benchmark_parallel(benchmark, max_workers=args.workers)
            result["benchmark"] = benchmark.name
            commands.append(result)
            if result["returncode"] != 0:
                break

        summary_command = [
            str(args.python),
            "experiments/summarize_mpe_results.py",
            "--input",
            str(benchmark.results),
            "--summary",
            str(benchmark.summary),
            "--methods",
            *benchmark.methods,
        ]
        summary_result = run_command(summary_command, cwd=root, env=env)
        summary_result["benchmark"] = f"{benchmark.name}-summary"
        commands.append(summary_result)
        if summary_result["returncode"] != 0:
            break

        stats_command = [
            str(args.python),
            "experiments/analyze_mpe_statistics.py",
            "--input",
            str(benchmark.results),
            "--out",
            str(benchmark.stats),
            "--baseline-method",
            benchmark.baseline,
            "--episode",
            str(benchmark.episodes),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
        ]
        stats_result = run_command(stats_command, cwd=root, env=env)
        stats_result["benchmark"] = f"{benchmark.name}-paired-stats"
        commands.append(stats_result)
        if stats_result["returncode"] != 0:
            break

    all_summary_rows: list[dict[str, str]] = []
    all_stat_rows: list[dict[str, str]] = []
    for benchmark in selected:
        summary_rows, stat_rows = summarize_benchmark(
            benchmark,
            benchmark.results,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        all_summary_rows.extend(summary_rows)
        all_stat_rows.extend(stat_rows)

    write_csv(
        args.summary,
        all_summary_rows,
        [
            "benchmark",
            "env",
            "method",
            "metric",
            "mean",
            "std",
            "seeds",
            "final_episode",
            "threshold",
            "first_threshold_median",
        ],
    )
    write_csv(
        args.stats,
        all_stat_rows,
        [
            "benchmark",
            "env",
            "metric",
            "comparison",
            "mean_diff",
            "ci95_low",
            "ci95_high",
            "win_rate",
            "paired_seeds",
            "final_episode",
        ],
    )

    failures = [result for result in commands if result["returncode"] != 0]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "selected": [benchmark.name for benchmark in selected],
        "summary": str(args.summary),
        "stats": str(args.stats),
        "commands": commands,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"status: {report['status']}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.stats}")
    print(f"wrote {args.report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
