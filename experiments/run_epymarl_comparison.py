"""Run a tiny paired EPyMARL MAPPO-vs-ACCT comparison on LBF.

The script is intentionally conservative: it uses an external EPyMARL checkout,
applies the local ACCT smoke patch, runs a small fixed-budget MAPPO baseline and
learned-Q-head ACCT variant for matched seeds, and writes anonymous CSV files.
It is not a benchmark-quality training campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from frameworks.patch_epymarl_acct import apply_patch_to_epymarl


DEFAULT_ENV_KEY = "lbforaging:Foraging-5x5-2p-1f-v3"
METHODS = (
    {
        "method": "epymarl_mappo",
        "use_acct": False,
        "acct_influence_source": "",
    },
    {
        "method": "epymarl_acct_learned_qhead",
        "use_acct": True,
        "acct_influence_source": "learned_q_head",
    },
)


def latest_run_dir(sacred_root: Path, start_time: float) -> Path:
    runs = [
        path.parent
        for path in sacred_root.rglob("run.json")
        if path.parent.is_dir()
        and path.parent.name.isdigit()
        and path.stat().st_mtime >= start_time - 1.0
    ]
    if not runs:
        raise FileNotFoundError(f"no new Sacred run.json files under {sacred_root}")
    return max(runs, key=lambda path: (path / "run.json").stat().st_mtime)


def final_metric(metrics: dict[str, dict[str, list[float]]], name: str) -> tuple[int | None, float | None]:
    item = metrics.get(name)
    if not item or not item.get("values"):
        return None, None
    return item["steps"][-1], item["values"][-1]


def csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_one(
    *,
    epymarl_root: Path,
    method: dict[str, object],
    seed: int,
    env_key: str,
    time_limit: int,
    t_max: int,
    test_nepisode: int,
    standardise_rewards: str,
    acct_lambda: float,
    acct_residual_weight: float,
    acct_transport_mode: str,
    acct_standardize: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        "src/main.py",
        "--config=mappo",
        "--env-config=gymma",
        "with",
        f"env_args.time_limit={time_limit}",
        f"env_args.key={env_key}",
        f"t_max={t_max}",
        f"seed={seed}",
        f"test_nepisode={test_nepisode}",
        "test_interval=10",
        "log_interval=10",
        "runner_log_interval=10",
        "learner_log_interval=10",
        "use_cuda=False",
        f"standardise_rewards={standardise_rewards}",
        "batch_size_run=1",
        "batch_size=1",
        "buffer_size=1",
        "save_model=False",
        "use_tensorboard=False",
    ]
    if method["use_acct"]:
        command.extend(
            [
                "use_acct=True",
                f"acct_lambda={acct_lambda}",
                f"acct_residual_weight={acct_residual_weight}",
                f"acct_transport_mode={acct_transport_mode}",
                f"acct_standardize={acct_standardize}",
                f"acct_influence_source={method['acct_influence_source']}",
            ]
        )

    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    start_time = time.time()
    proc = subprocess.run(
        command,
        cwd=epymarl_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    run_dir = latest_run_dir(epymarl_root / "results" / "sacred", start_time)
    run = json.loads((run_dir / "run.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    repo = run.get("experiment", {}).get("repositories", [{}])[0]

    return_step, return_mean = final_metric(metrics, "return_mean")
    test_step, test_return_mean = final_metric(metrics, "test_return_mean")
    ep_len_step, ep_length_mean = final_metric(metrics, "ep_length_mean")
    test_ep_len_step, test_ep_length_mean = final_metric(metrics, "test_ep_length_mean")
    _, critic_loss = final_metric(metrics, "critic_loss")
    _, td_error_abs = final_metric(metrics, "td_error_abs")
    _, pg_loss = final_metric(metrics, "pg_loss")
    _, agent_grad_norm = final_metric(metrics, "agent_grad_norm")
    _, critic_grad_norm = final_metric(metrics, "critic_grad_norm")
    _, acct_q_loss = final_metric(metrics, "acct_q_loss")
    _, acct_q_grad_norm = final_metric(metrics, "acct_q_grad_norm")

    status = "PASS" if proc.returncode == 0 and run.get("status") == "COMPLETED" else "FAIL"
    return {
        "status": status,
        "script_returncode": proc.returncode,
        "epymarl_repository": "https://github.com/uoe-agents/epymarl.git",
        "epymarl_commit": repo.get("commit"),
        "algorithm": config.get("name"),
        "learner": config.get("learner"),
        "env": config.get("env"),
        "env_key": config.get("env_args", {}).get("key"),
        "time_limit": config.get("env_args", {}).get("time_limit"),
        "episode": config.get("t_max"),
        "t_max": config.get("t_max"),
        "seed": config.get("seed"),
        "method": method["method"],
        "standardise_rewards": config.get("standardise_rewards"),
        "use_acct": bool(config.get("use_acct", False)),
        "acct_influence_source": config.get("acct_influence_source") if config.get("use_acct", False) else "",
        "acct_lambda": config.get("acct_lambda") if config.get("use_acct", False) else "",
        "acct_residual_weight": config.get("acct_residual_weight") if config.get("use_acct", False) else "",
        "acct_transport_mode": config.get("acct_transport_mode") if config.get("use_acct", False) else "",
        "acct_standardize": config.get("acct_standardize") if config.get("use_acct", False) else "",
        "return_step": return_step,
        "sample_return": return_mean,
        "test_return_step": test_step,
        "greedy_return": test_return_mean,
        "ep_length_step": ep_len_step,
        "ep_length_mean": ep_length_mean,
        "test_ep_length_step": test_ep_len_step,
        "test_ep_length_mean": test_ep_length_mean,
        "critic_loss": critic_loss,
        "td_error_abs": td_error_abs,
        "pg_loss": pg_loss,
        "agent_grad_norm": agent_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "acct_q_loss": acct_q_loss,
        "acct_q_grad_norm": acct_q_grad_norm,
    }


def write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "status",
        "script_returncode",
        "epymarl_repository",
        "epymarl_commit",
        "algorithm",
        "learner",
        "env",
        "env_key",
        "time_limit",
        "episode",
        "t_max",
        "seed",
        "method",
        "standardise_rewards",
        "use_acct",
        "acct_influence_source",
        "acct_lambda",
        "acct_residual_weight",
        "acct_transport_mode",
        "acct_standardize",
        "return_step",
        "sample_return",
        "test_return_step",
        "greedy_return",
        "ep_length_step",
        "ep_length_mean",
        "test_ep_length_step",
        "test_ep_length_mean",
        "critic_loss",
        "td_error_abs",
        "pg_loss",
        "agent_grad_norm",
        "critic_grad_norm",
        "acct_q_loss",
        "acct_q_grad_norm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "env",
        "episode",
        "method",
        "n",
        "sample_return_mean",
        "sample_return_std",
        "greedy_return_mean",
        "greedy_return_std",
        "acct_q_loss_mean",
        "acct_q_loss_std",
    ]
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["status"] != "PASS":
            continue
        key = (str(row["env_key"]), int(row["episode"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for (env, episode, method), group in sorted(grouped.items()):
        sample = [float(row["sample_return"]) for row in group if row["sample_return"] is not None]
        greedy = [float(row["greedy_return"]) for row in group if row["greedy_return"] is not None]
        acct_q = [float(row["acct_q_loss"]) for row in group if row["acct_q_loss"] is not None]
        sample_mean, sample_std = mean_std(sample)
        greedy_mean, greedy_std = mean_std(greedy)
        acct_q_mean, acct_q_std = mean_std(acct_q)
        out_rows.append(
            {
                "env": env,
                "episode": episode,
                "method": method,
                "n": len(group),
                "sample_return_mean": sample_mean,
                "sample_return_std": sample_std,
                "greedy_return_mean": greedy_mean,
                "greedy_return_std": greedy_std,
                "acct_q_loss_mean": acct_q_mean,
                "acct_q_loss_std": acct_q_std,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epymarl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/epymarl_lbf_comparison.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/epymarl_lbf_comparison_summary.csv"))
    parser.add_argument("--env-key", default=DEFAULT_ENV_KEY)
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--t-max", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--test-nepisode", type=int, default=1)
    parser.add_argument("--standardise-rewards", choices=["True", "False"], default="False")
    parser.add_argument("--acct-lambda", type=float, default=0.90)
    parser.add_argument("--acct-residual-weight", type=float, default=0.1)
    parser.add_argument("--acct-transport-mode", choices=["absolute", "directional"], default="absolute")
    parser.add_argument("--acct-standardize", choices=["none", "masked"], default="masked")
    args = parser.parse_args()

    epymarl_root = args.epymarl_root.resolve()
    main_py = epymarl_root / "src" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(main_py)

    adapter_source = Path(__file__).resolve().parents[1] / "frameworks" / "epymarl_acct_adapter.py"
    apply_patch_to_epymarl(epymarl_root, adapter_source)

    rows = []
    for seed in args.seeds:
        for method in METHODS:
            print(f"running {method['method']} seed={seed}")
            row = run_one(
                epymarl_root=epymarl_root,
                method=method,
                seed=seed,
                env_key=args.env_key,
                time_limit=args.time_limit,
                t_max=args.t_max,
                test_nepisode=args.test_nepisode,
                standardise_rewards=args.standardise_rewards,
                acct_lambda=args.acct_lambda,
                acct_residual_weight=args.acct_residual_weight,
                acct_transport_mode=args.acct_transport_mode,
                acct_standardize=args.acct_standardize,
            )
            rows.append(row)
            write_raw(args.out, rows)
            if row["status"] != "PASS":
                raise SystemExit(f"{method['method']} seed={seed} failed")

    write_summary(args.summary, rows)
    print(f"wrote raw comparison to {args.out}")
    print(f"wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
