"""Run a tiny external-EPyMARL smoke test and write an anonymous summary.

This script expects an EPyMARL checkout supplied by the caller.  It does not
vendor EPyMARL into the artifact; it only verifies that the inspected learner
stack can execute a minimal MAPPO/LBF training loop in the local environment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from frameworks.patch_epymarl_acct import apply_patch_to_epymarl


DEFAULT_ENV_KEY = "lbforaging:Foraging-5x5-2p-1f-v3"


def latest_run_dir(sacred_root: Path) -> Path:
    runs = [
        path
        for path in sacred_root.rglob("run.json")
        if path.parent.is_dir() and path.parent.name.isdigit()
    ]
    if not runs:
        raise FileNotFoundError(f"no Sacred run.json files under {sacred_root}")
    return max(runs, key=lambda p: p.stat().st_mtime).parent


def final_metric(metrics: dict[str, dict[str, list[float]]], name: str) -> dict[str, float | int | None]:
    item = metrics.get(name)
    if not item or not item.get("values"):
        return {"step": None, "value": None}
    return {"step": item["steps"][-1], "value": item["values"][-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epymarl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/epymarl_smoke.json"))
    parser.add_argument("--env-key", default=DEFAULT_ENV_KEY)
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--t-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-nepisode", type=int, default=1)
    parser.add_argument("--standardise-rewards", choices=["True", "False"], default="True")
    parser.add_argument("--apply-acct-patch", action="store_true")
    parser.add_argument("--use-acct", action="store_true")
    parser.add_argument("--acct-lambda", type=float, default=0.90)
    parser.add_argument("--acct-residual-weight", type=float, default=0.5)
    parser.add_argument("--acct-transport-mode", choices=["absolute", "directional"], default="absolute")
    parser.add_argument("--acct-standardize", choices=["none", "masked"], default="masked")
    parser.add_argument(
        "--acct-influence-source",
        choices=["policy_confidence_proxy", "learned_q_head"],
        default="policy_confidence_proxy",
    )
    args = parser.parse_args()

    epymarl_root = args.epymarl_root.resolve()
    main_py = epymarl_root / "src" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(main_py)
    patch_applied = False
    if args.apply_acct_patch:
        adapter_source = Path(__file__).resolve().parents[1] / "frameworks" / "epymarl_acct_adapter.py"
        apply_patch_to_epymarl(epymarl_root, adapter_source)
        patch_applied = True

    command = [
        sys.executable,
        "src/main.py",
        "--config=mappo",
        "--env-config=gymma",
        "with",
        f"env_args.time_limit={args.time_limit}",
        f"env_args.key={args.env_key}",
        f"t_max={args.t_max}",
        f"test_nepisode={args.test_nepisode}",
        "test_interval=10",
        "log_interval=10",
        "runner_log_interval=10",
        "learner_log_interval=10",
        "use_cuda=False",
        f"standardise_rewards={args.standardise_rewards}",
        "batch_size_run=1",
        "batch_size=1",
        "buffer_size=1",
        "save_model=False",
        "use_tensorboard=False",
    ]
    if args.seed is not None:
        command.append(f"seed={args.seed}")
    if args.use_acct:
        command.extend(
            [
                "use_acct=True",
                f"acct_lambda={args.acct_lambda}",
                f"acct_residual_weight={args.acct_residual_weight}",
                f"acct_transport_mode={args.acct_transport_mode}",
                f"acct_standardize={args.acct_standardize}",
                f"acct_influence_source={args.acct_influence_source}",
            ]
        )
    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    proc = subprocess.run(
        command,
        cwd=epymarl_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    sacred_root = epymarl_root / "results" / "sacred"
    run_dir = latest_run_dir(sacred_root)
    run = json.loads((run_dir / "run.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    repo = run.get("experiment", {}).get("repositories", [{}])[0]

    summary = {
        "status": "PASS" if proc.returncode == 0 and run.get("status") == "COMPLETED" else "FAIL",
        "script_returncode": proc.returncode,
        "epymarl_repository": "https://github.com/uoe-agents/epymarl.git",
        "epymarl_commit": repo.get("commit"),
        "algorithm": config.get("name"),
        "learner": config.get("learner"),
        "environment": config.get("env"),
        "env_key": config.get("env_args", {}).get("key"),
        "time_limit": config.get("env_args", {}).get("time_limit"),
        "t_max": config.get("t_max"),
        "batch_size": config.get("batch_size"),
        "batch_size_run": config.get("batch_size_run"),
        "standardise_rewards": config.get("standardise_rewards"),
        "seed": config.get("seed"),
        "acct_patch_applied": patch_applied,
        "use_acct": bool(config.get("use_acct", False)),
        "acct_smoke_mode": config.get("acct_influence_source") if config.get("use_acct", False) else "none",
        "acct_influence_source": config.get("acct_influence_source"),
        "acct_lambda": config.get("acct_lambda"),
        "acct_residual_weight": config.get("acct_residual_weight"),
        "acct_transport_mode": config.get("acct_transport_mode"),
        "acct_standardize": config.get("acct_standardize"),
        "final_metrics": {
            "return_mean": final_metric(metrics, "return_mean"),
            "test_return_mean": final_metric(metrics, "test_return_mean"),
            "ep_length_mean": final_metric(metrics, "ep_length_mean"),
            "test_ep_length_mean": final_metric(metrics, "test_ep_length_mean"),
            "critic_loss": final_metric(metrics, "critic_loss"),
            "td_error_abs": final_metric(metrics, "td_error_abs"),
            "pg_loss": final_metric(metrics, "pg_loss"),
            "agent_grad_norm": final_metric(metrics, "agent_grad_norm"),
            "critic_grad_norm": final_metric(metrics, "critic_grad_norm"),
            "acct_q_loss": final_metric(metrics, "acct_q_loss"),
            "acct_q_grad_norm": final_metric(metrics, "acct_q_grad_norm"),
        },
        "notes": [
            "Tiny framework smoke test only; not a benchmark-quality result.",
            "The run verifies EPyMARL MAPPO/gymma/LBF learner execution on this machine.",
            "No personal filesystem paths or hostnames are stored in this summary.",
        ],
    }
    if summary["use_acct"]:
        if summary["acct_smoke_mode"] == "learned_q_head":
            summary["notes"].append(
                "ACCT used a tiny learned actual-action Q head and counterfactual action replacements for smoke testing; it is not a tuned benchmark."
            )
        else:
            summary["notes"].append(
                "ACCT used a policy-confidence proxy influence to verify learner plumbing; it is not a counterfactual-Q benchmark."
            )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"status: {summary['status']}")
    print(f"wrote {out}")
    if summary["status"] != "PASS":
        print(proc.stdout[-4000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
