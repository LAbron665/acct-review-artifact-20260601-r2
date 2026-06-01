"""Verify the anonymous artifact ZIP in a clean extraction directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REQUIRED_FILES = (
    "MANIFEST.md",
    "README.md",
    "ARTIFACT.md",
    "THIRD_PARTY.md",
    "Makefile",
    "requirements.txt",
    "requirements_epymarl_smoke.txt",
    "package_anonymous_artifact.py",
    "prepare_anonymous_release_repo.py",
    "publish_anonymous_release_repo.py",
    "finalize_anonymous_url.py",
    "acct/credit.py",
    "experiments/run_local_experiments.py",
    "experiments/run_medium_benchmark_suite.py",
    "experiments/plot_paper_figures.py",
    "experiments/audit_anonymity.py",
    "experiments/analyze_learned_critic_sample_sweep.py",
    "experiments/analyze_learning_efficiency.py",
    "experiments/analyze_query_budget.py",
    "experiments/audit_dependency_licenses.py",
    "experiments/run_epymarl_smoke.py",
    "experiments/run_epymarl_comparison.py",
    "experiments/audit_paper_claims.py",
    "frameworks/epymarl_acct_adapter.py",
    "frameworks/patch_epymarl_acct.py",
    "frameworks/README_EPymarl_ACCT.md",
    "tests/test_credit.py",
    "tests/test_audit_anonymity.py",
    "tests/test_epymarl_adapter.py",
    "results/local_summary.csv",
    "results/local_efficiency_summary.csv",
    "results/local_efficiency_stat_tests.csv",
    "results/query_budget.csv",
    "results/paper_claim_audit.csv",
    "results/anonymity_audit.csv",
    "results/dependency_license_audit.csv",
    "results/learned_critic_sample_sweep.csv",
    "results/learned_critic_sample_sweep_quality.csv",
    "results/learned_critic_sample_sweep_summary.csv",
    "results/epymarl_smoke.json",
    "results/epymarl_acct_smoke.json",
    "results/epymarl_acct_qhead_smoke.json",
    "results/epymarl_lbf_comparison.csv",
    "results/epymarl_lbf_comparison_summary.csv",
    "results/epymarl_lbf_comparison_stat_tests.csv",
    "results/medium_benchmark_summary.csv",
    "results/medium_benchmark_stat_tests.csv",
    "results/figure_manifest.csv",
)

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".toml", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_manifest(manifest_path: Path) -> tuple[int | None, dict[str, str]]:
    declared_count: int | None = None
    entries: dict[str, str] = {}
    for line in read_text(manifest_path).splitlines():
        if line.startswith("Files: "):
            declared_count = int(line.removeprefix("Files: ").strip())
        if not line.startswith("- `") or "` sha256=" not in line:
            continue
        rel, digest = line.removeprefix("- `").split("` sha256=", maxsplit=1)
        entries[rel] = digest.strip()
    return declared_count, entries


def verify_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "MANIFEST.md"
    if not manifest_path.exists():
        return {
            "status": "FAIL",
            "declared_files": None,
            "manifest_entries": 0,
            "missing_from_manifest": [],
            "missing_from_tree": ["MANIFEST.md"],
            "sha_mismatches": [],
        }

    declared_count, entries = parse_manifest(manifest_path)
    tree_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "MANIFEST.md"
    )
    tree_file_set = set(tree_files)
    manifest_file_set = set(entries)
    missing_from_manifest = sorted(tree_file_set - manifest_file_set)
    missing_from_tree = sorted(manifest_file_set - tree_file_set)
    sha_mismatches = []
    for rel, expected in sorted(entries.items()):
        path = root / rel
        if not path.exists():
            continue
        actual = sha256(path)
        if actual != expected:
            sha_mismatches.append({"path": rel, "expected": expected, "actual": actual})

    count_mismatch = declared_count is not None and declared_count != len(entries)
    ok = not missing_from_manifest and not missing_from_tree and not sha_mismatches and not count_mismatch
    return {
        "status": "PASS" if ok else "FAIL",
        "declared_files": declared_count,
        "manifest_entries": len(entries),
        "missing_from_manifest": missing_from_manifest,
        "missing_from_tree": missing_from_tree,
        "sha_mismatches": sha_mismatches,
    }


def run_command(command: list[str], cwd: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("dist/acct_anonymous_artifact.zip"))
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/acct_artifact_verify"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path, default=Path("dist/artifact_verification.json"))
    parser.add_argument(
        "--forbidden",
        nargs="*",
        default=None,
        help="Identity-sensitive strings to scan for. Defaults to generic path/remote patterns.",
    )
    args = parser.parse_args()
    forbidden = args.forbidden
    if forbidden is None:
        forbidden = ["/" + "Users" + "/", "git." + "overleaf" + ".com"]

    archive = args.archive.resolve()
    if not archive.exists():
        raise FileNotFoundError(archive)

    if args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(args.workdir)

    roots = [path for path in args.workdir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one extracted root, found {len(roots)}")
    root = roots[0]

    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    manifest_report = verify_manifest(root)
    leaks: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        for token in forbidden:
            if token and token in text:
                leaks.append({"path": path.relative_to(root).as_posix(), "token": token})

    commands = [
        run_command([str(args.python), "-m", "unittest", "discover", "tests"], cwd=root),
        run_command(["make", "artifact-smoke", f"PYTHON={args.python}"], cwd=root),
        run_command(
            [
                str(args.python),
                "experiments/analyze_mpe_statistics.py",
                "--input",
                "results/mpe_mappo_results.csv",
                "--out",
                "results/_verification_mpe_mappo_stat_tests.csv",
                "--baseline-method",
                "mappo_shared_gae",
                "--episode",
                "300",
            ],
            cwd=root,
        ),
    ]
    failed_commands = [cmd for cmd in commands if cmd["returncode"] != 0]
    manifest_failed = manifest_report["status"] != "PASS"

    report = {
        "archive": str(archive),
        "extracted_root": str(root),
        "required_files_checked": len(REQUIRED_FILES),
        "missing_required_files": missing,
        "manifest": manifest_report,
        "forbidden_tokens": forbidden,
        "identity_leaks": leaks,
        "commands": commands,
        "status": "PASS" if not missing and not manifest_failed and not leaks and not failed_commands else "FAIL",
    }

    report_path = args.report if args.report.is_absolute() else archive.parent.parent / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"status: {report['status']}")
    print(f"wrote {report_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
