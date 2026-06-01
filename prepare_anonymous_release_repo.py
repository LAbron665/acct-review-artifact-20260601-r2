"""Prepare a clean Git repository from the anonymous artifact ZIP.

The resulting directory is the exact tree intended to be pushed to a fresh
review-only GitHub repository before creating an anonymous.4open.science link.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path


DEFAULT_ARCHIVE = Path("dist/acct_anonymous_artifact.zip")
DEFAULT_OUT = Path("dist/acct_anonymous_release_repo")
FORBIDDEN_DEFAULTS = ("/" + "Users" + "/", "git." + "overleaf" + ".com")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".toml", ".yaml", ".yml"}


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> dict[str, object]:
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_leaks(root: Path, forbidden: list[str]) -> list[dict[str, str]]:
    leaks: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        for token in forbidden:
            if token and token in text:
                leaks.append({"path": path.relative_to(root).as_posix(), "token": token})
    return leaks


def safe_remove_output(out_dir: Path, project_root: Path) -> None:
    resolved = out_dir.resolve()
    allowed_parent = (project_root / "dist").resolve()
    if resolved == allowed_parent or allowed_parent not in resolved.parents:
        raise ValueError(f"refusing to remove output outside dist/: {resolved}")
    robust_rmtree(resolved)


def chmod_and_retry(func, path: str, _exc_info: object) -> None:
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass
    try:
        func(path)
    except OSError:
        pass


def remove_remaining_children(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    for child in list(path.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True, onerror=chmod_and_retry)
        else:
            try:
                child.unlink(missing_ok=True)
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def robust_rmtree(path: Path) -> None:
    for attempt in range(3):
        if not path.exists():
            return
        shutil.rmtree(path, ignore_errors=attempt > 0, onerror=chmod_and_retry)
        if not path.exists():
            return
        remove_remaining_children(path)
        time.sleep(0.05)
    if path.exists():
        raise OSError(f"could not remove generated output directory: {path}")


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def is_git_control_residue(name: str) -> bool:
    return name == ".git" or name.startswith(".git ") or name.startswith(".git.")


def remove_spurious_git_artifacts(root: Path, *, keep_git_dir: bool) -> None:
    """Remove top-level .git-like residue that should not enter review copies."""
    for path in root.iterdir():
        if not is_git_control_residue(path.name):
            continue
        if keep_git_dir and path.name == ".git":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def find_spurious_git_artifacts(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if is_git_control_residue(path.name) and path.name != ".git"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="replace the default generated output directory")
    parser.add_argument("--skip-git", action="store_true", help="prepare files without running git init/commit")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit-message", default="Initial anonymous ACCT artifact")
    parser.add_argument("--report", type=Path, default=Path("dist/anonymous_release_repo_report.json"))
    parser.add_argument("--forbidden", nargs="*", default=list(FORBIDDEN_DEFAULTS))
    args = parser.parse_args()

    project_root = Path(".").resolve()
    archive = args.archive if args.archive.is_absolute() else project_root / args.archive
    out_dir = args.out if args.out.is_absolute() else project_root / args.out
    report_path = args.report if args.report.is_absolute() else project_root / args.report

    if not archive.exists():
        raise FileNotFoundError(archive)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to replace the generated release repo")
        safe_remove_output(out_dir, project_root)

    tmp_dir = out_dir.parent / f".{out_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_dir)
        roots = [path for path in tmp_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"expected one extracted root, found {len(roots)}")

        copy_tree(roots[0], out_dir)
        remove_spurious_git_artifacts(out_dir, keep_git_dir=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    leaks = find_leaks(out_dir, args.forbidden)
    git_results: list[dict[str, object]] = []
    if not leaks and not args.skip_git:
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": "Anonymous Authors",
                "GIT_AUTHOR_EMAIL": "anonymous@example.com",
                "GIT_COMMITTER_NAME": "Anonymous Authors",
                "GIT_COMMITTER_EMAIL": "anonymous@example.com",
            }
        )
        git_results.append(run(["git", "init", "-b", args.branch], cwd=out_dir, env=env))
        remove_spurious_git_artifacts(out_dir, keep_git_dir=True)
        git_results.extend(
            [
                run(["git", "add", "."], cwd=out_dir, env=env),
                run(["git", "commit", "-m", args.commit_message], cwd=out_dir, env=env),
                run(["git", "status", "--short"], cwd=out_dir, env=env),
            ]
        )

    failed_git = [result for result in git_results if result["returncode"] != 0]
    spurious_git_paths = find_spurious_git_artifacts(out_dir)
    report = {
        "archive": str(archive),
        "release_repo": str(out_dir),
        "forbidden_tokens": args.forbidden,
        "identity_leaks": leaks,
        "spurious_git_paths": spurious_git_paths,
        "git_initialized": not args.skip_git and not leaks and not failed_git,
        "git_results": git_results,
        "status": "PASS" if not leaks and not failed_git and not spurious_git_paths else "FAIL",
        "next_steps": [
            "Create a fresh review-only GitHub repository.",
            f"From {out_dir}, add that repository as origin and push branch {args.branch}.",
            "Log in to https://anonymous.4open.science/ and anonymize the pushed GitHub repository.",
            "Replace the paper's local-artifact sentence with the generated anonymous review URL.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"status: {report['status']}")
    print(f"release_repo: {out_dir}")
    print(f"report: {report_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)
    if args.skip_git:
        print("git: skipped")
    elif report["git_initialized"]:
        print("git: initialized with anonymous author metadata")


if __name__ == "__main__":
    main()
