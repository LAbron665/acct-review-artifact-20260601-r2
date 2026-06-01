"""Publish the prepared anonymous release repository to a review-only remote."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prepare_anonymous_release_repo import (
    FORBIDDEN_DEFAULTS,
    find_leaks,
    find_spurious_git_artifacts,
)


DEFAULT_REPO = Path("dist/acct_anonymous_release_repo")
DEFAULT_REPORT = Path("dist/anonymous_publish_report.json")


def run(
    command: list[str],
    cwd: Path,
    *,
    check: bool = False,
    display_command: list[str] | None = None,
) -> dict[str, object]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = {
        "command": " ".join(display_command or command),
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-4000:],
    }
    if check and proc.returncode != 0:
        raise RuntimeError(result["output_tail"])
    return result


def git_output(command: list[str], cwd: Path) -> str:
    result = run(command, cwd, check=True)
    return str(result["output_tail"]).strip()


def add_git_remote_push(commands: list[dict[str, object]], repo: Path, remote: str, branch: str, *, force_push: bool) -> None:
    commands.append(run(["git", "remote", "remove", "origin"], cwd=repo))
    commands.append(
        run(
            ["git", "remote", "add", "origin", remote],
            cwd=repo,
            display_command=["git", "remote", "add", "origin", "<redacted-remote>"],
        )
    )
    push_command = ["git", "push", "-u", "origin", branch]
    display_command = push_command
    if force_push:
        push_command = ["git", "push", "--force", "-u", "origin", branch]
        display_command = ["git", "push", "--force", "-u", "origin", branch]
    commands.append(run(push_command, cwd=repo, display_command=display_command))


def create_github_repo_and_push(
    commands: list[dict[str, object]],
    repo: Path,
    github_repo: str,
    visibility: str,
) -> None:
    commands.append(
        run(
            [
                "gh",
                "repo",
                "create",
                github_repo,
                f"--{visibility}",
                "--source",
                ".",
                "--remote",
                "origin",
                "--push",
                "--disable-wiki",
                "--description",
                "Anonymous review artifact for ACCT.",
            ],
            cwd=repo,
            display_command=[
                "gh",
                "repo",
                "create",
                "<redacted-review-repo>",
                f"--{visibility}",
                "--source",
                ".",
                "--remote",
                "origin",
                "--push",
                "--disable-wiki",
                "--description",
                "<redacted-description>",
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    publish_target = parser.add_mutually_exclusive_group(required=True)
    publish_target.add_argument("--remote", help="fresh review-only Git remote URL")
    publish_target.add_argument("--github-repo", help="fresh review-only GitHub repo, for example owner/acct-review-artifact")
    parser.add_argument("--visibility", choices=("public", "private"), default="public")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-push", action="store_true", help="force-with-lease update an existing review-only remote")
    parser.add_argument("--forbidden", nargs="*", default=list(FORBIDDEN_DEFAULTS))
    args = parser.parse_args()

    repo = args.repo.resolve()
    report_path = args.report.resolve()
    commands: list[dict[str, object]] = []
    failures: list[str] = []

    if not repo.exists():
        failures.append(f"release repo does not exist: {repo}")
    elif not (repo / ".git").exists():
        failures.append(f"release repo is not initialized as git: {repo}")

    leaks = find_leaks(repo, args.forbidden) if repo.exists() else []
    spurious_git_paths = find_spurious_git_artifacts(repo) if repo.exists() else []
    if leaks:
        failures.append("identity-sensitive strings found in release repo")
    if spurious_git_paths:
        failures.append("spurious .git-like paths found in release repo")

    status_short = ""
    head_author = ""
    if not failures:
        status_short = git_output(["git", "status", "--short"], cwd=repo)
        head_author = git_output(["git", "log", "-1", "--format=%an <%ae>"], cwd=repo)
        if status_short:
            failures.append("release repo has uncommitted changes")
        if head_author != "Anonymous Authors <anonymous@example.com>":
            failures.append(f"unexpected HEAD author metadata: {head_author}")

    if not failures and not args.dry_run:
        if args.remote:
            add_git_remote_push(commands, repo=repo, remote=args.remote, branch=args.branch, force_push=args.force_push)
        else:
            create_github_repo_and_push(
                commands,
                repo=repo,
                github_repo=args.github_repo,
                visibility=args.visibility,
            )
        failed_commands = [
            command for command in commands
            if command["returncode"] != 0 and "remote remove" not in command["command"]
        ]
        if failed_commands:
            failures.append("git push path failed")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "repo": str(repo),
        "branch": args.branch,
        "publish_mode": "remote" if args.remote else "github_repo",
        "visibility": args.visibility if args.github_repo else "",
        "dry_run": args.dry_run,
        "identity_leaks": leaks,
        "spurious_git_paths": spurious_git_paths,
        "status_short": status_short,
        "head_author": head_author,
        "commands": commands,
        "failures": failures,
        "next_step": "Create an anonymous.4open.science link from the pushed review-only repository.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"status: {report['status']}")
    print(f"wrote {report_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
