"""Report final-submission readiness blockers for the ACCT draft.

The local artifact can be internally consistent while the paper is still not
ready for final conference submission. This audit records that distinction
without making ordinary release checks fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ANONYMOUS_URL_RE = re.compile(r"https://anonymous\.4open\.science/r/[A-Za-z0-9_-]+")


def add_row(
    rows: list[dict[str, str]],
    check: str,
    status: str,
    detail: str,
    next_action: str,
) -> None:
    rows.append({"check": check, "status": status, "detail": detail, "next_action": next_action})


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_readiness(paper_text: str, root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    anonymous_urls = sorted(set(ANONYMOUS_URL_RE.findall(paper_text)))

    if anonymous_urls:
        add_row(
            rows,
            "public_anonymous_url",
            "PASS",
            f"anonymous.4open.science URL appears in the manuscript: {anonymous_urls[0]}",
            "None",
        )
    else:
        add_row(
            rows,
            "public_anonymous_url",
            "NOT_READY",
            "the manuscript still says the release is a local ZIP or lacks a public anonymous URL",
            "Push the prepared review-only repository and replace the local-artifact placeholder with the anonymous review URL.",
        )

    if "does not yet provide a public anonymous repository URL" in paper_text:
        add_row(
            rows,
            "checklist_public_code_answer",
            "NOT_READY",
            "the NeurIPS checklist still discloses that no public anonymous repository URL exists",
            "After creating the anonymous URL, update the checklist open-code answer and justification.",
        )
    else:
        add_row(
            rows,
            "checklist_public_code_answer",
            "PASS",
            "checklist no longer contains the no-public-URL caveat",
            "None",
        )

    finalize_report = load_json(root / "dist/finalize_anonymous_url_report.json")
    if not anonymous_urls:
        add_row(
            rows,
            "finalize_url_report",
            "PENDING",
            "no anonymous URL is present yet, so finalization report matching is deferred",
            "After anonymous.4open.science returns a URL, run finalize_anonymous_url.py without --dry-run.",
        )
    elif (
        finalize_report.get("status") == "PASS"
        and finalize_report.get("dry_run") is False
        and finalize_report.get("url") in anonymous_urls
    ):
        add_row(
            rows,
            "finalize_url_report",
            "PASS",
            "finalize_anonymous_url.py report is PASS and matches the manuscript URL",
            "None",
        )
    else:
        add_row(
            rows,
            "finalize_url_report",
            "NOT_READY",
            "anonymous URL is present, but finalization report is absent, dry-run-only, failed, or mismatched",
            "Run finalize_anonymous_url.py without --dry-run using the final anonymous review URL.",
        )

    if "No SMAC, SMACv2, or cloud-scale benchmark result is reported" in paper_text:
        add_row(
            rows,
            "large_scale_benchmark_evidence",
            "WARN",
            "the draft intentionally reports no SMAC/SMACv2 or cloud-scale benchmark result",
            "Keep broad performance claims disabled, or run and report tuned benchmark comparisons before upgrading the claim scope.",
        )
    else:
        add_row(
            rows,
            "large_scale_benchmark_evidence",
            "PASS",
            "no explicit no-SMAC caveat remains",
            "Confirm that benchmark results are actually present before broadening claims.",
        )

    release_report = load_json(root / "dist/anonymous_release_repo_report.json")
    if release_report.get("status") == "PASS":
        add_row(rows, "local_release_repo", "PASS", "prepared release repository report is PASS", "None")
    else:
        add_row(rows, "local_release_repo", "NOT_READY", "prepared release repository report is absent or not PASS", "Run make release-repo PYTHON=.venv/bin/python.")

    publish_report = load_json(root / "dist/anonymous_publish_report.json")
    if publish_report.get("status") == "PASS" and publish_report.get("dry_run") is False:
        add_row(rows, "publish_report", "PASS", "publish helper report is PASS for an actual push", "None")
    elif publish_report.get("status") == "PASS" and publish_report.get("dry_run") is True:
        add_row(rows, "publish_report", "PASS", "publish helper dry-run report is PASS", "None")
    else:
        add_row(rows, "publish_report", "NOT_READY", "publish helper report is absent or not PASS", "Run publish_anonymous_release_repo.py with the intended fresh remote.")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--paper", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex"))
    parser.add_argument("--checklist", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/checklist.tex"))
    parser.add_argument("--out", type=Path, default=Path("results/submission_readiness.csv"))
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any NOT_READY check remains")
    args = parser.parse_args()

    root = args.root.resolve()
    paper_path = args.paper if args.paper.is_absolute() else root / args.paper
    checklist_path = args.checklist if args.checklist.is_absolute() else root / args.checklist
    paper_text = paper_path.read_text(encoding="utf-8")
    if checklist_path.exists():
        paper_text += "\n" + checklist_path.read_text(encoding="utf-8")

    rows = audit_readiness(paper_text, root)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail", "next_action"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    not_ready = [row for row in rows if row["status"] == "NOT_READY"]
    warnings = [row for row in rows if row["status"] in {"PENDING", "WARN"}]
    overall = "FINAL_READY" if not not_ready else "DRAFT_NOT_FINAL"
    print(f"status: {overall}")
    print(f"not_ready: {len(not_ready)}")
    print(f"warnings: {len(warnings)}")
    print(f"wrote {out}")
    if args.strict and not_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
