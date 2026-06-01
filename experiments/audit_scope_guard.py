"""Audit manuscript scope guardrails against unsupported benchmark claims."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FORBIDDEN_PATTERNS = [
    ("state_of_the_art_claim", re.compile(r"\bstate[- ]of[- ]the[- ]art\b", re.IGNORECASE)),
    ("sota_claim", re.compile(r"\bSOTA\b")),
    (
        "unqualified_outperformance",
        re.compile(r"\boutperform(?:s|ed)?\s+(?:all\s+)?(?:existing|prior|published|state[- ]of[- ]the[- ]art|SOTA)\b", re.IGNORECASE),
    ),
    (
        "superior_benchmark_performance",
        re.compile(r"\bsuperior\s+(?:benchmark\s+)?performance\b", re.IGNORECASE),
    ),
]

REQUIRED_SNIPPETS = [
    ("local_evidence_abstract", "current evidence is intentionally local rather than a claim of large-scale benchmark dominance"),
    ("mechanistic_headline", "The headline empirical claim is mechanistic"),
    ("smac_future_work", "SMAC/SMACv2-scale experiments are left as future work unless actually run"),
    ("not_tuned_suite", "not a tuned top-conference benchmark suite"),
    ("stronger_submission_comparisons", "a stronger submission should compare learned \\acct{} with AREL"),
    ("no_smac_result", "No SMAC, SMACv2, or cloud-scale benchmark result is reported"),
    ("public_anonymous_url", "https://anonymous.4open.science/r/"),
    ("submission_readiness_target", "\\texttt{submission-readiness}"),
]

FORBIDDEN_SNIPPETS = [
    ("raw_github_artifact_url", "https://github.com/LAbron665/acct-review-artifact-20260601-r2"),
    ("stale_no_public_url", "does not yet provide a public anonymous repository URL"),
]


def read_manuscript(paper: Path, checklist: Path) -> str:
    text = paper.read_text()
    if checklist.exists():
        text += "\n" + checklist.read_text()
    return text


def audit_scope(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, snippet in REQUIRED_SNIPPETS:
        rows.append(
            {
                "check": label,
                "status": "PASS" if snippet in text else "FAIL",
                "detail": snippet,
            }
        )

    for label, snippet in FORBIDDEN_SNIPPETS:
        rows.append(
            {
                "check": label,
                "status": "FAIL" if snippet in text else "PASS",
                "detail": snippet if snippet in text else "not present",
            }
        )

    for label, pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        rows.append(
            {
                "check": label,
                "status": "FAIL" if match else "PASS",
                "detail": match.group(0) if match else "not present",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex"))
    parser.add_argument("--checklist", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/checklist.tex"))
    parser.add_argument("--out", type=Path, default=Path("results/scope_guard_audit.csv"))
    args = parser.parse_args()

    text = read_manuscript(args.paper.resolve(), args.checklist.resolve())
    rows = audit_scope(text)

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["status"] != "PASS"]
    print(f"checked {len(rows)} scope guard conditions")
    print(f"failed: {len(failures)}")
    print(f"wrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
