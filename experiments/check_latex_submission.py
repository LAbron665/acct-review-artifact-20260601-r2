"""Audit LaTeX submission hygiene for the ACCT draft."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LOG_FAILURE_PATTERNS = {
    "undefined_reference": r"undefined references|Reference `[^']+' .* undefined",
    "undefined_citation": r"Citation `[^']+' .* undefined|There were undefined citations",
    "changed_labels": r"Label\(s\) may have changed|Rerun to get",
    "overfull_box": r"Overfull \\[hv]box",
    "pdfendlink": r"pdfendlink",
}


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing required LaTeX file: {path}")
    return path.read_text(errors="replace")


def parse_label_page(aux_text: str, label: str) -> int:
    escaped = re.escape(label)
    match = re.search(rf"\\newlabel\{{{escaped}\}}\{{\{{[^}}]*\}}\{{(\d+)\}}", aux_text)
    if not match:
        raise ValueError(f"label {label!r} not found in aux file")
    return int(match.group(1))


def parse_total_pages(aux_text: str) -> int:
    match = re.search(r"\\gdef\s+\\@abspage@last\{(\d+)\}", aux_text)
    if not match:
        raise ValueError("total page marker \\@abspage@last not found in aux file")
    return int(match.group(1))


def find_log_failures(log_text: str) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for name, pattern in LOG_FAILURE_PATTERNS.items():
        matches = sorted(set(re.findall(pattern, log_text, flags=re.IGNORECASE)))
        if matches:
            failures[name] = matches
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026"))
    parser.add_argument("--tex-name", default="neurips_2026")
    parser.add_argument("--max-main-pages", type=int, default=9)
    parser.add_argument("--out", type=Path, default=Path("results/latex_submission_audit.json"))
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    aux_text = read_text(paper_dir / f"{args.tex_name}.aux")
    log_text = read_text(paper_dir / f"{args.tex_name}.log")

    main_end_page = parse_label_page(aux_text, "sec:main-end")
    appendix_start_page = parse_label_page(aux_text, "app:evidence-scope")
    checklist_start_page = parse_label_page(aux_text, "app:checklist")
    total_pages = parse_total_pages(aux_text)
    log_failures = find_log_failures(log_text)

    checks = {
        "main_page_budget": main_end_page <= args.max_main_pages,
        "appendix_after_main": appendix_start_page > main_end_page,
        "checklist_after_appendix": checklist_start_page >= appendix_start_page,
        "total_pages_consistent": total_pages >= checklist_start_page,
        "latex_log_clean": not log_failures,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "max_main_pages": args.max_main_pages,
        "main_end_page": main_end_page,
        "appendix_start_page": appendix_start_page,
        "checklist_start_page": checklist_start_page,
        "total_pages": total_pages,
        "checks": checks,
        "log_failures": log_failures,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"status: {report['status']}")
    print(f"main_end_page: {main_end_page} / {args.max_main_pages}")
    print(f"total_pages: {total_pages}")
    print(f"wrote {args.out}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
