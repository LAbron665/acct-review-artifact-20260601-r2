"""Insert the final anonymous artifact URL into the paper and checklist."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANONYMOUS_URL_RE = re.compile(r"^https://anonymous\.4open\.science/r/[A-Za-z0-9_-]+/?$")
INLINE_ANONYMOUS_URL_RE = re.compile(r"https://anonymous\.4open\.science/r/[A-Za-z0-9_-]+/?")

PAPER_DRAFT_SENTENCE = (
    "At the current draft stage, the code release is a verified local ZIP rather than a public "
    "anonymous repository; a final submission should upload or mirror it through an anonymous "
    "review service and replace this sentence with the review URL."
)

PAPER_FINAL_SENTENCE_TEMPLATE = "The code artifact is available for anonymous review at \\url{{{url}}}."

CHECKLIST_DRAFT_BLOCK = (
    "\\item[] Answer: \\answerNo{}\n"
    "\\item[] Justification: The submission package contains the local code artifact, raw/generated "
    "CSV files, verification targets, and a release-repo preparation target, but the draft does not yet "
    "provide a public anonymous repository URL. The artifact includes a publish helper for either a "
    "fresh Git remote or authenticated GitHub CLI repository creation, plus \\texttt{finalize\\_anonymous\\_url.py} "
    "for updating this checklist once the anonymous review URL exists; before final submission, the "
    "prepared repository should be mirrored through an anonymous repository or anonymous.4open.science "
    "artifact and this checklist item should be updated."
)

CHECKLIST_FINAL_BLOCK_TEMPLATE = (
    "\\item[] Answer: \\answerYes{{}}\n"
    "\\item[] Justification: The code and data are available at \\url{{{url}}}. "
    "The artifact contains source, raw/generated CSV files, verification targets, "
    "and release-repository tooling for faithful reproduction."
)


def validate_url(url: str) -> str:
    normalized = url.rstrip("/")
    if not ANONYMOUS_URL_RE.fullmatch(url):
        raise ValueError("URL must match https://anonymous.4open.science/r/<review-id>")
    return normalized


def finalize_paper_text(text: str, url: str) -> tuple[str, bool]:
    final_sentence = PAPER_FINAL_SENTENCE_TEMPLATE.format(url=url)
    if PAPER_DRAFT_SENTENCE in text:
        return text.replace(PAPER_DRAFT_SENTENCE, final_sentence, 1), True
    final_pattern = re.compile(
        r"The code artifact is available for anonymous review at \\url\{"
        r"https://anonymous\.4open\.science/r/[A-Za-z0-9_-]+/?\}\."
    )
    new_text, count = final_pattern.subn(final_sentence, text, count=1)
    return new_text, bool(count)


def finalize_checklist_text(text: str, url: str) -> tuple[str, bool]:
    final_block = CHECKLIST_FINAL_BLOCK_TEMPLATE.format(url=url)
    if CHECKLIST_DRAFT_BLOCK in text:
        return text.replace(CHECKLIST_DRAFT_BLOCK, final_block, 1), True

    if "\\item[] Answer: \\answerYes{}" in text and INLINE_ANONYMOUS_URL_RE.search(text):
        new_text = INLINE_ANONYMOUS_URL_RE.sub(url, text)
        return new_text, new_text != text

    return text, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="anonymous.4open.science review URL")
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("../Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex"),
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=Path("../Formatting_Instructions_For_NeurIPS_2026/checklist.tex"),
    )
    parser.add_argument("--report", type=Path, default=Path("dist/finalize_anonymous_url_report.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    url = validate_url(args.url)
    paper_path = args.paper.resolve()
    checklist_path = args.checklist.resolve()
    report_path = args.report.resolve()

    paper_text = paper_path.read_text(encoding="utf-8")
    checklist_text = checklist_path.read_text(encoding="utf-8")
    new_paper, paper_changed = finalize_paper_text(paper_text, url)
    new_checklist, checklist_changed = finalize_checklist_text(checklist_text, url)

    failures: list[str] = []
    if not paper_changed:
        failures.append("paper artifact sentence was not updated")
    if not checklist_changed:
        failures.append("checklist open-code answer was not updated")

    if not failures and not args.dry_run:
        paper_path.write_text(new_paper, encoding="utf-8")
        checklist_path.write_text(new_checklist, encoding="utf-8")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "url": url,
        "dry_run": args.dry_run,
        "paper": str(paper_path),
        "checklist": str(checklist_path),
        "paper_changed": paper_changed,
        "checklist_changed": checklist_changed,
        "failures": failures,
        "next_step": "Run make release-check and make submission-readiness after applying the URL.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"status: {report['status']}")
    print(f"wrote {report_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
