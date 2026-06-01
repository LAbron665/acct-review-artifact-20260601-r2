"""Audit manuscript citations against the BibTeX database."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CITE_RE = re.compile(
    r"\\cite(?:alp|alt|author|yearpar|year|p|t)?\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}"
)
ENTRY_START_RE = re.compile(r"@(?P<kind>\w+)\{(?P<key>[^,]+),")
FIELD_RE = re.compile(r"^\s*(?P<field>[A-Za-z][A-Za-z0-9_-]*)\s*=\s*[{\"](?P<value>.*?)[}\"],?\s*$")


def citation_keys(tex_text: str) -> set[str]:
    keys: set[str] = set()
    for match in CITE_RE.finditer(tex_text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def parse_bib_entries(bib_text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    current_key: str | None = None
    brace_depth = 0
    for line in bib_text.splitlines():
        start = ENTRY_START_RE.search(line)
        if start:
            current_key = start.group("key").strip()
            entries[current_key] = {"entry_type": start.group("kind").lower()}
            brace_depth = line.count("{") - line.count("}")
            continue
        if current_key is None:
            continue
        field = FIELD_RE.match(line)
        if field:
            entries[current_key][field.group("field").lower()] = field.group("value").strip()
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            current_key = None
    return entries


def has_locator(entry: dict[str, str]) -> bool:
    return any(entry.get(field, "").strip() for field in ("doi", "url", "eprint"))


def is_arxiv_entry(entry: dict[str, str]) -> bool:
    return "arxiv" in entry.get("journal", "").lower() or "arxiv" in entry.get("url", "").lower()


def add_row(rows: list[dict[str, str]], check: str, key: str, status: str, detail: str) -> None:
    rows.append({"check": check, "key": key, "status": status, "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex"))
    parser.add_argument("--bib", type=Path, default=Path("../Formatting_Instructions_For_NeurIPS_2026/example_paper.bib"))
    parser.add_argument("--out", type=Path, default=Path("results/bibliography_audit.csv"))
    args = parser.parse_args()

    paper = args.paper.resolve()
    bib = args.bib.resolve()
    cited = citation_keys(paper.read_text())
    entries = parse_bib_entries(bib.read_text())

    rows: list[dict[str, str]] = []
    for key in sorted(cited - set(entries)):
        add_row(rows, "citation_defined", key, "FAIL", "citation key is used in the paper but missing from BibTeX")
    for key in sorted(set(entries) - cited):
        add_row(rows, "citation_used", key, "FAIL", "BibTeX entry is not cited in the paper")
    for key in sorted(cited & set(entries)):
        entry = entries[key]
        add_row(rows, "citation_defined", key, "PASS", entry.get("title", ""))
        if is_arxiv_entry(entry):
            status = "PASS" if has_locator(entry) else "FAIL"
            add_row(rows, "arxiv_locator", key, status, "arXiv-style entries should include a DOI, URL, or eprint")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "key", "status", "detail"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["status"] != "PASS"]
    print(f"checked {len(rows)} bibliography conditions")
    print(f"failed: {len(failures)}")
    print(f"wrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
