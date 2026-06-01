"""Audit source and artifact text files for identity-sensitive residue."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_anonymous_release_repo import FORBIDDEN_DEFAULTS, TEXT_SUFFIXES


EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".mplconfig",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "dist",
}
EXCLUDED_FILES = {"results/anonymity_audit.csv"}


def should_scan(path: Path, rel: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in TEXT_SUFFIXES
        and rel.as_posix() not in EXCLUDED_FILES
        and not any(part in EXCLUDED_PARTS for part in rel.parts)
    )


def scan_tree(root: Path, forbidden: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not should_scan(path, rel):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token and token in text:
                rows.append({"path": rel.as_posix(), "token": token, "status": "FAIL"})
    if not rows:
        rows.append({"path": ".", "token": "", "status": "PASS"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/anonymity_audit.csv"))
    parser.add_argument("--forbidden", nargs="*", default=list(FORBIDDEN_DEFAULTS))
    args = parser.parse_args()

    root = args.root.resolve()
    rows = scan_tree(root, args.forbidden)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "token", "status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["status"] != "PASS"]
    print(f"checked anonymity tokens: {len(args.forbidden)}")
    print(f"failed: {len(failures)}")
    print(f"wrote {out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
