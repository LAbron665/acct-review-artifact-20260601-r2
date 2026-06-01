"""Write a lightweight dependency license metadata audit.

The audit is intentionally descriptive: optional benchmark dependencies may be
absent in a review environment, so missing packages are recorded rather than
treated as failures.
"""

from __future__ import annotations

import argparse
import csv
import re
from importlib import metadata
from pathlib import Path


SPEC_SPLIT_RE = re.compile(r"[<>=!~;\[]")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("git+"):
        egg_match = re.search(r"[#&]egg=([^&]+)", text)
        if egg_match:
            return egg_match.group(1)
        stem = text.rstrip("/").rsplit("/", maxsplit=1)[-1]
        return stem.removesuffix(".git")
    return SPEC_SPLIT_RE.split(text, maxsplit=1)[0].strip()


def read_requirements(path: Path, group: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        name = requirement_name(line)
        if name is None:
            continue
        rows.append(
            {
                "group": group,
                "requirement_file": path.name,
                "line": str(line_number),
                "requirement": line.strip(),
                "package": name,
            }
        )
    return rows


def installed_distributions() -> dict[str, metadata.Distribution]:
    distributions: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name:
            distributions[normalize_name(name)] = dist
    return distributions


def license_summary(dist: metadata.Distribution) -> str:
    expression = dist.metadata.get("License-Expression", "").strip()
    license_field = dist.metadata.get("License", "").strip()
    classifiers = [
        classifier.removeprefix("License :: ").strip()
        for classifier in dist.metadata.get_all("Classifier", [])
        if classifier.startswith("License :: ")
    ]
    parts = []
    if expression:
        parts.append(expression)
    if license_field:
        parts.append(license_field)
    if classifiers:
        parts.append("; ".join(classifiers))
    summary = " | ".join(parts) if parts else "UNKNOWN"
    summary = " ".join(summary.split())
    if len(summary) > 240:
        return summary[:237] + "..."
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/dependency_license_audit.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    requirement_specs = [
        (root / "requirements.txt", "main"),
        (root / "requirements_epymarl_smoke.txt", "optional_epymarl_smoke"),
    ]
    requirements: list[dict[str, str]] = []
    for path, group in requirement_specs:
        if path.exists():
            requirements.extend(read_requirements(path, group))

    installed = installed_distributions()
    rows: list[dict[str, str]] = []
    for req in requirements:
        dist = installed.get(normalize_name(req["package"]))
        row = dict(req)
        if dist is None:
            row.update(
                {
                    "installed_distribution": "",
                    "version": "",
                    "license_metadata": "",
                    "status": "MISSING_OR_OPTIONAL",
                }
            )
        else:
            row.update(
                {
                    "installed_distribution": dist.metadata.get("Name", req["package"]),
                    "version": dist.version,
                    "license_metadata": license_summary(dist),
                    "status": "INSTALLED_METADATA_RECORDED",
                }
            )
        rows.append(row)

    out_path = args.out if args.out.is_absolute() else root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "requirement_file",
        "line",
        "requirement",
        "package",
        "installed_distribution",
        "version",
        "license_metadata",
        "status",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    installed_count = sum(row["status"] == "INSTALLED_METADATA_RECORDED" for row in rows)
    print(f"wrote {out_path}")
    print(f"requirements: {len(rows)}")
    print(f"installed metadata records: {installed_count}")


if __name__ == "__main__":
    main()
