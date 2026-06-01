"""Package the local ACCT artifact for anonymous review."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


INCLUDE_PATTERNS = (
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
    "acct/**/*.py",
    "experiments/**/*.py",
    "frameworks/**/*.md",
    "frameworks/**/*.py",
    "tests/**/*.py",
    "results/*.csv",
    "results/*.json",
)

EXCLUDE_PARTS = {
    ".cache",
    ".mplconfig",
    ".venv",
    "__pycache__",
    "dist",
}

EXCLUDE_SUFFIXES = (
    "_smoke.csv",
    "_smoke_summary.csv",
    ".pyc",
)


def should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return not any(path.name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("dist/acct_anonymous_artifact.zip"))
    args = parser.parse_args()

    root = args.root.resolve()
    archive_path = args.out if args.out.is_absolute() else root / args.out
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for pattern in INCLUDE_PATTERNS:
        files.extend(path for path in root.glob(pattern) if path.is_file() and should_include(path.relative_to(root)))
    files = sorted(set(files), key=lambda p: p.relative_to(root).as_posix())

    manifest_lines = ["# ACCT Anonymous Artifact Manifest", "", f"Files: {len(files)}", ""]
    for path in files:
        rel = path.relative_to(root)
        manifest_lines.append(f"- `{rel.as_posix()}` sha256={sha256(path)}")

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=f"acct_anonymous_artifact/{path.relative_to(root).as_posix()}")
        zf.writestr("acct_anonymous_artifact/MANIFEST.md", "\n".join(manifest_lines) + "\n")

    print(f"wrote {archive_path}")
    print(f"files: {len(files)}")
    print(f"sha256: {sha256(archive_path)}")


if __name__ == "__main__":
    main()
