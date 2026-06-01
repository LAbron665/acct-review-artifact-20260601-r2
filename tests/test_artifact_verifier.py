"""Tests for anonymous artifact verification helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.verify_anonymous_artifact import sha256, verify_manifest
from prepare_anonymous_release_repo import find_spurious_git_artifacts, remove_spurious_git_artifacts, safe_remove_output


def write_manifest(root: Path, entries: list[Path]) -> None:
    lines = ["# ACCT Anonymous Artifact Manifest", "", f"Files: {len(entries)}", ""]
    for path in entries:
        rel = path.relative_to(root).as_posix()
        lines.append(f"- `{rel}` sha256={sha256(path)}")
    (root / "MANIFEST.md").write_text("\n".join(lines) + "\n")


class ArtifactVerifierTests(unittest.TestCase):
    def test_manifest_verification_passes_for_matching_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello\n")
            nested = root / "results"
            nested.mkdir()
            (nested / "local_summary.csv").write_text("a,b\n1,2\n")
            write_manifest(root, [root / "README.md", nested / "local_summary.csv"])

            report = verify_manifest(root)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["manifest_entries"], 2)
            self.assertEqual(report["sha_mismatches"], [])

    def test_manifest_verification_detects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readme = root / "README.md"
            readme.write_text("hello\n")
            write_manifest(root, [readme])
            readme.write_text("changed\n")

            report = verify_manifest(root)

            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(report["sha_mismatches"])

    def test_release_repo_cleanup_removes_git_control_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / ".git 2").mkdir()
            (root / ".git.backup").write_text("old metadata\n")
            (root / ".gitignore").write_text("dist/\n")

            remove_spurious_git_artifacts(root, keep_git_dir=True)

            self.assertTrue((root / ".git").exists())
            self.assertTrue((root / ".gitignore").exists())
            self.assertFalse((root / ".git 2").exists())
            self.assertFalse((root / ".git.backup").exists())
            self.assertEqual(find_spurious_git_artifacts(root), [])

    def test_release_repo_cleanup_can_remove_real_git_dir_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / ".gitignore").write_text("dist/\n")

            remove_spurious_git_artifacts(root, keep_git_dir=False)

            self.assertFalse((root / ".git").exists())
            self.assertTrue((root / ".gitignore").exists())

    def test_safe_remove_output_handles_git_residue_inside_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            out = project_root / "dist" / "acct_anonymous_release_repo"
            (out / ".git 3").mkdir(parents=True)
            (out / ".git 3" / "stale").write_text("old git metadata\n")

            safe_remove_output(out, project_root)

            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
