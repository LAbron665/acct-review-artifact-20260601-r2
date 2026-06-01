"""Tests for the source anonymity audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.audit_anonymity import scan_tree


class AnonymityAuditTests(unittest.TestCase):
    def test_scan_tree_passes_when_forbidden_tokens_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("anonymous artifact\n", encoding="utf-8")

            rows = scan_tree(root, forbidden=["secret-token"])

        self.assertEqual(rows, [{"path": ".", "token": "", "status": "PASS"}])

    def test_scan_tree_reports_forbidden_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = "/" + "Users" + "/"
            (root / "README.md").write_text(f"path {token}example/project\n", encoding="utf-8")

            rows = scan_tree(root, forbidden=[token])

        self.assertEqual(rows, [{"path": "README.md", "token": token, "status": "FAIL"}])

    def test_scan_tree_ignores_dist_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = "/" + "Users" + "/"
            (root / "dist").mkdir()
            (root / "dist" / "report.json").write_text(f"{token}example/project\n", encoding="utf-8")
            (root / "README.md").write_text("anonymous artifact\n", encoding="utf-8")

            rows = scan_tree(root, forbidden=[token])

        self.assertEqual(rows, [{"path": ".", "token": "", "status": "PASS"}])
