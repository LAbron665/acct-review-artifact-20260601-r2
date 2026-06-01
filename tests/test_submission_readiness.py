"""Tests for final-submission readiness reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_submission_readiness import audit_readiness


class SubmissionReadinessTests(unittest.TestCase):
    def test_current_draft_caveats_are_reported_as_not_final(self) -> None:
        text = (
            "The artifact is a local ZIP and does not yet provide a public anonymous repository URL.\n"
            "No SMAC, SMACv2, or cloud-scale benchmark result is reported in this draft.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            rows = audit_readiness(text, Path(tmp))

        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["public_anonymous_url"], "NOT_READY")
        self.assertEqual(statuses["checklist_public_code_answer"], "NOT_READY")
        self.assertEqual(statuses["finalize_url_report"], "PENDING")
        self.assertEqual(statuses["large_scale_benchmark_evidence"], "WARN")
        self.assertEqual(statuses["local_release_repo"], "NOT_READY")
        self.assertEqual(statuses["publish_report"], "NOT_READY")

    def test_anonymous_url_and_reports_pass_local_readiness_checks(self) -> None:
        url = "https://anonymous.4open.science/r/ACCT-1234"
        text = f"Code will be released at {url}.\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            dist.mkdir()
            (dist / "anonymous_release_repo_report.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            (dist / "anonymous_publish_report.json").write_text(
                json.dumps({"status": "PASS", "dry_run": True}), encoding="utf-8"
            )
            (dist / "finalize_anonymous_url_report.json").write_text(
                json.dumps({"status": "PASS", "dry_run": False, "url": url}), encoding="utf-8"
            )

            rows = audit_readiness(text, root)

        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["public_anonymous_url"], "PASS")
        self.assertEqual(statuses["checklist_public_code_answer"], "PASS")
        self.assertEqual(statuses["finalize_url_report"], "PASS")
        self.assertEqual(statuses["large_scale_benchmark_evidence"], "PASS")
        self.assertEqual(statuses["local_release_repo"], "PASS")
        self.assertEqual(statuses["publish_report"], "PASS")

    def test_actual_publish_report_also_passes_readiness_checks(self) -> None:
        url = "https://anonymous.4open.science/r/ACCT-5678"
        text = f"Code will be released at {url}.\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            dist.mkdir()
            (dist / "anonymous_release_repo_report.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            (dist / "anonymous_publish_report.json").write_text(
                json.dumps({"status": "PASS", "dry_run": False}), encoding="utf-8"
            )
            (dist / "finalize_anonymous_url_report.json").write_text(
                json.dumps({"status": "PASS", "dry_run": False, "url": url}), encoding="utf-8"
            )

            rows = audit_readiness(text, root)

        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["publish_report"], "PASS")

    def test_anonymous_url_without_matching_finalization_report_is_not_ready(self) -> None:
        text = "Code will be released at https://anonymous.4open.science/r/ACCT-1234.\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            dist.mkdir()
            (dist / "anonymous_release_repo_report.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            (dist / "anonymous_publish_report.json").write_text(
                json.dumps({"status": "PASS", "dry_run": True}), encoding="utf-8"
            )
            (dist / "finalize_anonymous_url_report.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "dry_run": True,
                        "url": "https://anonymous.4open.science/r/ACCT-1234",
                    }
                ),
                encoding="utf-8",
            )

            rows = audit_readiness(text, root)

        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["public_anonymous_url"], "PASS")
        self.assertEqual(statuses["finalize_url_report"], "NOT_READY")


if __name__ == "__main__":
    unittest.main()
