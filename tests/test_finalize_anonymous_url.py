"""Tests for inserting the final anonymous review URL."""

from __future__ import annotations

import unittest
from pathlib import Path

from experiments.audit_submission_readiness import audit_readiness
from finalize_anonymous_url import (
    CHECKLIST_DRAFT_BLOCK,
    PAPER_DRAFT_SENTENCE,
    finalize_checklist_text,
    finalize_paper_text,
    validate_url,
)


class FinalizeAnonymousUrlTests(unittest.TestCase):
    def test_validate_url_accepts_anonymous_4open_links(self) -> None:
        self.assertEqual(
            validate_url("https://anonymous.4open.science/r/ACCT-1234"),
            "https://anonymous.4open.science/r/ACCT-1234",
        )

    def test_validate_url_rejects_non_anonymous_links(self) -> None:
        with self.assertRaises(ValueError):
            validate_url("https://github.com/example/acct")

    def test_finalize_texts_switch_draft_caveats_to_public_url(self) -> None:
        url = "https://anonymous.4open.science/r/ACCT-1234"
        paper, paper_changed = finalize_paper_text(f"Before. {PAPER_DRAFT_SENTENCE} After.", url)
        checklist, checklist_changed = finalize_checklist_text(CHECKLIST_DRAFT_BLOCK, url)

        self.assertTrue(paper_changed)
        self.assertTrue(checklist_changed)
        self.assertIn(f"\\url{{{url}}}", paper)
        self.assertIn("\\answerYes{}", checklist)
        self.assertNotIn("does not yet provide a public anonymous repository URL", checklist)

        readiness_rows = audit_readiness(paper + "\n" + checklist, root=Path("/tmp"))
        statuses = {row["check"]: row["status"] for row in readiness_rows}
        self.assertEqual(statuses["public_anonymous_url"], "PASS")
        self.assertEqual(statuses["checklist_public_code_answer"], "PASS")


if __name__ == "__main__":
    unittest.main()
