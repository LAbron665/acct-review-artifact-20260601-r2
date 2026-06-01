import unittest

from experiments.audit_scope_guard import REQUIRED_SNIPPETS, audit_scope


class ScopeGuardTests(unittest.TestCase):
    def test_scope_guard_passes_when_required_caveats_present(self):
        text = "\n".join(snippet for _, snippet in REQUIRED_SNIPPETS)

        rows = audit_scope(text)

        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))

    def test_scope_guard_rejects_unqualified_sota_claims(self):
        text = "\n".join(snippet for _, snippet in REQUIRED_SNIPPETS)
        text += "\nOur method outperforms existing baselines and is state-of-the-art."

        rows = audit_scope(text)
        failures = {row["check"] for row in rows if row["status"] == "FAIL"}

        self.assertIn("state_of_the_art_claim", failures)
        self.assertIn("unqualified_outperformance", failures)


if __name__ == "__main__":
    unittest.main()
