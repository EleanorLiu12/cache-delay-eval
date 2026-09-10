import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExperimentPlanSiteTests(unittest.TestCase):
    def test_plan_has_one_authoritative_source(self):
        page = (ROOT / "site" / "experimental-plan.html").read_text(encoding="utf-8")
        loader = (ROOT / "site" / "plan.js").read_text(encoding="utf-8")

        self.assertFalse((ROOT / "site" / "assets" / "experimental-plan.md").exists())
        self.assertFalse((ROOT / "scripts" / "sync_experimental_plan.py").exists())
        self.assertIn('<script src="plan.js" defer></script>', page)
        self.assertNotIn("9. Completion criteria", page)
        self.assertIn('const sourcePath = "docs/experimental-plan.md"', loader)
        self.assertIn('const sourceBranch = "main"', loader)

    def test_plan_fetch_fails_closed_instead_of_showing_stale_content(self):
        loader = (ROOT / "site" / "plan.js").read_text(encoding="utf-8")

        self.assertIn('cache: "no-store"', loader)
        self.assertIn("fresh=${nonce}", loader)
        self.assertIn("can never silently display an older plan", loader)


if __name__ == "__main__":
    unittest.main()
