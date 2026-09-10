import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SiteCommentsTests(unittest.TestCase):
    def test_discussion_embeds_giscus_with_repository_identity(self):
        page = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        comments = (ROOT / "site" / "comments.js").read_text(encoding="utf-8")

        self.assertIn('<script src="comments.js" defer></script>', page)
        self.assertIn('id="comments-thread"', page)
        self.assertIn('script.src = "https://giscus.app/client.js"', comments)
        self.assertIn('script.dataset.repo = "EleanorLiu12/cache-delay-eval"', comments)
        self.assertIn('script.dataset.repoId = "R_kgDOUPuuJw"', comments)
        self.assertIn('script.dataset.categoryId = "DIC_kwDOUPuuJ84DFUIu"', comments)
        self.assertIn('script.dataset.term = "week-1-2-results"', comments)

    def test_comments_follow_the_site_theme(self):
        comments = (ROOT / "site" / "comments.js").read_text(encoding="utf-8")

        self.assertIn("new MutationObserver(setFrameTheme)", comments)
        self.assertIn('attributeFilter: ["data-theme"]', comments)
        self.assertIn('"https://giscus.app"', comments)


if __name__ == "__main__":
    unittest.main()
