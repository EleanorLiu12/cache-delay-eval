import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES = ("index.html", "experimental-plan.html")


def read(name: str) -> str:
    return (ROOT / "site" / name).read_text(encoding="utf-8")


class AcademicSiteDesignTests(unittest.TestCase):
    def test_both_pages_use_the_single_academic_stylesheet(self):
        self.assertFalse((ROOT / "site" / "styles.css").exists())
        for name in PAGES:
            page = read(name)
            self.assertIn('<link rel="stylesheet" href="academic.css">', page)
            self.assertNotIn("styles.css", page)
            self.assertIn('<main class="paper"', page)
            self.assertIn("CS 699 research report", page)

    def test_results_page_is_laid_out_as_a_paper(self):
        page = read("index.html")

        self.assertIn('<header class="titleblock">', page)
        self.assertIn('<section class="abstract"', page)
        self.assertIn('<span class="label">Abstract.</span>', page)
        for number in ("Table 1.", "Table 2.", "Table 3."):
            self.assertIn(f'<span class="label">{number}</span>', page)
        for number in ("Figure 1.", "Figure 2.", "Figure 3."):
            self.assertIn(f'<span class="label">{number}</span>', page)
        self.assertEqual(page.count('<section class="sec"'), 6)

    def test_marketing_template_structures_are_gone(self):
        for name in PAGES:
            page = read(name)
            for marker in (
                "hero-chart",
                "finding-cards",
                "discussion-grid",
                "method-grid",
                "download-list",
                "summary-grid",
                "milestone-card",
                "week-sidebar",
                "section-tabs",
                'class="eyebrow"',
                'class="metric"',
            ):
                self.assertNotIn(marker, page, f"{marker} still present in {name}")

    def test_stylesheet_keeps_print_typography_not_card_decoration(self):
        styles = (ROOT / "site" / "academic.css").read_text(encoding="utf-8")

        self.assertNotIn("box-shadow", styles)
        self.assertNotIn("border-radius", styles)
        self.assertNotIn("linear-gradient", styles)
        self.assertIn("--measure:", styles)
        self.assertIn(".data-table thead th {", styles)
        self.assertIn("@media print {", styles)

    def test_every_page_carries_the_two_week_stage_navigation(self):
        stages = (
            ("Calibration", "Week 1–2"),
            ("Emulator", "Week 3–4"),
            ("Router and policies", "Week 5–6"),
            ("Delay injection", "Week 7–8"),
            ("Workload sweeps", "Week 9–10"),
            ("Trace replay", "Week 11–12"),
            ("Analysis and report", "Week 13–14"),
        )
        for name in PAGES:
            page = read(name)
            self.assertIn('<table class="schedule">', page)
            self.assertIn("Reports by stage", page)
            for stage, weeks in stages:
                self.assertIn(stage, page)
                self.assertIn(weeks, page)
            self.assertEqual(page.count('data-current="true"'), 1)
            self.assertEqual(page.count('<td class="when">'), len(stages))
            self.assertIn('<a href="index.html"', page)

    def test_interactive_features_survive_the_redesign(self):
        for name in PAGES:
            page = read(name)
            self.assertIn('class="theme-toggle"', page)
            self.assertIn('localStorage.setItem("cache-delay-theme", next)', page)
            self.assertIn('class="skip-link"', page)
            self.assertIn('href="experimental-plan.html"', page)
            self.assertIn("github.com/EleanorLiu12/cache-delay-eval", page)
            self.assertIn("Week 13–14", page)

        index = read("index.html")
        for anchor in ("#results", "#discussion", "#validation", "#methods", "#data"):
            self.assertIn(f'href="{anchor}"', index)
            self.assertIn(f'id="{anchor[1:]}"', index)
        self.assertIn("assets/calibration-lookup.csv", index)
        self.assertIn("assets/calibration-validation-report.json", index)
        for figure in ("calibration-response", "cache-benefit", "calibration-validation"):
            self.assertIn(f"assets/{figure}.png", index)
            self.assertIn(f"assets/{figure}.pdf", index)


if __name__ == "__main__":
    unittest.main()
