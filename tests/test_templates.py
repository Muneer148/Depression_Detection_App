import unittest
from pathlib import Path

from jinja2 import Environment


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"


class TemplateSafetyTests(unittest.TestCase):
    def test_templates_parse(self):
        env = Environment(autoescape=True)
        for name in ("index.html", "result.html"):
            with self.subTest(template=name):
                source = (TEMPLATES / name).read_text(encoding="utf-8")
                env.parse(source)

    def test_home_disables_form_without_reviewed_model(self):
        source = (TEMPLATES / "index.html").read_text(encoding="utf-8")
        self.assertIn("fieldset class=\"form-fieldset\"", source)
        self.assertIn("{% if not model_ready %}disabled aria-disabled=\"true\"{% endif %}", source)
        self.assertIn("Predictions are currently disabled.", source)

    def test_media_consent_exists_in_ui(self):
        source = (TEMPLATES / "index.html").read_text(encoding="utf-8")
        self.assertIn('name="media_consent" value="yes"', source)
        self.assertIn("Uploaded media is held in a temporary file and deleted after the request.", source)

    def test_result_page_does_not_claim_modality_attribution_or_severity(self):
        source = (TEMPLATES / "result.html").read_text(encoding="utf-8")
        self.assertNotIn("Modality Contributions", source)
        self.assertNotIn("Risk Gauge", source)
        self.assertIn("Uncalibrated score", source)
        self.assertIn("Severity", source)
        self.assertIn("not a diagnosis", source.lower())


if __name__ == "__main__":
    unittest.main()
