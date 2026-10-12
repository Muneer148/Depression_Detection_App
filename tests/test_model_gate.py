import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from model_gate import check_model_readiness


class ModelGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model.pt"
        self.manifest = self.root / "model_manifest.json"
        self.report = self.root / "validation-report.json"
        self.checkpoint.write_bytes(b"checkpoint-fixture")
        self.report.write_text('{"report": "synthetic test fixture"}', encoding="utf-8")
        self.payload = {
            "schema_version": 1,
            "model_version": "test-v1",
            "task": "synthetic_binary_task",
            "intended_use": "research_only",
            "validated_for_inference": True,
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "dataset": {"name": "synthetic-test-fixture", "source_url": "https://example.invalid/dataset"},
            "labels": {"definition": "synthetic labels only", "class_names": ["class_a", "class_b"]},
            "evaluation": {
                "independent_test": True,
                "split_unit": "participant",
                "metrics": {"f1": 0.8, "roc_auc": 0.82},
            },
            "validation_report_path": "validation-report.json",
        }
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_reviewed_fixture_is_ready(self):
        ready, reason, manifest = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertTrue(ready, reason)
        self.assertEqual(manifest["model_version"], "test-v1")

    def test_missing_checkpoint_fails_closed(self):
        self.checkpoint.unlink()
        ready, reason, manifest = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertFalse(ready)
        self.assertIn("checkpoint", reason.lower())

    def test_missing_manifest_fails_closed(self):
        self.manifest.unlink()
        ready, _, _ = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertFalse(ready)

    def test_unreviewed_model_fails_closed(self):
        self.payload["validated_for_inference"] = False
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")
        ready, _, _ = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertFalse(ready)

    def test_checkpoint_hash_mismatch_fails_closed(self):
        self.checkpoint.write_bytes(b"changed")
        ready, reason, _ = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertFalse(ready)
        self.assertIn("match", reason.lower())

    def test_file_level_random_split_fails_closed(self):
        self.payload["evaluation"]["split_unit"] = "file"
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")
        ready, reason, _ = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertFalse(ready)
        self.assertIn("participant", reason.lower())

    def test_missing_validation_report_fails_closed(self):
        self.report.unlink()
        ready, _, _ = check_model_readiness(self.checkpoint, self.manifest, self.root)
        self.assertFalse(ready)


if __name__ == "__main__":
    unittest.main()
