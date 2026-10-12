import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.audit_daic_woz import audit_daic_woz


class DaicWozAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_split(self, filename, rows, headers=None):
        path = self.root / filename
        if headers is None:
            headers = list(rows[0].keys()) if rows else ["Participant_ID"]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_aggregate_train_dev_labels_and_no_participant_ids_in_report(self):
        self.write_split("train_split_Depression_AVEC2017.csv", [
            {"Participant_ID": "301", "PHQ8_Binary": "0", "PHQ8_Score": "8"},
            {"Participant_ID": "302", "PHQ8_Binary": "1", "PHQ8_Score": "14"},
        ])
        self.write_split("dev_split_Depression_AVEC2017.csv", [
            {"Participant_ID": "303", "PHQ8_Binary": "1", "PHQ8_Score": "10"},
        ])
        report = audit_daic_woz(self.root)
        self.assertEqual(report["splits"]["train"]["rows"], 2)
        self.assertEqual(report["splits"]["train"]["class_counts"], {"0": 1, "1": 1})
        self.assertEqual(report["splits"]["train"]["score_range"], [8, 14])
        self.assertEqual(report["participant_overlap_counts"]["train_vs_dev"], 0)
        self.assertNotIn("301", str(report))
        self.assertNotIn("_participant_ids", str(report))

    def test_unlabelled_official_test_is_not_fabricated(self):
        self.write_split("test_split_Depression_AVEC2017.csv", [
            {"Participant_ID": "400", "Gender": "0"},
            {"Participant_ID": "401", "Gender": "1"},
        ])
        report = audit_daic_woz(self.root)
        test_report = report["splits"]["test"]
        self.assertFalse(test_report["has_binary_labels"])
        self.assertEqual(test_report["labelled_rows"], 0)
        self.assertEqual(test_report["unlabelled_rows"], 2)
        self.assertEqual(test_report["class_counts"], {})

    def test_detects_binary_score_disagreement_and_invalid_score(self):
        self.write_split("train_split_Depression_AVEC2017.csv", [
            {"Participant_ID": "301", "PHQ8_Binary": "0", "PHQ8_Score": "12"},
            {"Participant_ID": "302", "PHQ8_Binary": "2", "PHQ8_Score": "30"},
        ])
        report = audit_daic_woz(self.root)["splits"]["train"]
        self.assertEqual(report["inconsistent_binary_score_pairs"], 1)
        self.assertEqual(report["invalid_binary_values"], 1)
        self.assertEqual(report["invalid_scores"], 1)
        self.assertEqual(report["class_counts"], {"0": 1})
        self.assertEqual(report["labelled_rows"], 1)
        self.assertEqual(report["score_labelled_rows"], 1)
        self.assertEqual(report["score_unlabelled_rows"], 1)

    def test_reads_zip_central_directory_for_modalities_without_extracting(self):
        archive = self.root / "301_P.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("301_P/301_TRANSCRIPT.csv", "start_time,duration,speaker,value\n")
            zf.writestr("301_P/301_AUDIO.wav", b"metadata only")
            zf.writestr("301_P/301_VIDEO.mp4", b"metadata only")
            zf.writestr("301_P/301_CLNF_AUs.csv", "frame,AU01\n")
        report = audit_daic_woz(self.root)
        self.assertEqual(report["media"]["participant_archives_found"], 1)
        self.assertEqual(report["media"]["valid_zip_archives_opened"], 1)
        self.assertEqual(report["media"]["archives_with_transcript"], 1)
        self.assertEqual(report["media"]["archives_with_audio"], 1)
        self.assertEqual(report["media"]["archives_with_video"], 1)
        self.assertEqual(report["media"]["archives_with_facial_features"], 1)

    def test_rejects_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            audit_daic_woz(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
