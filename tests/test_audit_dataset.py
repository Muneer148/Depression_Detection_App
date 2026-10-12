import tempfile
import wave
from pathlib import Path
import unittest

from scripts.audit_dataset import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_wav(self, path: Path, sample_rate: int = 16000, frames: int = 1600):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(b"\x00\x00" * frames)

    def test_counts_readable_wavs_and_emotion_suffixes_without_relabelling(self):
        self._write_wav(self.root / "depression1" / "OAF_word_sad.wav")
        self._write_wav(self.root / "normal1" / "OAF_word_neutral.wav")
        report = audit_dataset(self.root)

        self.assertEqual(report["wav_paths"], 2)
        self.assertEqual(report["readable_wav_files"], 2)
        self.assertEqual(report["unreadable_wav_files"], 0)
        self.assertEqual(report["filename_emotion_suffix_counts"]["sad"], 1)
        self.assertEqual(report["filename_emotion_suffix_counts"]["neutral"], 1)
        self.assertTrue(any("not verified clinical ground truth" in w for w in report["warnings"]))

    def test_detects_git_lfs_pointer_instead_of_counting_it_as_audio(self):
        pointer = self.root / "depression2" / "sample_sad.wav"
        pointer.parent.mkdir(parents=True)
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "a" * 64 + "\nsize 80000\n",
            encoding="utf-8",
        )

        report = audit_dataset(self.root)
        self.assertEqual(report["wav_paths"], 1)
        self.assertEqual(report["lfs_pointer_files"], 1)
        self.assertEqual(report["readable_wav_files"], 0)
        self.assertEqual(report["unreadable_wav_files"], 1)
        self.assertIn("Git LFS pointer", report["unreadable_audio"][0]["reason"])

    def test_rejects_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            audit_dataset(self.root / "does-not-exist")


if __name__ == "__main__":
    unittest.main()
