"""Audit local audio paths without inferring depression labels from directory names.

This script checks file availability and metadata only. It does not assign clinical
labels, train models, upload media, or assume that acted sadness means depression.
"""
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any


EMOTION_SUFFIXES = {
    "angry", "disgust", "fear", "happy", "neutral", "ps", "sad", "surprise"
}


def audit_dataset(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    wav_paths = [p for p in all_files if p.suffix.lower() == ".wav"]

    directory_counts = Counter(p.parent.relative_to(root).as_posix() for p in wav_paths)
    suffix_counts: Counter[str] = Counter()
    lfs_pointer_paths: list[str] = []
    readable_audio: list[dict[str, Any]] = []
    unreadable_audio: list[dict[str, str]] = []

    for path in wav_paths:
        relative = path.relative_to(root).as_posix()
        stem_parts = path.stem.lower().split("_")
        suffix = stem_parts[-1] if stem_parts else ""
        if suffix in EMOTION_SUFFIXES:
            suffix_counts[suffix] += 1

        try:
            prefix = path.open("rb")
            with prefix:
                first_line = prefix.read(120)
            if first_line.startswith(b"version https://git-lfs.github.com/spec/v1"):
                lfs_pointer_paths.append(relative)
                unreadable_audio.append({"path": relative, "reason": "Git LFS pointer; actual audio is not present in the working tree"})
                continue

            with wave.open(str(path), "rb") as audio:
                sample_rate = audio.getframerate()
                frame_count = audio.getnframes()
                duration_seconds = frame_count / sample_rate if sample_rate else 0.0
                channels = audio.getnchannels()
                if frame_count <= 0 or sample_rate <= 0:
                    raise ValueError("empty or invalid WAV metadata")
                readable_audio.append({
                    "path": relative,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration_seconds": round(duration_seconds, 3),
                    "frames": frame_count,
                })
        except Exception as exc:
            unreadable_audio.append({"path": relative, "reason": f"{type(exc).__name__}: {exc}"})

    return {
        "data_dir": str(root.resolve()),
        "all_files": len(all_files),
        "wav_paths": len(wav_paths),
        "readable_wav_files": len(readable_audio),
        "lfs_pointer_files": len(lfs_pointer_paths),
        "unreadable_wav_files": len(unreadable_audio),
        "directory_counts": dict(sorted(directory_counts.items())),
        "filename_emotion_suffix_counts": dict(sorted(suffix_counts.items())),
        "readable_audio_metadata": readable_audio,
        "unreadable_audio": unreadable_audio,
        "warnings": [
            "Folder names such as depression1/normal1 and filename emotion suffixes are not verified clinical ground truth.",
            "This audit reports file availability and audio metadata only; it must not assign depression labels.",
            "If the files are TESS acted-emotion samples, the valid task is emotion recognition (for example, sadness versus neutral), not clinical depression detection.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dataset-depression",
        help="Dataset directory to audit (default: repo data/dataset-depression)",
    )
    parser.add_argument("--json", action="store_true", help="Print full machine-readable JSON")
    args = parser.parse_args()

    try:
        report = audit_dataset(args.data_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset directory: {report['data_dir']}")
        print(f"WAV paths: {report['wav_paths']}")
        print(f"Readable WAVs: {report['readable_wav_files']}")
        print(f"Git LFS pointers: {report['lfs_pointer_files']}")
        print(f"Unreadable WAVs: {report['unreadable_wav_files']}")
        print("\nDirectory counts:")
        for name, count in report["directory_counts"].items():
            print(f"  {name}: {count}")
        print("\nFilename emotion suffix counts:")
        for name, count in report["filename_emotion_suffix_counts"].items():
            print(f"  {name}: {count}")
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    # Unreadable data is an audit finding, not a Python-runtime failure. Keep
    # exit success so the JSON/report can be examined by CI or local tooling.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
