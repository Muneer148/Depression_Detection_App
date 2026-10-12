"""Audit a locally held DAIC-WOZ package without printing participant-level data.

This is a structural/label audit, not model training. It reads split-table
headers/labels and ZIP central-directory metadata only. It does not extract
interviews, print participant IDs or transcript contents, or upload data.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


SPLIT_HINTS = {
    "train": ("train_split_depression_avec2017.csv",),
    "dev": ("dev_split_depression_avec2017.csv", "validation_split.csv", "val_split.csv"),
    "test": ("test_split_depression_avec2017.csv",),
    "full_test": ("full_test_split.csv",),
}
PARTICIPANT_RE = re.compile(r"^(\d+)_P$", re.IGNORECASE)
ARCHIVE_RE = re.compile(r"^(\d+)_P\.zip$", re.IGNORECASE)


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    by_normalized = {_norm_key(name): name for name in fieldnames if name}
    for candidate in candidates:
        matched = by_normalized.get(_norm_key(candidate))
        if matched is not None:
            return matched
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None


def _split_kind(path: Path) -> str | None:
    lowered = path.name.lower()
    for kind, hints in SPLIT_HINTS.items():
        if lowered in hints:
            return kind
    return None


def _inspect_split(path: Path) -> dict[str, Any]:
    """Read only label metadata and return aggregate information, never row data."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            id_col = _find_column(headers, ("Participant_ID", "participant_ID", "participant_id", "ParticipantID"))
            binary_col = _find_column(headers, ("PHQ8_Binary", "PHQ_Binary", "PHQ8Binary", "PHQBinary"))
            score_col = _find_column(headers, ("PHQ8_Score", "PHQ_Score", "PHQ8Score", "PHQScore"))

            if id_col is None:
                return {
                    "present": True, "readable": False, "rows": 0,
                    "has_participant_id": False, "has_binary_labels": binary_col is not None,
                    "has_phq8_scores": score_col is not None, "labelled_rows": 0,
                    "unlabelled_rows": 0, "class_counts": {}, "score_range": None,
                    "invalid_binary_values": 0, "invalid_scores": 0,
                    "inconsistent_binary_score_pairs": 0, "error": "participant ID column not found",
                    "_participant_ids": set(),
                }

            rows = 0
            labelled_rows = 0
            unlabelled_rows = 0
            classes: Counter[str] = Counter()
            observed_scores: list[int] = []
            invalid_binary = 0
            invalid_scores = 0
            inconsistent = 0
            participant_ids: set[str] = set()
            duplicates = 0

            for row in reader:
                rows += 1
                raw_id = str(row.get(id_col, "") or "").strip()
                if raw_id:
                    if raw_id in participant_ids:
                        duplicates += 1
                    participant_ids.add(raw_id)

                binary = _int_or_none(row.get(binary_col)) if binary_col else None
                score = _int_or_none(row.get(score_col)) if score_col else None

                if binary_col:
                    raw_binary = str(row.get(binary_col, "") or "").strip()
                    if raw_binary and (binary is None or binary not in (0, 1)):
                        invalid_binary += 1
                if score_col:
                    raw_score = str(row.get(score_col, "") or "").strip()
                    if raw_score and (score is None or score < 0 or score > 24):
                        invalid_scores += 1
                    if score is not None and 0 <= score <= 24:
                        observed_scores.append(score)

                has_target = binary is not None if binary_col else False
                if has_target:
                    labelled_rows += 1
                    classes[str(binary)] += 1
                else:
                    unlabelled_rows += 1

                if binary_col and score_col and binary is not None and score is not None:
                    # Official PHQ-8 binary target is commonly thresholded at score >= 10.
                    # Report disagreement for review; never silently rewrite the labels.
                    if binary != int(score >= 10):
                        inconsistent += 1

            return {
                "present": True,
                "readable": True,
                "rows": rows,
                "has_participant_id": True,
                "has_binary_labels": binary_col is not None,
                "has_phq8_scores": score_col is not None,
                "labelled_rows": labelled_rows,
                "unlabelled_rows": unlabelled_rows,
                "class_counts": dict(sorted(classes.items())),
                "score_range": [min(observed_scores), max(observed_scores)] if observed_scores else None,
                "invalid_binary_values": invalid_binary,
                "invalid_scores": invalid_scores,
                "inconsistent_binary_score_pairs": inconsistent,
                "duplicate_participant_rows": duplicates,
                "error": None,
                "_participant_ids": participant_ids,
            }
    except (OSError, UnicodeError, csv.Error) as exc:
        return {
            "present": True, "readable": False, "rows": 0,
            "has_participant_id": False, "has_binary_labels": False,
            "has_phq8_scores": False, "labelled_rows": 0, "unlabelled_rows": 0,
            "class_counts": {}, "score_range": None, "invalid_binary_values": 0,
            "invalid_scores": 0, "inconsistent_binary_score_pairs": 0,
            "error": type(exc).__name__, "_participant_ids": set(),
        }


def _archive_media_summary(root: Path) -> dict[str, Any]:
    archives = sorted(root.rglob("*_P.zip"))
    archive_ids: set[str] = set()
    modality_counts: Counter[str] = Counter()
    broken_archives = 0
    zip_archives = 0
    for archive in archives:
        match = ARCHIVE_RE.match(archive.name)
        if match:
            archive_ids.add(match.group(1))
        try:
            with zipfile.ZipFile(archive) as zf:
                zip_archives += 1
                members = zf.infolist()
                has_transcript = any(name.lower().endswith("_transcript.csv") for name in (m.filename for m in members))
                has_audio = any(name.lower().endswith("_audio.wav") for name in (m.filename for m in members))
                has_video = any(name.lower().endswith(("_video.mp4", "_video.avi", "_video.mov", "_video.mkv")) for name in (m.filename for m in members))
                if has_transcript:
                    modality_counts["archives_with_transcript"] += 1
                if has_audio:
                    modality_counts["archives_with_audio"] += 1
                if has_video:
                    modality_counts["archives_with_video"] += 1
        except (OSError, zipfile.BadZipFile):
            broken_archives += 1

    extracted_dirs = []
    for path in root.rglob("*"):
        if path.is_dir() and PARTICIPANT_RE.match(path.name):
            extracted_dirs.append(path.name)

    return {
        "participant_archives_found": len(archives),
        "valid_zip_archives_opened": zip_archives,
        "broken_or_unreadable_archives": broken_archives,
        "archives_with_transcript": modality_counts["archives_with_transcript"],
        "archives_with_audio": modality_counts["archives_with_audio"],
        "archives_with_video": modality_counts["archives_with_video"],
        "extracted_participant_directories": len(extracted_dirs),
        "_archive_ids": archive_ids,
        "_extracted_directory_ids": {PARTICIPANT_RE.match(name).group(1) for name in extracted_dirs},
    }


def audit_daic_woz(root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    found_split_paths: dict[str, Path] = {}
    for path in sorted(root.rglob("*.csv")):
        kind = _split_kind(path)
        if kind and kind not in found_split_paths:
            found_split_paths[kind] = path

    split_reports: dict[str, dict[str, Any]] = {}
    for kind, path in found_split_paths.items():
        split_reports[kind] = _inspect_split(path)

    # The full_test table, when supplied, may be an alternate labelled metadata
    # table for the same official test IDs; do not count it as a separate split.
    core_kinds = ("train", "dev", "test")
    id_sets = {
        kind: split_reports[kind]["_participant_ids"]
        for kind in core_kinds if kind in split_reports and split_reports[kind]["readable"]
    }
    overlap_counts: dict[str, int] = {}
    kinds = list(id_sets)
    for i, left in enumerate(kinds):
        for right in kinds[i + 1:]:
            overlap_counts[f"{left}_vs_{right}"] = len(id_sets[left] & id_sets[right])

    archive_report = _archive_media_summary(root)
    all_label_ids: set[str] = set()
    for kind in ("train", "dev", "test"):
        if kind in id_sets:
            all_label_ids.update(id_sets[kind])

    available_ids = archive_report["_archive_ids"] | archive_report["_extracted_directory_ids"]
    missing_media_count = len(all_label_ids - available_ids) if available_ids else None

    public_splits = {}
    for kind, value in split_reports.items():
        public_splits[kind] = {key: item for key, item in value.items() if not key.startswith("_")}

    return {
        "root": str(root.resolve()),
        "split_tables_found": sorted(found_split_paths),
        "splits": public_splits,
        "participant_overlap_counts": overlap_counts,
        "labelled_participants_without_detected_media_archive_or_directory": missing_media_count,
        "media": {key: value for key, value in archive_report.items() if not key.startswith("_")},
        "interpretation_notes": [
            "This audit reads participant IDs and label columns only to calculate aggregate metadata; it never emits IDs or transcript contents.",
            "Official train/dev/test CSVs are not interchangeable. The official test split may lack PHQ-8 labels; do not compute test metrics without legitimate ground truth.",
            "PHQ8_Binary is a thresholded questionnaire-derived research target, not a clinical diagnosis generated by this app.",
            "Audio/video/transcripts are sensitive participant data. Keep them in a locally protected directory and do not commit, upload to public issue trackers, or include them in CI artifacts.",
            "This tool audits package structure. It does not claim the files are complete, correctly licensed for a new use, or clinically suitable without reviewing the USC-provided documentation and agreement.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local root directory of the USC-provided DAIC-WOZ package")
    parser.add_argument("--json", action="store_true", help="Print machine-readable aggregate JSON")
    args = parser.parse_args()
    try:
        report = audit_daic_woz(args.root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset root exists: {report['root']}")
        print(f"Split tables found: {', '.join(report['split_tables_found']) or 'none'}")
        for kind, entry in report["splits"].items():
            print(
                f"{kind}: rows={entry['rows']}, labels={entry['labelled_rows']}, "
                f"unlabelled={entry['unlabelled_rows']}, classes={entry['class_counts']}, "
                f"score_range={entry['score_range']}, error={entry['error']}"
            )
        print(f"Participant archives: {report['media']['participant_archives_found']}")
        print(f"Readable ZIP archives: {report['media']['valid_zip_archives_opened']}")
        print(f"Archives with transcripts: {report['media']['archives_with_transcript']}")
        print(f"Archives with audio: {report['media']['archives_with_audio']}")
        print(f"Archives with video: {report['media']['archives_with_video']}")
        print(f"Broken archives: {report['media']['broken_or_unreadable_archives']}")
        print("This is a structural audit only; it does not extract or upload participant media.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
