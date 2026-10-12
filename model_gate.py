"""Fail-closed gate for research-model inference.

A checkpoint is not considered usable merely because a .pt file exists. It must
have a matching manifest, independent grouped evaluation metadata, and a
reviewable validation report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_readiness(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Return (ready, safe user-facing reason, parsed manifest).

    Manifest review is an explicit release gate; this function does not claim
    that reported metrics are scientifically correct. A reviewer must verify
    the referenced report and dataset provenance before setting the flag.
    """
    checkpoint = Path(checkpoint_path)
    manifest_file = Path(manifest_path)
    root = Path(project_root).resolve()

    if not checkpoint.is_file():
        return False, "Predictions are disabled: no trained checkpoint is installed.", None
    if not manifest_file.is_file():
        return False, "Predictions are disabled: no reviewed model manifest is installed.", None

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "Predictions are disabled: the model manifest is unreadable.", None

    if not isinstance(manifest, dict):
        return False, "Predictions are disabled: the model manifest has an invalid structure.", None
    if manifest.get("schema_version") != 1:
        return False, "Predictions are disabled: unsupported model-manifest version.", manifest
    if manifest.get("validated_for_inference") is not True:
        return False, "Predictions are disabled until a model has passed the documented review gate.", manifest
    if manifest.get("intended_use") != "research_only":
        return False, "Predictions are disabled: this app only supports a research-only model configuration.", manifest
    if not _nonempty(manifest.get("model_version")) or not _nonempty(manifest.get("task")):
        return False, "Predictions are disabled: model identity/task metadata is incomplete.", manifest

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not _nonempty(dataset.get("name")) or not _nonempty(dataset.get("source_url")):
        return False, "Predictions are disabled: dataset provenance is missing.", manifest

    labels = manifest.get("labels")
    if not isinstance(labels, dict) or not _nonempty(labels.get("definition")):
        return False, "Predictions are disabled: the target-label definition is missing.", manifest
    class_names = labels.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != 2 or not all(_nonempty(v) for v in class_names):
        return False, "Predictions are disabled: a binary class mapping is required.", manifest

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("independent_test") is not True:
        return False, "Predictions are disabled: an independent test evaluation is required.", manifest
    if evaluation.get("split_unit") not in {"participant", "patient", "speaker", "subject", "person"}:
        return False, "Predictions are disabled: evaluation must split by participant/speaker, not individual files.", manifest

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        return False, "Predictions are disabled: required evaluation metrics are missing.", manifest
    for key in ("f1", "roc_auc"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False, f"Predictions are disabled: a valid {key.upper()} metric is required.", manifest

    report_relative = manifest.get("validation_report_path")
    if not _nonempty(report_relative):
        return False, "Predictions are disabled: a validation report must be referenced.", manifest
    report = (root / report_relative).resolve()
    if root != report and root not in report.parents:
        return False, "Predictions are disabled: the validation-report path is invalid.", manifest
    if not report.is_file():
        return False, "Predictions are disabled: the referenced validation report is missing.", manifest

    expected_hash = manifest.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return False, "Predictions are disabled: checkpoint integrity metadata is missing.", manifest
    try:
        actual_hash = _sha256(checkpoint)
    except OSError:
        return False, "Predictions are disabled: the checkpoint cannot be read.", manifest
    if actual_hash.lower() != expected_hash.lower():
        return False, "Predictions are disabled: checkpoint and manifest do not match.", manifest

    return True, "Reviewed research model is ready.", manifest
