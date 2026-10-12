"""Deprecated unsafe training entry point.

The previous script used filename-derived labels and synthetic text that directly
encoded the class. It could also train on a dummy sample when Kaggle access failed,
had no independent participant-level evaluation, and did not train severity output.

It is intentionally disabled so it cannot emit a misleading checkpoint. See:
- docs/AUDIT_REPORT.md
- docs/MODERNIZATION_ROADMAP.md

Do not turn this back on by removing this guard. Implement a new, dataset-specific
training pipeline only after the target labels and permitted dataset are verified.
"""
raise SystemExit(
    "Training stopped: the legacy pipeline is unsafe and has been disabled. "
    "Review docs/AUDIT_REPORT.md and docs/MODERNIZATION_ROADMAP.md before training."
)
