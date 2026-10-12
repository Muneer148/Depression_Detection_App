---
title: MindScope Research Prototype
sdk: docker
app_port: 7860
---

# MindScope — Multimodal Research Prototype

MindScope is a research/learning web app exploring text, image and audio processing. The repository previously presented outputs as depression classification; the model/data audit found that the current training path did not provide a valid basis for that claim.

## Current safety state

**Predictions are disabled by default.** The app will only load a model when a checkpoint matches a manifest that records the target task, dataset provenance, class mapping, independent participant/speaker-level test metrics, a validation report, and a checkpoint SHA-256. This gate prevents an absent checkpoint or random model weights from generating a result.

A passing code gate is not clinical validation. The report and dataset still require human review.

## Important dataset distinction

The repository's local audio paths resemble TESS acted-emotion data, with filenames such as `OAF_*_sad.wav` and `OAF_*_neutral.wav`. Folder names such as `depression1` and `normal1` do not establish a clinically diagnosed depression label. TESS can support emotion-recognition experiments such as sadness versus neutral; it must not be relabelled as clinical depression detection.

For depression-related research, a separately authorized dataset with a documented target is required. For example, access to DAIC-WOZ is controlled by its official data-use process; access must be requested and granted before use.

## Repository guide

- `app.py`: Flask entry point; readiness routes and guarded inference.
- `model_gate.py`: fail-closed checkpoint/manifest/validation-report checks.
- `models/model_manifest.example.json`: schema example with inference explicitly disabled.
- `train_real.py`: intentionally disabled legacy trainer. Do not remove its guard.
- `scripts/audit_dataset.py`: count paths, inspect WAV metadata, detect Git LFS pointer files; does not assign depression labels.
- `docs/AUDIT_REPORT.md`: evidence-based audit of the existing implementation.
- `docs/MODERNIZATION_ROADMAP.md`: module replacement plan and acceptance criteria.
- `tests/`: standard-library regression tests.

## Local setup

Use Python 3.11 or newer, then:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

The home page and `/healthz` should work without a model checkpoint; `/readyz` returns HTTP 503 and `/predict` is disabled until a reviewed artifact is configured.

## Audit datasets locally

The previous branch contained an acted-emotion corpus under misleading `depression1/normal1` folder names. It has been removed from this modernization branch so it cannot be mistaken for the DAIC-WOZ target or included in deployment.

For the retired audio corpus only, the general metadata checker remains available:
```bash
python check_data.py
python scripts/audit_dataset.py --json
```

### Audit the USC-provided DAIC-WOZ package

Keep the original DAIC-WOZ archive and extracted data on a locally protected drive. Do not upload participant interviews, transcripts, videos, audio, labels, or archives to GitHub or CI artifacts. The USC distribution is subject to its consent and data-use restrictions.

Run the new local-only package auditor:
```bash
python scripts/audit_daic_woz.py --root "D:\\path\\to\\your\\DAIC-WOZ" --json
```

It reads the split-table label columns and ZIP central-directory metadata. It prints aggregate counts only; it does not print participant IDs or transcript content, extract the archives, or upload anything. Confirm the input path is the directory containing the USC-provided split CSVs and participant ZIPs (or the extracted participant folders).

The report checks which splits are present, whether target labels and PHQ-8 scores exist, class counts, score ranges, invalid/inconsistent label-score pairs, overlap between official train/dev/test participant IDs, and whether media archives contain transcript/audio/video files. An official test split without PHQ-8 ground truth must remain unlabelled; the audit never fabricates values. Review USC's own documentation and the data-use agreement before finalizing the task and split strategy.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py model_gate.py scripts tests
```

## Model artifact contract

The example manifest is deliberately marked `validated_for_inference: false`. Do not change that flag just to enable the demo. Before creating a real `models/model_manifest.json`, the matching checkpoint, dataset provenance, class mapping, independent participant/speaker-level evaluation metrics and human-reviewed validation report must exist and agree. The checkpoint hash must match the manifest.

## Planned module evolution

1. **Correctness/safety:** model readiness gate, upload validation/cleanup, honest UI.
2. **Data contract:** verify data provenance and target labels; group splits by participant.
3. **ML modules:** separate dataset audit, preprocessing, baseline training, evaluation, model registry and inference.
4. **Evaluation:** held-out metrics, calibration review and no unsupported modality-attribution/severity claims.
5. **UI:** accessible input validation, explicit camera/microphone consent, truthful model status.
6. **Phase II video:** defer until target labels and evaluation are trustworthy; measure whether sampled frames add value on an independent split.
7. **Deployment:** reproducible dependencies, non-root container, CI and minimal runtime image.

## Disclaimer

This is a research/learning prototype, not a clinical diagnostic system. It must not be used to diagnose or rule out depression or make treatment decisions.
