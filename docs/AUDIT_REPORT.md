# Depression Detection App — Code and Dataset Audit

**Audit baseline:** GitHub `main`, commit `c055f5dcb9673ed4f84009226d0087d5dbf6d957`  
**Audit date:** 2026-10-12  
**Purpose:** research/learning prototype review; not a clinical validation.

## Executive finding

The current application must not be treated as a valid depression predictor. It can produce a result when it has random output-layer weights, and the current training code can learn a shortcut from synthetic text instead of the intended input signal. These are correctness blockers, not merely accuracy improvements.

## Repository contents

The top-level project contains a Flask app (`app.py`), two HTML templates, `train_real.py`, `check_data.py`, a Dockerfile and a short requirements file. There is no trained checkpoint in the audited GitHub tree, although `app.py` expects `models/multimodal_depression_model.pt`.

At audit time, the old data tree contained 800 WAV paths: 200 each under `depression1`, `depression2`, `normal1`, and `normal2`. Filenames used acted-emotion suffixes such as `_sad` and `_neutral`, with prefixes such as `OAF` and `YAF`. This resembles the Toronto Emotional Speech Set (TESS), an acted-emotion speech corpus—not clinically diagnosed depression cases. The old `data/` tree has been removed from this modernization branch, and `data/`, `datasets/`, and `downloads/` are now git-ignored. Historical commits may still contain the previous files; the deletion in this branch does not rewrite repository history.

The Git tree exposed 601 of those WAV entries as 131-byte blobs, consistent with Git LFS pointer objects. A file-path count is not proof that audio content was downloaded.

## Blocking model/training defects

### 1. Random weights can be used for predictions

In `app.py`, the model is constructed before checkpoint availability is established. If the checkpoint does not exist, or loading fails, the code prints a warning and continues using randomly initialized fusion/classification heads. The UI still labels the output `Depressed` or `Not Depressed` and displays a confidence score. This is not acceptable. Runtime must fail closed until a compatible, reviewed model artifact and validation manifest are present.

### 2. The trainer does not train the intended multimodal target

`train_real.py` currently:

- Reads files from the external Kaggle slug `creamplum/depressionfigures`, not the repository's local TESS folders.
- Assigns labels from filename/folder substrings containing `depression`, not from a verified participant-level target label.
- Substitutes one fixed synthetic sentence for every sample of each class. That makes text directly encode the label and creates severe target leakage.
- Sets the image modality to `None` for every sample, so image processing is never trained.
- Returns a dummy one-sample dataset when Kaggle is unavailable, then still runs training and writes a checkpoint.
- Has no independent validation/test split, no reported classification metrics and no reproducible split seed.
- Optimizes only cross-entropy for the classification head. The severity regression output is not included in the loss and therefore is not meaningfully trained.
- Drops the entire audio batch if any audio sample fails to load, rather than isolating/validating bad examples.

Consequently, a saved checkpoint from this script must not be accepted as an inference-ready depression model.

### 3. Result-page scores are misleading

- "Confidence" is a softmax score, not shown to be calibrated.
- The three "modality contribution" values are all set to the same `confidence / number_of_active_modalities` value. They are not attribution scores and do not indicate which input influenced the decision.
- The severity number comes from a head that has no regression loss in the trainer.
- Risk Gauge and "High/Low" are derived from the class score without validated clinical thresholds.
- The disclaimer says the app is not diagnostic, but the confident labels and invented contribution/risk graphics still imply more evidence than the code supports.

### 4. Upload/privacy/security issues

- Uploaded files are written under `uploads/` using the submitted filename, and are not deleted after inference.
- There is no explicit request-size cap or a strict allowlist of media extensions.
- Invalid image/audio decoding is silently treated as a missing modality; the request can continue with a different input configuration than the user intended.
- Unexpected exception text is returned directly to the client.
- Webcam/microphone features collect sensitive content, so processing must be temporary, consent-based and minimally logged.

### 5. Deployment/reproducibility issues

- The Dockerfile uses Python 3.9, installs system packages without cleaning the apt cache, copies the whole repository (including the data directory) into the image, and starts a Flask development server plus a browser-opening timer.
- Dependencies are broad lower bounds rather than a tested lock/constraints file; `opencv-python-headless` appears unused in the current inference code.
- There are no automated tests or CI workflow in the audited repository.

## DAIC-WOZ intake

The user reports receiving DAIC-WOZ from USC. No DAIC-WOZ archive or split CSV was attached to the current conversation workspace during this audit, so the actual package structure, provided labels, and usable modalities have not yet been verified here.

A local-only auditor `scripts/audit_daic_woz.py` now checks split-table columns/aggregate labels and ZIP central-directory modality metadata without extracting or uploading participant media, and deliberately does not print participant IDs or transcript content. Run it against the user's protected local copy before creating a dataset loader.

The [official USC ICT DAIC-WOZ page](https://dcapswoz.ict.usc.edu/) describes 189 clinical interview sessions with transcripts, participant audio, and facial-feature information; access is restricted by consent constraints to academics and other non-profit researchers. We must follow the agreement supplied with the actual distribution. Do not commit the data, upload it to CI, or publish any participant-level outputs.

## Current project direction retained

Prior project material describes BERT + ResNet-50 + Wav2Vec2 as the original multimodal direction, and lists **short-video input as a possible Phase II extension** to capture temporal cues. Keep that as a later feature, not the immediate fix. A video branch must not be added before the data target, evaluation and privacy rules are credible.

## Immediate safety rule

No user-facing depression prediction should be enabled until a checkpoint matches the model architecture and is accompanied by an explicit model manifest and an independently reviewable validation report. If the currently available data only supports sadness-versus-neutral emotion recognition, the output must be named as an emotion task—not depression detection.
