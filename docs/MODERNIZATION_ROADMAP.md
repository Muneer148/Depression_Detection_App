# MindScope / Depression Detection App — Modernization Roadmap

## Product boundary

This remains a research/learning prototype. It is not a clinical diagnostic system, and output must not be used to diagnose, rule out, or make treatment decisions about depression. Model outputs must be described according to what the chosen dataset actually labels.

## Prior direction to preserve

The existing concept combines text, image and audio; earlier project material described short video as a possible Phase II feature. Keep the multimodal direction, but implement it only after the core training and evaluation process is sound.

## Priority order

### Phase 0 — Correctness and safe deployment (first)

1. **Replace the permissive model-loading path.** Remove the random-weight fallback. Do not construct/download three large pretrained encoders unless a usable model artifact is available. Require a model manifest and a reviewable validation report before enabling inference.
2. **Add readiness visibility.** A health endpoint should separate "web app is running" from "a validated model is ready". The front page must show model unavailability and disable submission when inference is unavailable.
3. **Make uploads temporary and constrained.** Enforce request limits and extension/content validation, use generated temporary paths instead of user filenames, and delete media in a `finally` block. Do not persist user text, images or audio by default.
4. **Remove unsupported result claims.** No fake modality-contribution bars, uncalibrated "confidence" presented as certainty, or severity/risk gauges without a trained target and a validation/calibration method.
5. **Add tests and CI** for model readiness, upload validation, input errors, cleanup and a clean startup without model files.

### Phase 1 — Dataset and target contract

Replace `train_real.py` rather than incrementally patching its current label heuristic and dummy-text approach.

Before training, choose the actual prediction task and document:

- Dataset/source and permitted use.
- Participant/session identifiers and target labels.
- Meaning of each class and whether the target is emotion, questionnaire-screening status or a clinical diagnosis.
- Inclusion/exclusion rules, missing modalities and modality availability.
- Split strategy, with data grouped by participant to avoid speaker/participant leakage.
- Preprocessing/tokenization/sample-rate/image-resize configuration.
- Class distribution, quality-control results and an immutable training configuration.

**Dataset decision:** the repository's current audio paths correspond to TESS-style acted sadness/neutral speech. That can support an *emotion-recognition experiment*, not clinically labeled depression detection. For a depression-related research task, request access to an appropriate dataset such as DAIC-WOZ through its official process and follow its data-use restrictions; the official dataset includes clinical interviews, recordings, transcripts and questionnaire information. Do not assume that access has been granted.

No training script should create a dummy sample or write a model file when the dataset is missing/invalid.

### Phase 2 — Replace the ML pipeline with testable modules

Target structure (keep it small and add modules only when they have a tested responsibility):

```
app.py                       # thin Flask entry point
mindscope/
  config.py                  # paths, limits, supported formats
  model_registry.py          # checkpoint + manifest + compatibility gate
  preprocessing/
    text.py
    audio.py
    image.py
  inference.py               # one inference contract, missing-modality rules
  uploads.py                 # allowlist, temp files, cleanup
  schemas.py                 # result and manifest validation
scripts/
  audit_dataset.py
  train_baseline.py
  evaluate.py
tests/
  test_model_registry.py
  test_preprocessing.py
  test_uploads.py
  test_routes.py
docs/
  DATASET_AUDIT.md
  MODERNIZATION_ROADMAP.md
```

Train and compare simple, reproducible baselines before the large fusion model. Start with a modality whose data and labels are truly available; add text/audio fusion only after each unimodal baseline works. Add the image branch only if the selected dataset supplies corresponding images/video and a valid label. Keep severity regression only when the selected target includes a valid continuous score and its loss/metrics are trained and reported.

### Phase 3 — Evaluation and trustworthy result page

Every release candidate should record at least accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC and PR-AUC where appropriate, a confusion matrix and calibration analysis. Report results on an independent, participant-level holdout; do not report validation metrics as test results.

Result page changes:

- Use target labels from the manifest, not hard-coded "Depressed/Not Depressed" if the target differs.
- Label probability as uncalibrated until calibration has been measured.
- Replace "Modality Contributions" with modality availability unless a real attribution method is implemented and validated.
- Show model version, dataset/task description, scope and limitations.
- Keep severity unavailable unless a trained regression output and its target interpretation are defined.
- Keep the non-clinical disclaimer prominent; never imply "not depressed" means the user is free from depression.

### Phase 4 — UI/UX updates

- Keep the current MindScope visual identity, but show an honest model status badge.
- Add clear file type/size/duration guidance and validation errors.
- Add explicit consent before using a camera/microphone and explain that media is processed temporarily.
- Add per-modality preview, remove/replace input, reset form, progress/cancel state and graceful timeout/error messages.
- Show which modalities were actually processed, not simply which upload controls had files selected.
- Support English text until multilingual training and evaluation exist; remove "any language works" / "more detail = better accuracy" claims unless supported.
- Make upload failures visible instead of silently dropping a modality.

### Phase 5 — Video (prior Phase II idea)

Only after the data contract and evaluation are stable:

- Support a short, size-limited clip with explicit consent.
- Sample frames at a controlled rate and cap frame count.
- Extract visual features without storing the original clip by default.
- Align frames/audio/transcripts only when timestamps and labels support the alignment.
- Compare text-only, audio-only, image/frame-only and fused results.
- Measure whether video adds signal on an independent participant-level test set before keeping it.

### Phase 6 — deployment and maintenance

- Use a supported Python version and reproducible dependency constraints.
- Run as a non-root container user; avoid unnecessary system packages and exclude raw data/uploads from the image.
- Use production-appropriate server startup and honour the platform's port.
- Add CI for syntax, tests and build checks.
- Keep only synthetic fixtures and data manifests in Git; never commit private clinical records, credentials or real user submissions.

## Acceptance criteria for an inference-ready release

- Clean checkout cannot silently run a random or unreviewed model.
- Model manifest matches checkpoint architecture, preprocessing and class mapping.
- Independent, participant-level evaluation report exists and is reviewable.
- No dummy labels or fixed class-specific text are used as real features.
- Severity output is trained and evaluated or is omitted.
- Modality scores are genuine attribution or are not displayed.
- Uploaded content is validated and temporary; cleanup is tested.
- Core tests and CI pass.
- Public UI says research prototype / not diagnosis and accurately names the task.

## Immediate implementation sequence

1. Add the model-readiness gate, disabled-state UI and health/readiness endpoints.
2. Add temporary upload handling and error sanitisation.
3. Quarantine the existing unsafe trainer so it cannot generate a "trained" checkpoint.
4. Add dataset audit utility and the first tests/CI workflow.
5. Select/verify the dataset contract, then implement the new training and evaluation modules.
