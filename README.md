---
title: Depression App
sdk: docker
app_port: 7860
---

# Depression Detection — Multimodal Classification

A research/learning prototype combining text, image, and audio representations for multimodal depression-related classification and severity regression.

## 🚀 Run the Model Online

**[▶ Run the Model on Hugging Face Spaces](https://huggingface.co/spaces/Muneeru26/Depression_Detection)**

No local setup is required for the online demo. Click the link above to open the deployed application.

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Visit the local application at the address shown by the app.

## Model Weights

Place your trained model at:

```
models/multimodal_depression_model.pt
```

The app runs with random weights if no checkpoint is found.

## Architecture

- Text → BERT-base-uncased (768-dim)
- Image → ResNet-50 (2048-dim)
- Audio → Wav2Vec2-base (768-dim)
- Fusion → Linear(3584→512→256) → 2-class head + severity regression

## ⚠ Disclaimer

Research/learning tool only. Not a clinical diagnosis.
