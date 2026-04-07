# MindScope — Multimodal Depression Detection
---
title: Depression App
sdk: docker
app_port: 7860
---

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```
Visit: http://localhost:5000

## Model Weights
Place your trained model at:
```
models/multimodal_depression_model.pt
```
The app runs with random weights if no checkpoint is found.

## Architecture
- Text  → BERT-base-uncased (768-dim)
- Image → ResNet-50 (2048-dim)
- Audio → Wav2Vec2-base (768-dim)
- Fusion → Linear(3584→512→256) → 2-class head + severity regression

## ⚠ Disclaimer
Research tool only. Not a clinical diagnosis.
