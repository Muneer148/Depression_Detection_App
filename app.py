# app.py
from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, BertModel
from torchvision import models, transforms
import numpy as np
from PIL import Image
import librosa
import traceback
import webbrowser
from threading import Timer

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ────────────────────────────────────────────────
#                MODEL DEFINITION
# ────────────────────────────────────────────────
class MultimodalDepressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained('bert-base-uncased')
        self.resnet  = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet.fc = torch.nn.Identity()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(768 + 2048 + 768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.class_head    = torch.nn.Linear(256, 2)
        self.severity_head = torch.nn.Linear(256, 1)

    def forward(self, text_inputs, image, audio):
        # Determine batch size from whatever is available
        if text_inputs is not None:
            bs = text_inputs['input_ids'].shape[0]
        elif image is not None:
            bs = image.shape[0]
        elif audio is not None:
            bs = audio.shape[0]
        else:
            raise ValueError("At least one modality must be provided")

        # Text
        if text_inputs is not None:
            text_emb = self.bert(**text_inputs).pooler_output
        else:
            text_emb = torch.zeros(bs, 768, device=device)

        # Image
        if image is not None:
            img_emb = self.resnet(image)
        else:
            img_emb = torch.zeros(bs, 2048, device=device)

        # Audio
        if audio is not None:
            audio_emb = self.wav2vec(audio).last_hidden_state.mean(dim=1)
        else:
            audio_emb = torch.zeros(bs, 768, device=device)

        fused    = torch.cat([text_emb, img_emb, audio_emb], dim=1)
        features = self.fusion(fused)
        logits   = self.class_head(features)
        severity = self.severity_head(features).squeeze(-1)
        return logits, severity


# ────────────────────────────────────────────────
#                DEVICE & LOADING
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

model = MultimodalDepressionModel().to(device)
model.eval()

MODEL_PATH = "models/multimodal_depression_model.pt"
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Loaded trained weights")
    except Exception as e:
        print(f"⚠️  Failed to load weights: {e}\nUsing random weights.")
else:
    print("⚠️  No trained model found → using random weights")

# Pre-load tokenizer / processor
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
wav_processor  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────
#              HELPER FUNCTIONS
# ────────────────────────────────────────────────
def preprocess_text(text: str):
    """Returns tokenised tensor or None if blank."""
    text = text.strip()
    if not text:
        return None
    return bert_tokenizer(
        text, padding='max_length', truncation=True,
        max_length=128, return_tensors="pt"
    )


def preprocess_image(file_path: str):
    """Returns image tensor or None on error."""
    try:
        img   = Image.open(file_path).convert("RGB")
        img_t = image_transform(img).unsqueeze(0)
        return img_t.to(device)
    except Exception as e:
        print(f"Image error: {e}")
        return None


def preprocess_audio(file_path: str, target_sr: int = 16000):
    """Returns audio tensor or None on error."""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        inputs = wav_processor(y, sampling_rate=target_sr, return_tensors="pt", padding=True)
        return inputs.input_values.to(device)
    except Exception as e:
        print(f"Audio error: {e}")
        return None


# ────────────────────────────────────────────────
#                   ROUTES
# ────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        text        = request.form.get("text", "").strip()
        image_file  = request.files.get("image")
        speech_file = request.files.get("speech")

        # ── Require at least one input ───────────────────────────────
        has_text  = bool(text)
        has_image = bool(image_file and image_file.filename)
        has_audio = bool(speech_file and speech_file.filename)

        if not has_text and not has_image and not has_audio:
            return jsonify({"error": "Please provide at least one input modality."}), 400

        # ── Text ─────────────────────────────────────────────────────
        text_inputs = preprocess_text(text) if has_text else None
        if text_inputs is not None:
            for k in text_inputs:
                text_inputs[k] = text_inputs[k].to(device)

        # ── Image ────────────────────────────────────────────────────
        image_tensor = None
        if has_image:
            img_path     = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
            image_file.save(img_path)
            image_tensor = preprocess_image(img_path)
            if image_tensor is None:
                has_image = False          # failed to decode → treat as missing

        # ── Audio ────────────────────────────────────────────────────
        audio_tensor = None
        if has_audio:
            audio_path   = os.path.join(app.config["UPLOAD_FOLDER"], speech_file.filename)
            speech_file.save(audio_path)
            audio_tensor = preprocess_audio(audio_path)
            if audio_tensor is None:
                has_audio = False

        # ── Inference ────────────────────────────────────────────────
        with torch.no_grad():
            logits, severity = model(text_inputs, image_tensor, audio_tensor)
            probs      = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item() * 100

        result     = "Depressed"     if pred_class == 1 else "Not Depressed"
        risk_level = "High"          if pred_class == 1 else "Low"
        sev_val    = round(severity.item(), 2)

        # Derive rough per-modality contribution scores (placeholder — replace with
        # gradient-based attribution in production)
        active = sum([has_text, has_image, has_audio])
        base   = round(confidence / active, 1) if active else 0

        text_score  = base if has_text  else None
        image_score = base if has_image else None
        audio_score = base if has_audio else None

        return render_template(
            "result.html",
            prediction  = result,
            confidence  = round(confidence, 1),
            risk        = risk_level,
            severity    = sev_val if (has_text or has_image or has_audio) else "N/A",
            text_input  = (text[:100] + "…" if len(text) > 100 else text) if has_text else "N/A",
            has_image   = has_image,
            has_audio   = has_audio,
            text_score  = text_score,
            image_score = image_score,
            audio_score = audio_score,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────
#              LAUNCH
# ────────────────────────────────────────────────
def open_browser():
    webbrowser.open_new("http://127.0.0.1:7860/")


if __name__ == "__main__":
    print("🚀 Starting MindScope Flask App…")
    Timer(2, open_browser).start()
    app.run(debug=False, host="0.0.0.0", port=7860, use_reloader=False)
