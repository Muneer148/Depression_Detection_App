# app.py
from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, BertModel
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import librosa
import traceback

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
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Identity()           # remove classification head
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(768 + 2048 + 768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        self.class_head = torch.nn.Linear(256, 2)       # depressed / not
        self.severity_head = torch.nn.Linear(256, 1)    # regression (optional)

    def forward(self, text_inputs, image, audio):
        # Text
        text_emb = self.bert(**text_inputs).pooler_output          # [bs, 768]
        
        # Image
        img_emb = self.resnet(image)                               # [bs, 2048]
        
        # Audio
        audio_emb = self.wav2vec(audio).last_hidden_state.mean(dim=1)  # [bs, 768]
        
        fused = torch.cat([text_emb, img_emb, audio_emb], dim=1)   # [bs, 768+2048+768]
        features = self.fusion(fused)
        
        logits = self.class_head(features)           # [bs, 2]
        severity = self.severity_head(features).squeeze(-1)  # [bs]
        
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

# Try to load weights — will fall back to random if file missing
MODEL_PATH = "models/multimodal_depression_model.pt"
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Loaded trained weights")
    except Exception as e:
        print(f"⚠️  Failed to load weights: {e}\nUsing random weights.")
else:
    print("⚠️  No trained model found → using random weights")

# Preprocessing tools
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
wav_processor   = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ────────────────────────────────────────────────
#              HELPER FUNCTIONS
# ────────────────────────────────────────────────
def preprocess_text(text: str):
    if not text.strip():
        text = "I feel neutral today."   # better than all pads
    return bert_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

def preprocess_image(file_path: str):
    try:
        img = Image.open(file_path).convert("RGB")
        img_t = image_transform(img).unsqueeze(0)  # [1,3,224,224]
        return img_t.to(device)
    except Exception as e:
        print(f"Image error: {e}")
        return torch.randn(1, 3, 224, 224, device=device)

def preprocess_audio(file_path: str, target_sr=16000):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        # resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        inputs = wav_processor(
            y,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values.to(device)   # [1, time]
    except Exception as e:
        print(f"Audio error: {e}")
        return torch.randn(1, 80000, device=device)

# ────────────────────────────────────────────────
#                   ROUTES
# ────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "").strip()
        image_file = request.files.get("image")
        speech_file = request.files.get("speech")

        # ── Text ─────────────────────────────────────
        text_inputs = preprocess_text(text)
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(device)

        # ── Image ────────────────────────────────────
        if image_file and image_file.filename:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
            image_file.save(img_path)
            image_tensor = preprocess_image(img_path)
        else:
            image_tensor = torch.randn(1, 3, 224, 224, device=device)

        # ── Audio ────────────────────────────────────
        if speech_file and speech_file.filename:
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], speech_file.filename)
            speech_file.save(audio_path)
            audio_tensor = preprocess_audio(audio_path)
        else:
            audio_tensor = torch.randn(1, 80000, device=device)

        # ── Inference ────────────────────────────────
        with torch.no_grad():
            logits, severity = model(text_inputs, image_tensor, audio_tensor)

            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item() * 100

        result = "Depressed" if pred_class == 1 else "Not Depressed"
        risk_level = "High" if pred_class == 1 else "Low"
        severity_score = severity.item()  # can be used later (0-1 scale?)

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(confidence, 1),
            risk=risk_level,
            severity=round(severity_score, 2) if severity_score is not None else "N/A",
            text_input=text[:100] + "..." if len(text) > 100 else text
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
    import webbrowser
from threading import Timer

def open_browser():
    """Opens the default web browser to the app's local address."""
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    print("🚀 Starting Flask Depression Detection App...")
    
    # Wait 2 seconds for the server to initialize, then open the browser
    Timer(2, open_browser).start()
    
    # Note: debug=False is better for shared apps to prevent accidental crashes
    app.run(debug=False, host="0.0.0.0", port=7860, use_reloader=False)
