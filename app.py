# app.py
from flask import Flask, jsonify, render_template, request
from pathlib import Path
import os
import tempfile

import librosa
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from werkzeug.utils import secure_filename

from model_gate import check_model_readiness

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "multimodal_depression_model.pt"
MODEL_MANIFEST_PATH = BASE_DIR / "models" / "model_manifest.json"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".aac"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The request is too large. Keep all selected files under 20 MB total."}), 413

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

MODEL_READY, MODEL_STATUS, model_manifest = check_model_readiness(
    MODEL_PATH, MODEL_MANIFEST_PATH, BASE_DIR
)
model = None
bert_tokenizer = None
wav_processor = None

# Fail closed: do not instantiate large encoders or serve predictions unless a
# matching checkpoint, reviewed manifest and validation report all exist.
if MODEL_READY:
    try:
        candidate = MultimodalDepressionModel().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        candidate.load_state_dict(checkpoint, strict=True)
        candidate.eval()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = candidate
        bert_tokenizer = tokenizer
        wav_processor = processor
        app.logger.info("Reviewed research model loaded: %s", model_manifest["model_version"])
    except Exception:
        app.logger.exception("Reviewed model could not be loaded; inference disabled.")
        model = None
        MODEL_READY = False
        MODEL_STATUS = "Predictions are disabled because the reviewed model could not be loaded."
else:
    app.logger.warning("%s", MODEL_STATUS)

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
    return render_template(
        "index.html",
        model_ready=MODEL_READY,
        model_status=MODEL_STATUS,
        model_task=model_manifest.get("task") if model_manifest else None,
    )


@app.route("/healthz")
def healthz():
    """Liveness endpoint: the web process is running."""
    return jsonify({"status": "ok", "model_ready": MODEL_READY}), 200


@app.route("/readyz")
def readyz():
    """Readiness endpoint: inference is enabled only for a reviewed artifact."""
    status_code = 200 if MODEL_READY else 503
    return jsonify({"ready": MODEL_READY, "detail": MODEL_STATUS}), status_code


def _stage_upload(storage, allowed_extensions):
    safe_name = secure_filename(storage.filename or "")
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in allowed_extensions:
        raise ValueError("Unsupported file type. Choose a supported image or audio file.")

    handle = tempfile.NamedTemporaryFile(prefix="mindscope-", suffix=suffix, delete=False)
    temporary_path = Path(handle.name)
    handle.close()
    try:
        storage.save(str(temporary_path))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY or model is None or model_manifest is None:
        return jsonify({"error": MODEL_STATUS}), 503

    temporary_paths = []
    try:
        text = request.form.get("text", "").strip()
        image_file = request.files.get("image")
        speech_file = request.files.get("speech")

        has_text = bool(text)
        has_image = bool(image_file and image_file.filename)
        has_audio = bool(speech_file and speech_file.filename)

        if not has_text and not has_image and not has_audio:
            return jsonify({"error": "Please provide at least one input modality."}), 400

        if (has_image or has_audio) and request.form.get("media_consent") != "yes":
            return jsonify({
                "error": "Please confirm consent before submitting an image or audio recording."
            }), 400

        text_inputs = preprocess_text(text) if has_text else None
        if text_inputs is not None:
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

        image_tensor = None
        if has_image:
            image_path = _stage_upload(image_file, ALLOWED_IMAGE_EXTENSIONS)
            temporary_paths.append(image_path)
            image_tensor = preprocess_image(str(image_path))
            if image_tensor is None:
                return jsonify({"error": "The image could not be decoded. Try a JPG, PNG or WebP image."}), 400

        audio_tensor = None
        if has_audio:
            audio_path = _stage_upload(speech_file, ALLOWED_AUDIO_EXTENSIONS)
            temporary_paths.append(audio_path)
            audio_tensor = preprocess_audio(str(audio_path))
            if audio_tensor is None:
                return jsonify({"error": "The audio could not be decoded. Try a WAV, MP3, M4A, OGG, WebM or FLAC file."}), 400

        with torch.inference_mode():
            logits, _severity_output = model(text_inputs, image_tensor, audio_tensor)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = int(torch.argmax(probabilities, dim=-1).item())
            model_score = round(float(probabilities[0][predicted_class].item()) * 100, 1)

        class_names = model_manifest["labels"]["class_names"]
        prediction = class_names[predicted_class]

        # This project does not yet have a validated severity-regression protocol
        # or a validated per-modality attribution method. Never display either as
        # if it were measured.
        return render_template(
            "result.html",
            prediction=prediction,
            confidence=model_score,
            risk="Not assessed",
            severity="N/A",
            text_input=(text[:100] + "…" if len(text) > 100 else text) if has_text else "N/A",
            has_image=has_image,
            has_audio=has_audio,
            text_score=None,
            image_score=None,
            audio_score=None,
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Research inference failed.")
        return jsonify({"error": "Analysis failed. Check the input format and try again."}), 500
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove temporary upload after request.")


if __name__ == "__main__":
    # The container/platform opens the web URL; do not try to launch a browser
    # from a headless server process.
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        use_reloader=False,
    )
