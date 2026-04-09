"""
train_real.py — MindScope real-dataset training (Kaggle depression audio)
Streams audio files directly from Kaggle — no persistent download required.
Missing modalities are handled gracefully; the result page shows N/A for them.
"""

import os
import io
import glob
import random
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, BertModel
from torchvision import models, transforms
from PIL import Image
import librosa

# ====================== KAGGLE HELPERS ======================

def get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authenticated")
        return api
    except Exception as e:
        print(f"⚠️  Kaggle API unavailable: {e}")
        print("   Set KAGGLE_USERNAME / KAGGLE_KEY, or place ~/.kaggle/kaggle.json")
        return None


def list_kaggle_files(api, slug: str) -> list[str]:
    try:
        return [f.name for f in api.dataset_list_files(slug).files]
    except Exception as e:
        print(f"   list_kaggle_files({slug}): {e}")
        return []


def stream_kaggle_file(api, slug: str, filename: str) -> bytes | None:
    """Download one file into memory only — no permanent disk write."""
    try:
        tmp = tempfile.mkdtemp()
        api.dataset_download_file(slug, filename, path=tmp, quiet=True)
        for p in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            if os.path.isfile(p):
                data = open(p, "rb").read()
                shutil.rmtree(tmp, ignore_errors=True)
                return data
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        print(f"   stream({slug}/{filename}): {e}")
    return None


# ====================== MODEL ======================

class MultimodalDepressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained('bert-base-uncased')
        self.resnet  = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

        self.fusion = nn.Sequential(
            nn.Linear(768 + 2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.class_head    = nn.Linear(256, 2)
        self.severity_head = nn.Linear(256, 1)

    def forward(self, text_inputs=None, image=None, audio=None):
        if text_inputs is not None:
            bs = text_inputs['input_ids'].shape[0]
        elif image is not None:
            bs = image.shape[0]
        elif audio is not None:
            bs = audio.shape[0]
        else:
            raise ValueError("At least one modality required")

        text_emb  = self.bert(**text_inputs).pooler_output if text_inputs is not None \
                    else torch.zeros(bs, 768,  device=device)
        img_emb   = self.resnet(image) if image is not None \
                    else torch.zeros(bs, 2048, device=device)
        aud_emb   = self.wav2vec(audio).last_hidden_state.mean(dim=1) if audio is not None \
                    else torch.zeros(bs, 768,  device=device)

        fused    = torch.cat([text_emb, img_emb, aud_emb], dim=1)
        features = self.fusion(fused)
        return self.class_head(features), self.severity_head(features).squeeze(-1)


# ====================== DATASET ======================

# Depression audio dataset on Kaggle
AUDIO_SLUG = "creamplum/depressionfigures"          # main depression audio dataset
# Label heuristic: filenames or folder names containing 'depr' → 1, else 0
LABEL_KEYWORDS = ("depression", "depress", "dep_")

DUMMY_TEXT = {
    0: "I feel okay today, nothing special.",
    1: "I have been feeling very low and hopeless for weeks.",
}


class DepressionStreamDataset(Dataset):
    def __init__(self, api=None, max_samples: int = 300):
        self.api       = api
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.samples   = []   # list of (filename, label)

        if api is not None:
            print(f"  Listing files in {AUDIO_SLUG}…")
            all_files = list_kaggle_files(api, AUDIO_SLUG)
            wav_files = [f for f in all_files if f.lower().endswith('.wav')]
            random.shuffle(wav_files)
            wav_files = wav_files[:max_samples]
            for f in wav_files:
                label = 1 if any(kw in f.lower() for kw in LABEL_KEYWORDS) else 0
                self.samples.append((f, label))
            print(f"  → {len(self.samples)} audio samples "
                  f"({sum(l for _,l in self.samples)} depressed, "
                  f"{sum(1-l for _,l in self.samples)} normal)")
        else:
            print("  ⚠️  No Kaggle API — dataset is empty; add local wav files to data/ to train.")

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if not self.samples:
            # Dummy sample so DataLoader doesn't crash
            label = 0
            audio = None
        else:
            filename, label = self.samples[idx % len(self.samples)]
            audio = self._load_audio(filename)

        # Dummy text based on label
        text = DUMMY_TEXT[label]
        text_inputs = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=128, return_tensors="pt"
        )
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].squeeze(0)

        # Image: not available in this audio-only dataset → None → N/A in result
        image = None

        return text_inputs, image, audio, torch.tensor(label, dtype=torch.long)

    def _load_audio(self, filename: str):
        data = stream_kaggle_file(self.api, AUDIO_SLUG, filename)
        if data is None:
            return None
        try:
            y, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
            return self.processor(
                y, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_values.squeeze(0)
        except Exception as e:
            print(f"   Audio decode error ({filename}): {e}")
            return None


# ====================== COLLATE ======================

def collate_fn(batch):
    text_list, images, audios, labels = zip(*batch)

    collated_text = {
        k: torch.stack([t[k] for t in text_list])
        for k in text_list[0]
    }

    # Image — None for whole batch if any sample lacks it
    collated_image = None if any(img is None for img in images) \
                     else torch.stack(images)

    # Audio — None if any missing; else pad & stack
    if any(a is None for a in audios):
        collated_audio = None
    else:
        max_len = max(a.shape[0] for a in audios)
        padded  = [nn.functional.pad(a, (0, max_len - a.shape[0])) for a in audios]
        collated_audio = torch.stack(padded)

    return collated_text, collated_image, collated_audio, torch.stack(labels)


# ====================== TRAINING ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

api = get_kaggle_api()

print("\nBuilding dataset (streaming from Kaggle)…")
dataset    = DepressionStreamDataset(api=api, max_samples=300)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model     = MultimodalDepressionModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

print("\nStarting training…\n")
EPOCHS = 8
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for text_inputs, image, audio, labels in dataloader:
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(device)
        image  = image.to(device)  if image  is not None else None
        audio  = audio.to(device)  if audio  is not None else None
        labels = labels.to(device)

        logits, _ = model(text_inputs, image, audio)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / max(len(dataloader), 1)
    print(f"Epoch {epoch+1}/{EPOCHS}  |  Avg Loss: {avg:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/multimodal_depression_model.pt")
print("\n✅ Training finished! Weights saved to models/multimodal_depression_model.pt")
print("   Run:  python app.py")
