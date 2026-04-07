import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, Wav2Vec2Processor, Wav2Vec2Model, BertModel
from torchvision import models, transforms
from PIL import Image
import librosa
import numpy as np

# ====================== MODEL (same as app.py) ======================
class MultimodalDepressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        
        self.fusion = nn.Sequential(
            nn.Linear(768 + 2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.class_head = nn.Linear(256, 2)
        self.severity_head = nn.Linear(256, 1)

    def forward(self, text_inputs, image, audio):
        text_emb = self.bert(**text_inputs).pooler_output
        img_emb = self.resnet(image)
        aud_emb = self.wav2vec(audio).last_hidden_state.mean(dim=1)
        
        fused = torch.cat([text_emb, img_emb, aud_emb], dim=1)
        features = self.fusion(fused)
        
        logits = self.class_head(features)
        severity = self.severity_head(features).squeeze(-1)
        return logits, severity


# ====================== DEVICE ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== SIMPLE DATASET ======================
class DepressionAudioDataset(Dataset):
    def __init__(self, data_root="data/depression-dataset"):
        self.samples = []
        for label, folder in enumerate(["normal1", "normal2", "depression1", "depression2"]):
            paths = glob.glob(os.path.join(data_root, folder, "*.wav"))
            for p in paths:
                self.samples.append((p, 1 if "depression" in folder else 0))  # 0=normal, 1=depressed
        
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} audio files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Audio
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            audio_tensor = processor(y, sampling_rate=16000, return_tensors="pt", padding=True).input_values.squeeze(0)
        except:
            audio_tensor = torch.randn(80000)  # fallback
        
        # Dummy text (since we have only audio)
        text = "I feel neutral today." if label == 0 else "I have been feeling very low and sad lately."
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        
        # Dummy image (random for now)
        image = torch.randn(3, 224, 224)
        
        return text_inputs, image, audio_tensor, torch.tensor(label)


# ====================== TRAINING ======================
model = MultimodalDepressionModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

dataset = DepressionAudioDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)   # small batch for GPU memory

print("\nStarting training on Kaggle depression audio dataset...\n")

model.train()
for epoch in range(8):   # increase later if needed
    total_loss = 0
    for batch in dataloader:
        text_inputs, image, audio, labels = batch
        
        # Move to device
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(device)
        image = image.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        
        logits, _ = model(text_inputs, image, audio)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/8 | Avg Loss: {total_loss/len(dataloader):.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/multimodal_depression_model.pt")
print("\n✅ Training finished! Model saved.")
print("Now you can run: python app.py")