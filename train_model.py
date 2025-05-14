import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from model import DrumTranscriptionModel
from .dataset import DrumTranscriptionDataset

# === CONFIG ===
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === SETUP ===
dataset = DrumTranscriptionDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = DrumTranscriptionModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# === TRAIN ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x)
        loss = F.binary_cross_entropy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

# === SAVE ===
torch.save(model.state_dict(), "./model/percussive_detection_model.pt")
print("✅ Model saved.")
