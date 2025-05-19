import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from model import AudioToMidiModel
from dataset import AudioMidiDataset

from constants import DEFAULT_MODEL_PATH, DEVICE

class ModelTrainer:
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-5

    def __init__(self):
        self.dataset = AudioMidiDataset.get_training_dataset()
        self.dataloader = DataLoader(self.dataset, batch_size=ModelTrainer.BATCH_SIZE, shuffle=True)
        self.model = AudioToMidiModel().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ModelTrainer.LR)

    def train(self):
        for epoch in range(ModelTrainer.EPOCHS):
            self.model.train()
            total_loss = 0.0
            for x, y in self.dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = self.model(x)
                loss = F.binary_cross_entropy(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{ModelTrainer.EPOCHS} | Loss: {total_loss/len(self.dataloader):.4f}")

    def save(self, path: str = DEFAULT_MODEL_PATH):
        torch.save(self.model.state_dict(), path)
        print("✅ Model saved.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
    trainer.save()
