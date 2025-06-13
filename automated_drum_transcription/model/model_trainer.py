import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from model import AudioToMidiModel
from dataset import AudioMidiDataset
import os

from ..constants import DEFAULT_MODEL_PATH, DEVICE

class ModelTrainer:
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-3

    def __init__(
            self, 
            model: AudioToMidiModel,
            dataset: AudioMidiDataset):
        self.dataloader = DataLoader(dataset, batch_size=ModelTrainer.BATCH_SIZE, shuffle=True)
        self.model = model
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=ModelTrainer.LR,
            weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.8)

    def train(self):
        loss_function = torch.nn.BCEWithLogitsLoss()
        for epoch in range(ModelTrainer.EPOCHS):
            self.model.train()
            total_loss = 0.0
            for x, y in self.dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = self.model(x)
                loss = loss_function(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step(total_loss)
            print(f"Epoch {epoch+1}/{ModelTrainer.EPOCHS} | Loss: {total_loss/len(self.dataloader):.8f}")
            print(f"LR: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

    def save(self, path: str = DEFAULT_MODEL_PATH):
        if not os.path.isdir(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        if os.path.exists(path):
            os.remove(path)
        
        torch.save(self.model.state_dict(), path)
        print("✅ Model saved.")


def single_sample_overfitting_for_debug(reload_model: bool = False):
    model_path = "./model/v2/single_sample_model.pt"
    dataset = AudioMidiDataset(mel_path="./data/single_sample/input_mels.npy", label_path="./data/single_sample/output_labels.npy")

    trainer = ModelTrainer(dataset, load_existing_model=reload_model, model_path=model_path)
    trainer.train()

    trainer.save(path=model_path)

def main():
    model = AudioToMidiModel()
    model.load()
    dataset = AudioMidiDataset.get_training_dataset()
    trainer = ModelTrainer(model, dataset, load_existing_model=True)
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main()
    # single_sample_overfitting_for_debug(reload_model = False)
