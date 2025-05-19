from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import AudioToMidiModel
from dataset import AudioMidiDataset

from constants import DEFAULT_MODEL_PATH, DEVICE

class ModelEvaluator:
    BATCH_SIZE = 16

    def __init__(self):
        self.model = AudioToMidiModel().to(DEVICE)
        self.dataset = AudioMidiDataset.get_evaluation_dataset()
        self.dataloader = DataLoader(self.dataset, batch_size=ModelEvaluator.BATCH_SIZE, shuffle=False)

    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, Y in self.dataloader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                output = self.model(X)
                loss = F.binary_cross_entropy_with_logits(output, Y)
                total_loss += loss.item()

                # Threshold for binary accuracy
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == Y).float().sum().item()
                total += Y.numel()

        avg_loss = total_loss / len(self.dataloader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def evaluate(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        loss, accuracy = self._evaluate(self.model, self.dataloader, DEVICE)
        print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()