from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import AudioToMidiModel
from dataset import AudioMidiDataset

from constants import DEFAULT_MODEL_PATH, DEVICE

class ModelEvaluator:
    BATCH_SIZE = 16

    def __init__(self, dataset: AudioMidiDataset):
        self.model = AudioToMidiModel().to(DEVICE)
        self.dataloader = DataLoader(dataset, batch_size=ModelEvaluator.BATCH_SIZE, shuffle=False)

    def _evaluate(self) -> Tuple[float, float]:
        loss_function = torch.nn.BCEWithLogitsLoss()
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, Y in self.dataloader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                output = self.model(X)
                loss = loss_function(output, Y)
                total_loss += loss.item()

                # Threshold for binary accuracy
                pred = AudioToMidiModel.logits_to_predictions(output)
                correct += (pred == Y).float().sum().item()
                total += Y.numel()

        avg_loss = total_loss / len(self.dataloader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def evaluate(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        loss, accuracy = self._evaluate()
        print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
def main():
    dataset = AudioMidiDataset.get_evaluation_dataset()
    evaluator = ModelEvaluator(dataset)
    evaluator.evaluate()

if __name__ == "__main__":
    dataset = AudioMidiDataset(mel_path="./data/single_sample/input_mels.npy", label_path="./data/single_sample/output_labels.npy")
    evaluator = ModelEvaluator(dataset)
    evaluator.evaluate()