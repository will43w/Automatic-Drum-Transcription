import numpy as np
import torch
from torch.utils.data import Dataset

class DrumTranscriptionDataset(Dataset):
    def __init__(self, mel_path="./data/input_mels.npy", label_path="./data/output_labels.npy"):
        self.mels = np.load(mel_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        x = torch.tensor(self.mels[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
