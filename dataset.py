from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import os

class AudioMidiDataset(Dataset):
    TRAINING_DATA_PATH = "./data/train"
    EVALUATION_DATA_PATH = "./data/evaluate"
    INPUT_MELS_FILENAME = "input_mels.npy"
    OUTPUT_LABELS_FILENAME = "output_labels.npy"

    def __init__(self, mel_path="./data/input_mels.npy", label_path="./data/output_labels.npy"):
        self.mels = np.load(mel_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        x = torch.tensor(self.mels[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
    
    @staticmethod
    def get_training_dataset() -> AudioMidiDataset:
        return AudioMidiDataset(
            mel_path=os.path.join(AudioMidiDataset.TRAINING_DATA_PATH, AudioMidiDataset.INPUT_MELS_FILENAME),
            label_path=os.path.join(AudioMidiDataset.TRAINING_DATA_PATH, AudioMidiDataset.OUTPUT_LABELS_FILENAME))
    
    @staticmethod
    def get_evaluation_dataset() -> AudioMidiDataset:
        return AudioMidiDataset(
            mel_path=os.path.join(AudioMidiDataset.EVALUATION_DATA_PATH, AudioMidiDataset.INPUT_MELS_FILENAME),
            label_path=os.path.join(AudioMidiDataset.EVALUATION_DATA_PATH, AudioMidiDataset.OUTPUT_LABELS_FILENAME))
