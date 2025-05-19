import torch
import torch.nn as nn

class AudioToMidiModel(nn.Module):
    def __init__(self, n_mels=128, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # x: (B, T, M) => CNN expects (B, M, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # Back to (B, T, F)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return torch.sigmoid(self.out(x))
