import torch
import torch.nn as nn

class AudioToMidiModel(nn.Module):
    def __init__(self, n_mels=128, hidden_dim=128, n_classes=10, cnn_channels=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, cnn_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=cnn_channels, hidden_size=hidden_dim,
            num_layers=2, 
            batch_first=True, 
            bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (B, T, M) → CNN expects (B, M, T)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)           # (B, C, T)
        x = x.permute(0, 2, 1)    # (B, T, C)
        x, _ = self.rnn(x)        # (B, T, 2*hidden)
        x = self.fc(x)            # (B, T, n_classes)
        return x
    
    def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        return preds
