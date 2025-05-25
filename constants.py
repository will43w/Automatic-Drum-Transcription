import torch

MODEL_VERSION = "2"
DEFAULT_MODEL_PATH = f"./model/v={MODEL_VERSION}/audio_midi_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")