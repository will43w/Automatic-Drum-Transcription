import torch

DEFAULT_MODEL_PATH = "./model/audio_midi_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")