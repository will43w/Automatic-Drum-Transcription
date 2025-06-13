import torch
import numpy as np
import pretty_midi
import librosa

from .model.model import AudioToMidiModel
from .constants import SAMPLE_RATE, N_MELS, OUTPUT_CLASS_TO_MIDI_PITCH, HOP_SECONDS
from .data_processing.data_transformer import DataTransformer

class AudioToMidiConverter:
    THRESHOLD = 0.5
    
    def __init__(
        self,
        model: AudioToMidiModel,
    ):
        self.model = model

    def _audio_to_mel(self, audio: np.ndarray, audio_sample_rate: int) -> torch.Tensor:
        if audio_sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=audio_sample_rate, target_sr=SAMPLE_RATE)

        mels = DataTransformer.audio_to_mels(audio)
        return torch.tensor(mels).unsqueeze(0).float()

    def _predict(self, mel: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(mel)
        return AudioToMidiModel.logits_to_predictions(logits).squeeze(0) # (T, n_classes), squeeze out batch dimension for inference

    def _predictions_to_midi(self, predictions: torch.Tensor) -> pretty_midi.PrettyMIDI:
        return DataTransformer.labels_to_midi(predictions)

    def convert(self, audio: np.ndarray, audio_sample_rate: int) -> pretty_midi.PrettyMIDI:
        mel = self._audio_to_mel(audio, audio_sample_rate)
        predictions = self._predict(mel)
        midi = self._predictions_to_midi(predictions)
        return midi
