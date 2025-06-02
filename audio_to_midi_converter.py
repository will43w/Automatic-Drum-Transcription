import torch
import numpy as np
import pretty_midi
import librosa

from model import AudioToMidiModel
from constants import SAMPLE_RATE, N_MELS, OUTPUT_CLASS_TO_MIDI_PITCH, HOP_SECONDS

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

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel).T  # (T, M)
        return torch.tensor(mel_db).unsqueeze(0).float()

    def _predict(self, mel: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(mel)
        return AudioToMidiModel.logits_to_predictions(logits).squeeze(0) # (T, n_classes), squeeze out batch dimension for inference

    def _predictions_to_midi(self, predictions: torch.Tensor) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI()
        drum_track = pretty_midi.Instrument(program=0, is_drum=True)

        T, C = predictions.shape

        for t in range(T):
            time_sec = t * HOP_SECONDS
            for c in range(C):
                if predictions[t, c]:
                    pitch = OUTPUT_CLASS_TO_MIDI_PITCH.get(c)
                    if pitch is not None:
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=pitch,
                            start=time_sec,
                            end=time_sec + 0.05
                        )
                        drum_track.notes.append(note)

        midi.instruments.append(drum_track)
        return midi

    def convert(self, audio: np.ndarray, audio_sample_rate: int) -> pretty_midi.PrettyMIDI:
        mel = self._audio_to_mel(audio, audio_sample_rate)
        predictions = self._predict(mel)
        midi = self._predictions_to_midi(predictions)
        return midi
