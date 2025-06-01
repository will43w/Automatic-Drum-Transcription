from typing import Tuple
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import librosa

from constants import OUTPUT_CLASS_TO_MIDI_PITCH, HOP_LENGTH
from audio_to_midi_converter import AudioToMidiConverter
from model import AudioToMidiModel

class MidiVisualizer:
    THRESHOLD = 0.5

    def __init__(self, midi: pretty_midi.PrettyMIDI):
        self.midi = midi

    def _show_pianoroll(self) -> None:
        piano_roll = self.midi.get_piano_roll(fs=100)
        plt.figure(figsize=(14, 6))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='inferno')
        plt.title("Predicted MIDI (Piano Roll)")
        plt.xlabel("Time (frames @ 100 fps)")
        plt.ylabel("MIDI Pitch")
        plt.colorbar(label="Velocity")
        plt.tight_layout()
        plt.show()

    def visualize(self):
        self._show_pianoroll()

    def save_midi(self, path: str) -> None:
        self.midi.write(path)

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(file_path, sr=None)  # Keep native sample rate
    return audio, sample_rate

if __name__ == "__main__":
    model = AudioToMidiModel()
    model.load()
    converter = AudioToMidiConverter(model)

    audio, sample_rate = load_audio_file("/Users/william.hafner/dev/audio_samples/SONOR_/sample_1.mp3")
    midi = converter.convert(audio=audio, audio_sample_rate=sample_rate)

    visualizer = MidiVisualizer(midi)
    visualizer.visualize()