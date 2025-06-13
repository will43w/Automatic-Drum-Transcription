from typing import Tuple
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import librosa

from .constants import OUTPUT_CLASS_TO_MIDI_PITCH, HOP_LENGTH
from .audio_to_midi_converter import AudioToMidiConverter
from .model.model import AudioToMidiModel

class MidiVisualizer:
    def __init__(self, midi: pretty_midi.PrettyMIDI):
        self.midi = midi

    def _show_pianoroll(self, fs: int = 100, max_time: float = 10.0) -> None:
        piano_roll = self.midi.get_piano_roll(fs=fs)

        print("Piano roll shape:", piano_roll.shape)
        print("Max velocity:", np.max(piano_roll))
        print("Min velocity (non-zero):", np.min(piano_roll[piano_roll > 0]) if np.any(piano_roll > 0) else "None")
    
        if piano_roll.shape[1] == 0:
            print("Piano roll is empty — no time steps to display.")
            return
        
        plt.figure(figsize=(12, 4))
        plt.imshow(
            piano_roll,
            aspect='auto',
            origin='lower',
            cmap='gray_r',
            extent=[0, piano_roll.shape[1] / fs, 0, piano_roll.shape[0]]
        )
        plt.xlabel("Time (s)")
        plt.ylabel("MIDI Pitch")
        plt.title("MIDI Piano Roll")
        plt.colorbar(label="Velocity")
        plt.tight_layout()
        plt.show()

    def visualize(self):
        self._show_pianoroll()

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(file_path, sr=None)  # Keep native sample rate
    return audio, sample_rate

if __name__ == "__main__":
    model = AudioToMidiModel()
    model.load()
    converter = AudioToMidiConverter(model)

    audio, sample_rate = load_audio_file("/Users/william.hafner/dev/audio_samples/SONOR_/sample_1.mp3")
    midi = converter.convert(audio=audio, audio_sample_rate=sample_rate)

    # midi.write("/Users/william.hafner/dev/audio_samples/SONOR_/sample_1.midi")

    visualizer = MidiVisualizer(midi)
    visualizer.visualize()