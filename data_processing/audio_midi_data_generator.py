from typing import List

import os
import random
import numpy as np
import pretty_midi
import librosa
import tqdm
import glob

from audio_effect_chains import apply_audio_effects
from save_audio_as_mp3 import save_audio_as_mp3

if not hasattr(np, 'complex'):
    np.complex = complex

class AudioMidiData:
    def __init__(self, mels: List[np.ndarray] = [], labels: List[np.ndarray] = []):
        self.mels = mels
        self.labels = labels
    
    def save(self, input_data_path: str, output_data_path: str):
        # Pad to max length
        max_len = max(m.shape[0] for m in self.mels)
        n_samples = len(self.mels)
        assert(n_samples == len(self.labels))
        n_mels = self.mels[0].shape[1] # mels is (samples, time, mels)
        n_classes = self.labels[0].shape[1] # labels is (samples, time, labels)

        X = np.zeros((n_samples, max_len, n_mels), dtype=np.float32)
        Y = np.zeros((n_samples, max_len, n_classes), dtype=np.float32)

        for i in range(n_samples):
            X[i, :self.mels[i].shape[0], :] = self.mels[i]
            Y[i, :self.labels[i].shape[0], :] = self.labels[i]
        
        np.save(input_data_path, X)
        np.save(output_data_path, Y)
        print(f"Saved {len(X)} examples to disk.")

class AudioMidiDataGenerator:
    SAMPLE_RATE = 22050
    FRAME_LENGTH_MS = 10
    HOP_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0)
    N_MELS = 128
    AUDIO_DURATION = 10
    N_SAMPLES_TRAINING = 100_000
    N_SAMPLES_EVALUATION = 200
    SOUNDFONTS = glob.glob("./data_processing/soundfonts/*.sf2")

    MIDI_PITCH_TO_OUTPUT_CLASS = {
        36: 0, # Kick
        38: 1, # Snare
        42: 2, # Closed Hi Hat
        44: 3, # Pedal Hi-hat
        46: 4, # Open Hi Hat
        49: 5, # Crash
        57: 5,
        51: 6, # Ride
        59: 6, 
        53: 7, # Ride Bell
    }

    NUM_CLASSES = len(MIDI_PITCH_TO_OUTPUT_CLASS)

    NON_DRUM_MIDI_INSTRUMENTS = {
        "piano": list(range(0, 8)), # The numbers in each list correspond to distinct instrument sounds in the relevant instrument family, i.e., piano here
        "guitar": list(range(24, 32)),
        "bass": list(range(32, 40)),
        "trumpet": list(range(56, 60)),
        "saxophone": list(range(64, 72)),
    }
    NON_DRUM_PITCH_RANGE = (48, 84) # C3 to C6

    def __init__(self):
        self.audio_midi_data = AudioMidiData()

    def _select_random_midi_program(self, family: str) -> int:
        return random.choice(self.NON_DRUM_MIDI_INSTRUMENTS[family])

    def _generate_random_drum_track(self, duration: float) -> pretty_midi.Instrument:
        tempo = 40 + random.random() * (240 - 40) # Anywhere from 40 BPM to 240 BPM
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        for t in np.arange(0, duration, tempo / 60 / (3 * 4 * 5)): # Divide each beat up to sixteenth notes in 3, 4, and 5
            for pitch in self.MIDI_PITCH_TO_OUTPUT_CLASS.keys():
                random_roll = random.random()
                if random_roll < 0.05:
                    drum.notes.append(pretty_midi.Note(
                        velocity = random.randint(20, 127),
                        pitch=pitch,
                        start=t,
                        end=t + 0.05
                    ))
        return drum

    def _generate_random_melodic_track(self, duration: float, family: str) -> pretty_midi.Instrument:
        program = self._select_random_midi_program(family)
        instrument = pretty_midi.Instrument(program=program, is_drum=False)
        note_duration = random.choice([0.25, 0.5, 1.0, 0.125, 0.3333, 0.6666])
        
        t = 0.0
        while t < duration:
            pitch = random.randint(*self.NON_DRUM_PITCH_RANGE)
            #velocity = random.randint(20, 127)
            velocity = 0
            end = min(t + note_duration, duration)
            instrument.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=t,
                end=end
            ))
            t += note_duration * random.uniform(0.8, 1.2)

        return instrument

    def _generate_full_band_midi(self, duration: float = AUDIO_DURATION) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(self._generate_random_drum_track(duration))
        families = random.sample(list(self.NON_DRUM_MIDI_INSTRUMENTS.keys()), k=3)
        for family in families:
            pm.instruments.append(self._generate_random_melodic_track(duration, family))
        return pm

    def _synthesize_midi(self, pm: pretty_midi.PrettyMIDI, sf2_path: str = None) -> np.ndarray:
        return pm.fluidsynth(fs=self.SAMPLE_RATE, sf2_path=sf2_path)

    def _extract_log_mel(self, audio: np.ndarray):
        mel = librosa.feature.melspectrogram(y=audio, sr=self.SAMPLE_RATE, n_mels=self.N_MELS, hop_length=self.HOP_LENGTH)
        return librosa.power_to_db(mel).T # (frames, n_mels)

    def _midi_to_labels(self, pm: pretty_midi.PrettyMIDI, n_frames: int) -> np.ndarray:
        labels = np.zeros((n_frames, self.NUM_CLASSES), dtype=np.float32)
        for instrument in pm.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    class_index = self.MIDI_PITCH_TO_OUTPUT_CLASS.get(note.pitch)
                    if class_index is not None:
                        frame_index = int(note.start * self.SAMPLE_RATE / self.HOP_LENGTH)
                        if 0 <= frame_index < n_frames:
                            labels[frame_index, class_index] = 1
        return labels
    
    def _select_soundfont(self) -> str:
        return random.choice(self.SOUNDFONTS)

    def generate(self, n_samples: int) -> AudioMidiData:
        sample_mels = []
        sample_labels = []
        for i in tqdm.tqdm(range(n_samples), desc="Generating dataset"):
            pm = self._generate_full_band_midi()
            sf2 = self._select_soundfont()
            audio = self._synthesize_midi(pm, sf2) # audio = synthesize_midi(pm, sf2)
            audio, _ = apply_audio_effects(audio, self.SAMPLE_RATE, mode="mixed")
            mel = self._extract_log_mel(audio)
            n_frames = mel.shape[0]
            labels = self._midi_to_labels(pm, n_frames)

            sample_mels.append(mel)
            sample_labels.append(labels)

        return AudioMidiData(sample_mels, sample_labels)

def main():
    data_generator = AudioMidiDataGenerator()
    training_data = data_generator.generate(n_samples=1)
    training_data.save(
        input_data_path="./data/single_sample/input_mels.npy", 
        output_data_path="./data/single_sample/output_labels.npy")
    
    # evaluation_data = data_generator.generate(n_samples=200)
    # evaluation_data.save(
    #     input_data_path="./data/evaluation2/input_mels.npy", 
    #     output_data_path="./data/evaluation2/output_labels.npy")

if __name__ == "__main__":
    main()