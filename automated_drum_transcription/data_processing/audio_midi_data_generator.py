from typing import List

import os
import random
import numpy as np
import pretty_midi
import librosa
import tqdm
import glob

from .audio_effect_chains import apply_audio_effects
from .save_audio_as_mp3 import save_audio_as_mp3

from ..constants import MIDI_PITCH_TO_OUTPUT_CLASS, HOP_LENGTH, SAMPLE_RATE, N_MELS, DEFAULT_SOUNDFONTS_PATH
from .data_transformer import DataTransformer

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
    AUDIO_DURATION = 10
    N_SAMPLES_TRAINING = 100_000
    N_SAMPLES_EVALUATION = 200
    SOUNDFONTS = glob.glob(f"{DEFAULT_SOUNDFONTS_PATH}/*.sf2")

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

    def _generate_random_drum_track(self, duration: float = AUDIO_DURATION) -> pretty_midi.Instrument:
        tempo = 40 + random.random() * (240 - 40) # Anywhere from 40 BPM to 240 BPM
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        for t in np.arange(0, duration, tempo / 60 / (3 * 4 * 5)): # Divide each beat up to sixteenth notes in 3, 4, and 5
            for pitch in MIDI_PITCH_TO_OUTPUT_CLASS.keys():
                random_roll = random.random()
                if random_roll < 0.05:
                    drum.notes.append(pretty_midi.Note(
                        velocity = random.randint(20, 127),
                        pitch=pitch,
                        start=t,
                        end=t + 0.1 * random.uniform(0.2, 2.0) # Random duration between 80% and 120% of 100ms
                    ))
        return drum

    def _generate_random_melodic_track(self, duration: float, family: str) -> pretty_midi.Instrument:
        program = self._select_random_midi_program(family)
        instrument = pretty_midi.Instrument(program=program, is_drum=False)
        note_duration = random.choice([0.25, 0.5, 1.0, 2.0, 0.125, 0.3333, 0.6666])
        
        t = 0.0
        while t < duration:
            pitch = random.randint(*self.NON_DRUM_PITCH_RANGE)
            velocity = random.randint(20, 127)
            end = min(t + note_duration, duration)
            instrument.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=t,
                end=end
            ))
            t += note_duration * random.uniform(0.8, 1.2)

        return instrument
    
    def _instrument_to_midi(self, instrument: pretty_midi.Instrument):
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(instrument)
        return pm
    
    def _generate_drum_midi(self, duration: float = AUDIO_DURATION) -> pretty_midi.PrettyMIDI:
        return self._instrument_to_midi(self._generate_random_drum_track(duration=duration))

    def _generate_full_band_midi(self, duration: float = AUDIO_DURATION) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(self._generate_random_drum_track(duration))
        families = random.sample(list(self.NON_DRUM_MIDI_INSTRUMENTS.keys()), k=3)
        for family in families:
            pm.instruments.append(self._generate_random_melodic_track(duration, family))
        return pm
    
    def _generate_melodic_midi(self, duration: float = AUDIO_DURATION):
        pm = pretty_midi.PrettyMIDI()
        families = random.sample(list(self.NON_DRUM_MIDI_INSTRUMENTS.keys()), k=3)
        for family in families:
            pm.instruments.append(self._generate_random_melodic_track(duration, family))
        return pm

    def _synthesize_midi(self, pm: pretty_midi.PrettyMIDI, sf2_path: str = None) -> np.ndarray:
        return pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=sf2_path)

    def _extract_log_mel(self, audio: np.ndarray):
        return DataTransformer.audio_to_mels(audio)

    def _midi_to_labels(self, pm: pretty_midi.PrettyMIDI, n_frames: int) -> np.ndarray:
        return DataTransformer.midi_to_labels(pm, n_frames)
    
    def _select_soundfont(self) -> str:
        return random.choice(self.SOUNDFONTS)
    
    def generate_audio_sample(self) -> np.ndarray:
        drum_midi = self._generate_drum_midi()
        sf2 = self._select_soundfont()
        drum_audio = self._synthesize_midi(drum_midi, sf2_path=sf2)

        print(sf2)

        melodic_midi = self._generate_melodic_midi()
        melodic_audio = self._synthesize_midi(melodic_midi)

        snr_db = random.uniform(-3, 3)
        combined_audio = DataTransformer.combine_audio(drum_audio, melodic_audio, snr_db=snr_db)
        return combined_audio

    def generate(self, n_samples: int) -> AudioMidiData:
        sample_mels = []
        sample_labels = []
        for i in tqdm.tqdm(range(n_samples), desc="Generating dataset"):
            drum_midi = self._generate_drum_midi()
            sf2 = self._select_soundfont()
            drum_audio = self._synthesize_midi(drum_midi, sf2_path=sf2)

            melodic_midi = self._generate_melodic_midi()
            melodic_audio = self._synthesize_midi(melodic_midi)

            snr_db = random.uniform(-3, 3)
            combined_audio = DataTransformer.combine_audio(drum_audio, melodic_audio, snr_db=snr_db)

            # output_path = f"./audio_samples/{os.path.basename(sf2)[0:6]}/sample_{i}.mp3"
            # print(output_path)
            save_audio_as_mp3(combined_audio, sample_rate=SAMPLE_RATE, output_path=f"./audio_samples/{os.path.basename(sf2)[0:6]}/sample_{i}.mp3")

            audio, _ = apply_audio_effects(audio, SAMPLE_RATE, mode="mixed")
            mel = self._extract_log_mel(audio)
            n_frames = mel.shape[0]
            labels = self._midi_to_labels(drum_midi, n_frames)

            sample_mels.append(mel)
            sample_labels.append(labels)

        return AudioMidiData(sample_mels, sample_labels)

def main():
    data_generator = AudioMidiDataGenerator()
    training_data = data_generator.generate(n_samples=1)
    # training_data.save(
    #     input_data_path="./data/single_sample/input_mels.npy", 
    #     output_data_path="./data/single_sample/output_labels.npy")
    
    # evaluation_data = data_generator.generate(n_samples=200)
    # evaluation_data.save(
    #     input_data_path="./data/evaluation2/input_mels.npy", 
    #     output_data_path="./data/evaluation2/output_labels.npy")

if __name__ == "__main__":
    main()