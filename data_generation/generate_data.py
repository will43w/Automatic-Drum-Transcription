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

# === CONFIGURATION ===
SAMPLE_RATE = 22050
FRAME_LENGTH_MS = 10
HOP_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0)
N_MELS = 128
AUDIO_DURATION = 10
N_SAMPLES = 1000

SOUNDFONTS = glob.glob("./data_generation/soundfonts/*.sf2")
print(SOUNDFONTS)

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

# === STORAGE ===
all_mels = []
all_labels = []

# === FUNCTIONS ===
def select_random_midi_program(family: str) -> int:
    return random.choice(NON_DRUM_MIDI_INSTRUMENTS[family])

def generate_random_drum_track(duration: float) -> pretty_midi.Instrument:
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
                    end=t + 0.05
                ))
    return drum

def generate_random_melodic_track(duration: float, family: str) -> pretty_midi.Instrument:
    program = select_random_midi_program(family)
    instrument = pretty_midi.Instrument(program=program, is_drum=False)
    note_duration = random.choice([0.25, 0.5, 1.0, 0.125, 0.3333, 0.6666])
    
    t = 0.0
    while t < duration:
        pitch = random.randint(*NON_DRUM_PITCH_RANGE)
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

def generate_full_band_midi(duration: float = AUDIO_DURATION) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(generate_random_drum_track(duration))
    families = random.sample(list(NON_DRUM_MIDI_INSTRUMENTS.keys()), k=3)
    for family in families:
        pm.instruments.append(generate_random_melodic_track(duration, family))
    return pm

def synthesize_midi(pm: pretty_midi.PrettyMIDI, sf2_path: str = None) -> np.ndarray:
    return pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=sf2_path)

def extract_log_mel(audio: np.ndarray):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel).T # (frames, n_mels)

def midi_to_labels(pm: pretty_midi.PrettyMIDI, n_frames: int) -> np.ndarray:
    labels = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)
    for instrument in pm.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                class_index = MIDI_PITCH_TO_OUTPUT_CLASS.get(note.pitch)
                if class_index is not None:
                    frame_index = int(note.start * SAMPLE_RATE / HOP_LENGTH)
                    if 0 <= frame_index < n_frames:
                        labels[frame_index, class_index] = 1
    return labels

# === GENERATE DATASET ===
if __name__ == "__main__":
    for i in tqdm.tqdm(range(N_SAMPLES), desc="Generating dataset"):
        pm = generate_full_band_midi()

        sf2 = random.choice(SOUNDFONTS)
        audio = synthesize_midi(pm, sf2) # audio = synthesize_midi(pm, sf2)
        save_audio_as_mp3(audio, SAMPLE_RATE, output_path=f"./audio_samples/without_effects_{i}_{sf2}.mp3")
        audio, chain_name = apply_audio_effects(audio, SAMPLE_RATE, mode="mixed")
        save_audio_as_mp3(audio, SAMPLE_RATE, output_path=f"./audio_samples/with_effects_{chain_name}_{i}_{sf2}.mp3")

        mel = extract_log_mel(audio)
        n_frames = mel.shape[0]
        labels = midi_to_labels(pm, n_frames)
        all_mels.append(mel)
        all_labels.append(labels)

    # Pad to max length
    max_len = max(m.shape[0] for m in all_mels)
    X = np.zeros((N_SAMPLES, max_len, N_MELS), dtype=np.float32)
    Y = np.zeros((N_SAMPLES, max_len, NUM_CLASSES), dtype=np.float32)
    for i in range(N_SAMPLES):
        X[i, :all_mels[i].shape[0], :] = all_mels[i]
        Y[i, :all_labels[i].shape[0], :] = all_labels[i]

    # Save
    np.save("../data/input_mels.npy", X)
    np.save("../data/output_labels.npy", Y)
    print(f"Saved {N_SAMPLES} examples to disk.")