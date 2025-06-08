import pretty_midi
import numpy as np
import torch
import librosa

from constants import MIDI_PITCH_TO_OUTPUT_CLASS, OUTPUT_CLASS_TO_MIDI_PITCH, SAMPLE_RATE, HOP_LENGTH, HOP_SECONDS, N_MELS

class DataTransformer:
    def MidiToLabels(midi: pretty_midi.PrettyMIDI, n_frames: int) -> np.ndarray:
        labels = np.zeros((n_frames, len(MIDI_PITCH_TO_OUTPUT_CLASS)), dtype=np.float32)
        for instrument in midi.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    class_index = MIDI_PITCH_TO_OUTPUT_CLASS.get(note.pitch)
                    if class_index is not None:
                        frame_index = int(note.start * SAMPLE_RATE / HOP_LENGTH)
                        if 0 <= frame_index < n_frames:
                            labels[frame_index, class_index] = 1
        return labels
    
    def LabelsToMidi(predictions: torch.Tensor) -> pretty_midi.PrettyMIDI:
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
    
    def AudioToMels(audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
        return librosa.power_to_db(mel).T # (frames, n_mels)
    
    def CombineAudio(audio: np.ndarray, other_audio: np.ndarray, snr_db: float = 0.0) -> np.ndarray:
        # Trim to shortest
        min_len = min(len(audio), len(other_audio))
        audio = audio[:min_len]
        other_audio = other_audio[:min_len]

        # Normalize RMS
        def rms(x): 
            return np.sqrt(np.mean(x**2))
        
        drum_rms = rms(audio)
        non_drum_rms = rms(other_audio)

        desired_ratio = 10 ** (snr_db / 20.0)  # linear SNR

        # Scale other to get desired SNR
        scaled_other_audio = other_audio * (drum_rms / (non_drum_rms * desired_ratio + 1e-8))

        # Mix the two
        mix = audio + scaled_other_audio
        return mix / np.max(np.abs(mix))  # normalize to prevent clipping
