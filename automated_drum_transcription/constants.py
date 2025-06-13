import torch

MODEL_VERSION = "2"
DEFAULT_MODEL_PATH = f"./model/v={MODEL_VERSION}/audio_midi_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_AUDIO_SAMPLES_PATH = "./audio_samples"
DEFAULT_DATASET_PATH = "./data"
DEFAULT_SOUNDFONTS_PATH = "./automated_drum_transcription/data_processing/soundfonts"

SAMPLE_RATE = 22050
FRAME_LENGTH_MS = 10
HOP_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0)
N_MELS = 128
HOP_SECONDS = HOP_LENGTH / SAMPLE_RATE 

class Voice:
    KICK = 0
    SNARE = 1
    HI_HAT_STICK_CLOSED = 2
    HI_HAT_CHICK = 3
    HI_HAT_STICK_OPEN = 4
    CRASH = 5
    RIDE = 6
    BELL = 7

MIDI_PITCH_TO_OUTPUT_CLASS = {
    36: Voice.KICK,
    38: Voice.SNARE, 
    42: Voice.HI_HAT_STICK_CLOSED,
    44: Voice.HI_HAT_CHICK,
    46: Voice.HI_HAT_STICK_OPEN,
    49: Voice.CRASH,
    57: Voice.CRASH,
    51: Voice.RIDE,
    59: Voice.RIDE, 
    53: Voice.BELL,
}

OUTPUT_CLASS_TO_MIDI_PITCH = {
    Voice.KICK: 36,
    Voice.SNARE: 38,
    Voice.HI_HAT_STICK_CLOSED: 42,
    Voice.HI_HAT_CHICK: 44,
    Voice.HI_HAT_STICK_OPEN: 46,
    Voice.CRASH: 49,
    Voice.RIDE: 51,
    Voice.BELL: 53,
}