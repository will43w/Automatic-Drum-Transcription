import numpy as np
from audiomentations import Compose, Gain, PitchShift, AddGaussianNoise, LowPassFilter, HighPassFilter, ClippingDistortion, BandPassFilter
from typing import Tuple

SAMPLE_RATE = 22050

# Fixed named effect chains with randomized parameters inside
EFFECT_CHAINS = {
    "jazz_club": Compose([
        LowPassFilter(min_cutoff_freq=4000.0, max_cutoff_freq=7000.0, p=1.0),
        #Gain(min_gain_db=-1, max_gain_db=1, p=1.0)
    ]),

    "lofi_dirty": Compose([
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=1.0),
        #ClippingDistortion(min_percentile_threshold=60, max_percentile_threshold=80, p=1.0),
        LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1.0)
    ]),

    "fusion_modern": Compose([
        HighPassFilter(min_cutoff_freq=60.0, max_cutoff_freq=120.0, p=1.0),
        #Gain(min_gain_db=-1, max_gain_db=1, p=1.0)
    ]),

    "studio_dry": Compose([
        BandPassFilter(min_center_freq=300.0, max_center_freq=5000.0, p=1.0),
        #Gain(min_gain_db=-1, max_gain_db=1, p=1.0)
    ])
}


def random_effect_chain():
    """Fully randomized effect chain"""
    return Compose([
        PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.5),
        #Gain(min_gain_in_db=-1, max_gain_in_db=1, p=1.0),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.8),
        np.random.choice([
            LowPassFilter(min_cutoff_freq=2000.0, max_cutoff_freq=6000.0, p=0.5),
            HighPassFilter(min_cutoff_freq=60.0, max_cutoff_freq=300.0, p=0.5),
            BandPassFilter(min_center_freq=300.0, max_center_freq=4000.0, p=0.5),
            #ClippingDistortion(min_percentile_threshold=60, max_percentile_threshold=95, p=0.5),
        ])
    ])


def apply_audio_effects(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, mode="mixed") -> Tuple[np.ndarray, str]:
    """
    Apply audio effects to a waveform.

    Parameters:
        audio (np.ndarray): Raw audio data (mono or stereo)
        sample_rate (int): Audio sample rate
        mode (str): One of ['fixed', 'random', 'mixed']

    Returns:
        np.ndarray: Processed audio
    """
    
    chain_name = ""

    if mode == "fixed":
        chain_name = np.random.choice(list(EFFECT_CHAINS.keys()))
        chain = EFFECT_CHAINS[chain_name]
    elif mode == "random":
        chain = random_effect_chain()
        chain_name = "random"
    elif mode == "mixed":
        if np.random.rand() < 0.3:
            chain = random_effect_chain()
            chain_name = "random"
        else:
            chain_name = np.random.choice(list(EFFECT_CHAINS.keys()))
            chain = EFFECT_CHAINS[chain_name]
    else:
        raise ValueError("mode must be one of ['fixed', 'random', 'mixed']")

    return chain(audio, sample_rate=sample_rate), chain_name
