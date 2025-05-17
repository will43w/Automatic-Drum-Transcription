from pydub import AudioSegment
import numpy as np
import os

def save_audio_as_mp3(audio: np.ndarray, sample_rate: int, output_path: str):
    """
    Convert a NumPy float32 mono audio array to an MP3 file.
    """
    # Ensure the audio is in the correct format (int16)
    audio = audio / np.max(np.abs(audio) + 1e-8)  # normalize
    audio_array_int16 = (audio * 32767).astype(np.int16)

    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_array_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit PCM
        channels=1       # mono
    )

    # Export to MP3
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_segment.export(output_path, format="mp3")
    #print(f"Saved MP3 to {output_path}")
