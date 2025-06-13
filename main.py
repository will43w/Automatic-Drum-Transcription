from automated_drum_transcription.data_processing.audio_midi_data_generator import AudioMidiDataGenerator
from automated_drum_transcription.data_processing.save_audio_as_mp3 import save_audio_as_mp3

if __name__ == "__main__":
    data_generator = AudioMidiDataGenerator()
    audio = data_generator.generate_audio_sample()
    save_audio_as_mp3(audio, sample_rate=44100, output_path="./audio_samples/sample.mp3")