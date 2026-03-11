from pydub import AudioSegment
from pydub.effects import normalize


def clean_audio(input_file: str, output_file: str) -> str:
    """Clean and normalize audio file"""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    normalized_audio = normalize(audio)
    normalized_audio.export(output_file, format="wav")
    return output_file
