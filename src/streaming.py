import tempfile
import asyncio
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO


async def convert_audio_bytes(audio_bytes, content_type=None):
    """
    Convert audio bytes to PCM format.

    Args:
        audio_bytes: Raw audio bytes
        content_type: MIME type of the audio (e.g., 'audio/wav', 'audio/mp3')

    Returns:
        numpy array of float32 audio samples
    """
    # Run the conversion in a thread pool since it can be CPU-intensive
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _convert_audio_sync, audio_bytes, content_type
    )
    return result


def _convert_audio_sync(audio_bytes, content_type=None):
    """Synchronous implementation of audio conversion."""
    try:
        # Detect format from content type
        audio_format = None
        if content_type:
            if 'wav' in content_type:
                audio_format = 'wav'
            elif 'mp3' in content_type:
                audio_format = 'mp3'
            elif 'webm' in content_type:
                audio_format = 'webm'
            elif 'ogg' in content_type:
                audio_format = 'ogg'
            elif 'flac' in content_type:
                audio_format = 'flac'

        # Special handling based on format or try auto-detection
        if audio_format:
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format=audio_format)
        else:
            # Try to auto-detect format
            audio = AudioSegment.from_file(BytesIO(audio_bytes))

        # Convert to WAV at 16kHz mono
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)

        # Convert to raw PCM bytes
        wav_io = BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)

        # Skip WAV header (44 bytes) and read raw PCM
        wav_io.seek(44)
        pcm_data = wav_io.read()

        # Convert to float32 normalized to [-1, 1]
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0

        return samples

    except Exception as e:
        print(f"Error converting audio: {e}")
        raise


def write_audio_file(audio_data, filename, sample_rate=16000):
    """
    Write audio data to a file.

    Args:
        audio_data: Audio samples (numpy array)
        filename: Output filename
        sample_rate: Sample rate in Hz
    """
    try:
        sf.write(filename, audio_data, sample_rate)
        return True
    except Exception as e:
        print(f"Error writing audio file: {e}")
        return False