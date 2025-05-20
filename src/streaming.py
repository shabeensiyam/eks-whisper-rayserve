import tempfile
import asyncio
import os
import subprocess
import numpy as np
import soundfile as sf
from io import BytesIO


# Check if FFmpeg is installed and set path if needed
def ensure_ffmpeg():
    """Make sure FFmpeg is available."""
    try:
        # Check if ffmpeg is in PATH
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        print("FFmpeg found in PATH")
        return True
    except FileNotFoundError:
        print("FFmpeg not found in PATH, attempting to install...")
        try:
            # Try to install FFmpeg if not present (works on Debian/Ubuntu systems)
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
            print("FFmpeg installed successfully")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("WARNING: Could not install FFmpeg. Audio conversion may not work properly.")
            return False


# Run the check on module import
FFMPEG_AVAILABLE = ensure_ffmpeg()


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
        # If FFmpeg is not available, try a direct approach
        if not FFMPEG_AVAILABLE:
            # For WAV files, we can try a direct approach without FFmpeg
            if content_type and 'wav' in content_type:
                return _direct_wav_processing(audio_bytes)
            else:
                raise RuntimeError("FFmpeg is required for processing non-WAV audio formats")

        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as in_file:
            in_file.write(audio_bytes)
            in_file_path = in_file.name

        # Create a temporary file for the output audio
        out_file_path = in_file_path + '.wav'

        try:
            # Use FFmpeg to convert to WAV at 16kHz mono
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-i', in_file_path,  # Input file
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Channels (mono)
                '-f', 'wav',  # Format
                out_file_path  # Output file
            ]

            # Run the command
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Read the WAV file
            data, samplerate = sf.read(out_file_path, dtype='float32')

            return data

        finally:
            # Clean up temporary files
            for file_path in [in_file_path, out_file_path]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing temporary file {file_path}: {e}")

    except Exception as e:
        print(f"Error converting audio: {e}")
        raise


def _direct_wav_processing(wav_bytes):
    """Process WAV file directly without using FFmpeg."""
    try:
        # Skip WAV header (usually 44 bytes) and read raw PCM
        # This is a simplified approach that might not work for all WAV files
        pcm_start = 44
        pcm_data = wav_bytes[pcm_start:]

        # Convert to float32 normalized to [-1, 1]
        # Assumes 16-bit PCM
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0

        return samples

    except Exception as e:
        print(f"Error in direct WAV processing: {e}")
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
