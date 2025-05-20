import asyncio
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np


# Check if FFmpeg is installed and set path if needed
def is_ffmpeg_available():
    """Check if FFmpeg is available in the system."""
    try:
        # Check if ffmpeg is in PATH
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False


# Global variable to track FFmpeg availability
FFMPEG_AVAILABLE = is_ffmpeg_available()
print(f"FFmpeg available: {FFMPEG_AVAILABLE}")


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
        # For WAV files, try direct parsing first (doesn't need FFmpeg)
        if content_type and 'wav' in content_type:
            try:
                audio_data, _ = parse_wav_bytes(audio_bytes)
                if audio_data is not None:
                    return audio_data
            except Exception as e:
                print(f"Direct WAV parsing failed: {e}, falling back to other methods")

        # If FFmpeg is available, use it
        if FFMPEG_AVAILABLE:
            return _convert_with_ffmpeg(audio_bytes)

        # If all else fails and it looks like a WAV file, try a basic approach
        if content_type and 'wav' in content_type:
            return _basic_wav_processing(audio_bytes)

        # At this point, we can't process the audio
        raise RuntimeError("Cannot process audio: FFmpeg is not available and the audio is not in WAV format or WAV parsing failed")

    except Exception as e:
        print(f"Error converting audio: {e}")
        raise


def _convert_with_ffmpeg(audio_bytes):
    """Convert audio bytes using FFmpeg."""
    # Create a temporary file for the input audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as in_file:
        in_file.write(audio_bytes)
        in_file_path = in_file.name

    # Create a temporary file for the output audio
    out_file_path = in_file_path + '.pcm'

    try:
        # Use FFmpeg to convert to raw PCM at 16kHz mono
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-i', in_file_path,  # Input file
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Channels (mono)
            '-f', 's16le',  # Format (signed 16-bit little-endian)
            out_file_path  # Output file
        ]

        # Run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read the PCM file
        with open(out_file_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()

        # Convert to float32 normalized to [-1, 1]
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0

        return samples

    finally:
        # Clean up temporary files
        for file_path in [in_file_path, out_file_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing temporary file {file_path}: {e}")


def _basic_wav_processing(wav_bytes):
    """Basic processing for WAV files when FFmpeg is not available."""
    # Try to find the 'data' chunk
    data_index = wav_bytes.find(b'data')
    if data_index < 0:
        raise ValueError("WAV file doesn't contain 'data' chunk")

    # Skip the chunk header (8 bytes: 'data' + size)
    data_index += 8

    # Extract the PCM data
    pcm_data = wav_bytes[data_index:]

    # Convert to float32 normalized to [-1, 1]
    # Assumes 16-bit PCM
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    samples = samples / 32768.0

    return samples


def parse_wav_bytes(wav_bytes) -> Tuple[Optional[np.ndarray], int]:
    """
    Parse WAV file bytes directly without using external libraries.

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Minimum WAV header size
    if len(wav_bytes) < 44:
        raise ValueError("File too small to be a valid WAV")

    # Check RIFF header
    if wav_bytes[0:4] != b'RIFF':
        raise ValueError("Not a valid WAV file (RIFF signature missing)")

    # Check WAVE format
    if wav_bytes[8:12] != b'WAVE':
        raise ValueError("Not a valid WAV file (WAVE format missing)")

    # Find the 'fmt ' chunk
    fmt_index = wav_bytes.find(b'fmt ')
    if fmt_index < 0:
        raise ValueError("WAV format chunk not found")

    # Parse fmt chunk
    fmt_chunk_size = int.from_bytes(wav_bytes[fmt_index + 4:fmt_index + 8], byteorder='little')
    audio_format = int.from_bytes(wav_bytes[fmt_index + 8:fmt_index + 10], byteorder='little')
    num_channels = int.from_bytes(wav_bytes[fmt_index + 10:fmt_index + 12], byteorder='little')
    sample_rate = int.from_bytes(wav_bytes[fmt_index + 12:fmt_index + 16], byteorder='little')
    bits_per_sample = int.from_bytes(wav_bytes[fmt_index + 22:fmt_index + 24], byteorder='little')

    # Validate format (PCM only for now)
    if audio_format != 1:
        raise ValueError(f"Only PCM WAV supported (got format {audio_format})")

    # Find the 'data' chunk
    data_index = wav_bytes.find(b'data')
    if data_index < 0:
        raise ValueError("WAV data chunk not found")

    # Get data chunk size
    data_size = int.from_bytes(wav_bytes[data_index + 4:data_index + 8], byteorder='little')

    # Extract audio data
    audio_data_bytes = wav_bytes[data_index + 8:data_index + 8 + data_size]

    # Convert to numpy array
    if bits_per_sample == 16:
        dtype = np.int16
    elif bits_per_sample == 32:
        dtype = np.int32
    elif bits_per_sample == 8:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

    # Create numpy array
    audio_array = np.frombuffer(audio_data_bytes, dtype=dtype)

    # Normalize to float32 in range [-1, 1]
    if dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0
    elif dtype == np.uint8:
        audio_array = (audio_array.astype(np.float32) - 128) / 128.0

    # Handle multiple channels if needed
    if num_channels > 1:
        # Reshape into [frames, channels]
        audio_array = audio_array.reshape(-1, num_channels)
        # Average all channels to mono
        audio_array = np.mean(audio_array, axis=1)

    return audio_array, sample_rate


def write_audio_file(audio_data, filename, sample_rate=16000):
    """
    Write audio data to a WAV file.

    Args:
        audio_data: Audio samples (numpy array)
        filename: Output filename
        sample_rate: Sample rate in Hz
    """
    try:
        # Convert float32 to int16
        samples = (audio_data * 32767).astype(np.int16)

        # Create a WAV file manually
        with open(filename, 'wb') as f:
            # Write WAV header
            f.write(b'RIFF')
            f.write((36 + len(samples) * 2).to_bytes(4, byteorder='little'))  # File size
            f.write(b'WAVE')

            # Write format chunk
            f.write(b'fmt ')
            f.write((16).to_bytes(4, byteorder='little'))  # Chunk size
            f.write((1).to_bytes(2, byteorder='little'))  # Audio format (PCM)
            f.write((1).to_bytes(2, byteorder='little'))  # Num channels
            f.write(sample_rate.to_bytes(4, byteorder='little'))  # Sample rate
            f.write((sample_rate * 2).to_bytes(4, byteorder='little'))  # Byte rate
            f.write((2).to_bytes(2, byteorder='little'))  # Block align
            f.write((16).to_bytes(2, byteorder='little'))  # Bits per sample

            # Write data chunk
            f.write(b'data')
            f.write((len(samples) * 2).to_bytes(4, byteorder='little'))  # Chunk size
            f.write(samples.tobytes())

        return True
    except Exception as e:
        print(f"Error writing audio file: {e}")
        return False
