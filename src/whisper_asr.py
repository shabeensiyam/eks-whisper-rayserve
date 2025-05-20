# Configure logger
import logging
import os
import tempfile
import time

import torch
from ray import serve

logger = logging.getLogger("whisper_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


@serve.deployment(
    name="WhisperASR",
    ray_actor_options={"num_gpus": 1.0},
    max_ongoing_requests=10,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 2,
        "min_replicas": 1,
        "max_replicas": 10,
    }
)
class WhisperASR:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Whisper ASR on device: {self.device}")

        # Initialize with base model - will be replaced on first request
        self.models = {}
        self.current_model_size = None

        # Default options
        self.default_options = {
            "task": "transcribe",
            "language": None,  # Auto-detect language
            "temperature": 0,  # Greedy decoding
            "best_of": None,  # Only sample 1
            "beam_size": None,  # Disable beam search
            "patience": None,  # Default patience
            "length_penalty": None,
            "suppress_tokens": "-1",
            "initial_prompt": None,
            "condition_on_previous_text": True,
            "fp16": torch.cuda.is_available(),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }

        # Load the base model by default
        self._load_model("base")
        logger.info("Whisper ASR initialized and ready")

    def _load_model(self, model_size):
        """Load a Whisper model of the specified size if not already loaded"""
        if model_size not in self.models:
            # Import whisper here to avoid importing it in the parent process
            import whisper

            logger.info(f"Loading Whisper model: {model_size}")
            start_time = time.time()
            self.models[model_size] = whisper.load_model(model_size, device=self.device)
            logger.info(f"Model {model_size} loaded in {time.time() - start_time:.2f} seconds")

        self.current_model_size = model_size
        return self.models[model_size]

    async def transcribe(self, audio_data, language=None, prompt=None, additional_options=None):
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes or numpy array
            language: Optional language code (e.g., 'en', 'fr')
            prompt: Optional text to guide transcription
            additional_options: Additional Whisper options

        Returns:
            Dictionary with transcription results
        """
        start_time = time.time()
        temp_file_path = None

        # Parse options
        options = self.default_options.copy()
        if language:
            options["language"] = language
        if prompt:
            options["initial_prompt"] = prompt
        if additional_options:
            options.update(additional_options)

        # Determine model size
        model_size = options.pop("model_size", "base")

        try:
            # Add detailed logging
            logger.info(f"Audio data type: {type(audio_data)}, size: {len(audio_data)} bytes")
            if len(audio_data) > 20:
                # Log a sample of the data
                import binascii
                logger.info(f"First 20 bytes hex: {binascii.hexlify(audio_data[:20])}")

            # Handle audio data based on its format
            if isinstance(audio_data, bytes):
                import numpy as np
                from scipy.io import wavfile

                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file_path = temp_file.name

                    # Check if this looks like raw float32 data from WebSocket
                    if len(audio_data) % 4 == 0:  # float32 is 4 bytes per sample
                        try:
                            # Try to interpret as float32 data (browser's AudioBuffer)
                            samples = np.frombuffer(audio_data, dtype=np.float32)

                            # Create properly formatted WAV file
                            sample_rate = 16000  # Common sample rate for speech recognition

                            # Normalize if needed (usually the browser sends normalized data)
                            max_val = max(abs(samples.min()), abs(samples.max()))
                            if max_val > 0:
                                if max_val > 1.0:
                                    samples = samples / max_val  # Normalize if somehow > 1.0

                            # Convert float32 samples to int16 for WAV file
                            samples_int16 = (samples * 32767).astype(np.int16)

                            # Write WAV file with proper headers
                            wavfile.write(temp_file_path, sample_rate, samples_int16)
                            logger.info(f"Converted WebSocket float32 audio data to WAV: {temp_file_path}")
                        except Exception as e:
                            # Log the conversion error
                            logger.warning(f"Float32 conversion failed: {e}, trying direct write")

                            # Close the current file and create a new one
                            temp_file.close()
                            with open(temp_file_path, 'wb') as f:
                                f.write(audio_data)
                            logger.info(f"Wrote audio data directly to file: {temp_file_path}")
                    else:
                        # This is likely already a WAV file (from HTTP upload)
                        temp_file.write(audio_data)
                        logger.info(f"Wrote audio file data to: {temp_file_path}")
            else:
                # Handle numpy array case
                raise ValueError("Audio data must be bytes")

            # Load appropriate model
            model = self._load_model(model_size)

            # Perform transcription
            logger.info(f"Starting transcription with {model_size} model on file: {temp_file_path}")
            result = model.transcribe(temp_file_path, **options)
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")

            # Format response
            response = {
                "text": result["text"].strip(),
                "language": result.get("language", ""),
                "language_probability": result.get("language_probability", 0),
                "segments": [
                    {
                        "id": segment["id"],
                        "text": segment["text"].strip(),
                        "start": segment["start"],
                        "end": segment["end"],
                    }
                    for segment in result.get("segments", [])
                ],
                "processing_time": processing_time,
            }

            return response

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {
                "error": str(e),
                "text": "",
                "processing_time": time.time() - start_time
            }
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Error removing temp file {temp_file_path}: {e}")

    async def __call__(self, audio_data, options=None):
        """
        Main entry point for Ray Serve.

        Args:
            audio_data: Raw audio bytes or numpy array
            options: Dictionary with transcription options

        Returns:
            Dictionary with transcription results
        """
        options = options or {}
        return await self.transcribe(
            audio_data,
            language=options.get("language"),
            prompt=options.get("prompt"),
            additional_options=options
        )
