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
            # If audio_data is bytes, save to temp file
            if isinstance(audio_data, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(audio_data)
            else:
                # Handle numpy array case if needed
                raise ValueError("Audio data must be bytes")

            # Load appropriate model
            model = self._load_model(model_size)

            # Perform transcription
            logger.info(f"Starting transcription with {model_size} model")
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
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

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
