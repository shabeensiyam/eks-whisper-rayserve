import tempfile
import time

import numpy as np
import torch
import whisper
from ray import serve


@serve.deployment(
    ray_actor_options={"num_gpus": 1.0},
    max_ongoing_requests=10,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 2,
        "min_replicas": 1,
        "max_replicas": 10,
    }
)
class WhisperASR:
    def __init__(self, model_name="large-v3"):
        """Initialize the Whisper ASR model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model {model_name} on {self.device}")

        # Load the model
        self.model = whisper.load_model(model_name).to(self.device)

        # Set default options
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

        print("Whisper model loaded successfully")

    async def transcribe(self, audio_data: [bytes, np.ndarray], language=None, prompt=None, additional_options=None):
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

        # Set up options
        options = self.default_options.copy()
        if language:
            options["language"] = language
        if prompt:
            options["initial_prompt"] = prompt
        if additional_options:
            options.update(additional_options)

        try:
            # Use torch.amp for better performance with mixed precision
            with torch.amp.autocast(device_type=self.device.type, enabled=options.get("fp16", False)):
                # If audio_data is bytes, convert to numpy array
                if isinstance(audio_data, bytes):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                        temp_file.write(audio_data)
                        temp_file.flush()
                        result = self.model.transcribe(temp_file.name, **options)
                else:
                    # If it's already a numpy array
                    result = self.model.transcribe(audio_data, **options)

            processing_time = time.time() - start_time

            # Format response to be consistent
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
            print(f"Error in transcription: {e}")
            return {
                "error": str(e),
                "text": "",
                "processing_time": time.time() - start_time
            }

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
            additional_options=options.get("whisper_options")
        )
