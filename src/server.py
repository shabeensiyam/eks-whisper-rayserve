import ray
from ray import serve

from transcription import TranscriptionServer
from whisper_asr import WhisperASR


def entrypoint():
    """Entry point for the Ray Serve application."""
    # Create deployments with explicit names
    whisper_deployment = WhisperASR.bind()
    transcription_server = TranscriptionServer.bind()

    # Return the ingress deployment
    return transcription_server


if __name__ == "__main__":
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto", namespace="whisper-streaming")

    # Start Ray Serve and deploy the application
    serve.run(
        entrypoint,
        name="whisper_app",
        route_prefix="/",
        host="0.0.0.0",
        port=8000,
        _blocking=False
    )

    print("Whisper streaming application deployed!")
    print("Available at: http://localhost:8000")
    print("API endpoints:")
    print("  - / (GET): Service information")
    print("  - /transcribe (POST): Audio file transcription")
    print("  - /stream (WebSocket): Real-time audio streaming")