import ray
from ray import serve

from .transcription import TranscriptionServer
from .whisper_asr import WhisperASR


# For Ray 2.46.0, we need to explicitly create an Application
def create_app():
    """Create a Ray Serve application."""
    # Create the deployments
    whisper_deployment = WhisperASR.bind()
    transcription_server = TranscriptionServer.bind(whisper_deployment)

    # Return the transcription server as the application entry point
    return transcription_server


if __name__ == "__main__":
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto", namespace="whisper-streaming")

    print(f"Ray version: {ray.__version__}")

    # Start Ray Serve
    serve.start(detached=True)

    # Create the application
    app = create_app()

    # Deploy the application using serve.run
    serve.run(app, name="whisper_app")

    print("Whisper streaming application deployed!")
    print("Available at: http://localhost:8000")
    print("API endpoints:")
    print("  - / (GET): Service information")
    print("  - /transcribe (POST): Audio file transcription")
    print("  - /stream (WebSocket): Real-time audio streaming")
