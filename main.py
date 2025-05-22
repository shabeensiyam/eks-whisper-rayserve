import logging

import ray
from ray import serve

from app import create_app

logger = logging.getLogger("whisper_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

if __name__ == "__main__":
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(address="auto", namespace="whisper-streaming")

    # Log Ray version
    logger.info(f"Ray version: {ray.__version__}")

    # Start Ray Serve
    serve.start(detached=True)

    # Create and deploy the application
    app = create_app()
    serve.run(app, name="whisper_streaming")

    logger.info("Whisper ASR service is running")
    logger.info("API available at: http://localhost:8000")
    logger.info("API documentation: http://localhost:8000/docs")
    logger.info("WebSocket endpoint: ws://localhost:8000/stream")
    logger.info("HTTP endpoint: http://localhost:8000/transcribe")

    # Keep the process running
    try:
        logger.info("Press Ctrl+C to exit...")
        # Use a signal pause to keep running
        import signal

        signal.pause()
    except (KeyboardInterrupt, ImportError):
        # Fallback if signal.pause() is not available
        import time

        while True:
            time.sleep(60)
