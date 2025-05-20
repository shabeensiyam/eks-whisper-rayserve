import json
import logging
import os
import time

import websocket
from locust import User, task, between

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Audio sample to use for testing
AUDIO_SAMPLE_PATH = "assets/sample_audio.wav"


class WhisperStreamingUser(User):
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session."""
        self.ws = None
        self.session_id = f"user_{int(time.time())}_{id(self)}"
        self.audio_data = self.load_audio()

    def load_audio(self):
        """Load audio sample for testing."""
        try:
            # Check if audio file exists
            if not os.path.exists(AUDIO_SAMPLE_PATH):
                logger.error(f"Audio file not found: {AUDIO_SAMPLE_PATH}")
                return None

            # Load audio file in chunks
            with open(AUDIO_SAMPLE_PATH, "rb") as f:
                audio_data = f.read()

            # Split into 100ms chunks for streaming
            sample_rate = 16000
            chunk_size = int(0.1 * sample_rate * 4)  # 0.1 seconds of audio (4 bytes per sample)

            chunks = []
            for i in range(0, len(audio_data), chunk_size):
                chunks.append(audio_data[i:i + chunk_size])

            logger.info(f"Loading audio file")
            return chunks

        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None

    @task
    def stream_audio(self):
        """Stream audio to the Whisper service."""
        if not self.audio_data:
            logger.error("No audio data available")
            return

        # Connect to WebSocket
        host = os.environ.get("TARGET_HOST", "localhost")
        port = os.environ.get("TARGET_PORT", "8000")
        ws_url = f"ws://{host}:{port}/stream"

        try:
            # Record timing for WebSocket connection
            start_time = time.time()
            self.ws = websocket.create_connection(ws_url)
            connection_time = int((time.time() - start_time) * 1000)

            # Log connection performance
            self.environment.events.request.fire(
                request_type="Connect",
                name="Websocket",
                response_time=connection_time,
                response_length=0,
                exception=None,
            )

            # Send initial configuration
            config = {
                "language": "en",  # Fixed language for testing
                "chunk_duration": 5.0,
                "overlap": 0.5,
                "use_context": True
            }
            self.ws.send(json.dumps(config))

            # Log starting audio stream
            logger.info("Start sending audio")

            # Send audio chunks
            for i, chunk in enumerate(self.audio_data):
                start_time = time.time()
                self.ws.send(chunk, websocket.ABNF.OPCODE_BINARY)
                send_time = int((time.time() - start_time) * 1000)

                # Log chunk sending performance
                self.environment.events.request.fire(
                    request_type="Send",
                    name="Audio trunks",
                    response_time=send_time,
                    response_length=len(chunk),
                    exception=None,
                )

                # Simulate real-time streaming pacing
                time.sleep(0.1)  # Wait 100ms between chunks

                # Occasionally check for responses
                if i % 10 == 0:
                    self.ws.settimeout(0.1)
                    try:
                        response = self.ws.recv()
                        # Log received response
                        logger.info(response)
                        self.environment.events.request.fire(
                            request_type="Receive",
                            name="Response",
                            response_time=0,  # No request time for async responses
                            response_length=len(response),
                            exception=None,
                        )
                    except websocket.WebSocketTimeoutException:
                        pass

            # Wait for final responses
            self.ws.settimeout(5)
            try:
                while True:
                    response = self.ws.recv()
                    logger.info(response)
                    self.environment.events.request.fire(
                        request_type="Receive",
                        name="Response",
                        response_time=0,
                        response_length=len(response),
                        exception=None,
                    )
            except websocket.WebSocketTimeoutException:
                pass

        except Exception as e:
            logger.error(f"Error in WebSocket communication: {e}")
            self.environment.events.request.fire(
                request_type="WebSocket",
                name="Error",
                response_time=0,
                response_length=0,
                exception=e,
            )
        finally:
            if self.ws:
                self.ws.close()
