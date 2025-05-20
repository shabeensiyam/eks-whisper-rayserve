import asyncio
import json
import logging
import time
from typing import Dict, Any

import numpy as np
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import WebSocketRoute, Route
from starlette.websockets import WebSocket, WebSocketDisconnect

from streaming import convert_audio_bytes, write_audio_file

logger = logging.getLogger("transcription_server")


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=100,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 5,
        "min_replicas": 1,
        "max_replicas": 5,
    }
)
class TranscriptionServer:
    def __init__(self):
        """Initialize the transcription server."""
        # Get reference to the ASR model deployment
        self.asr_handle: DeploymentHandle = serve.get_deployment_handle("WhisperASR")

        # Configure server
        self.sample_rate = 16000  # Expected audio sample rate
        self.channels = 1  # Mono audio

        # Create a Starlette application with routes
        self.app = Starlette(
            routes=[
                Route("/", self.index),
                Route("/transcribe", self.http_transcribe, methods=["POST"]),
                WebSocketRoute("/stream", self.websocket_endpoint)
            ]
        )

        # Keep track of active connections
        self.active_connections = 0
        self.max_connections = 100

        logger.info("TranscriptionServer initialized")

    async def index(self, request: Request):
        """Handle root endpoint requests."""
        return JSONResponse({
            "status": "running",
            "service": "Whisper Streaming ASR",
            "endpoints": {
                "/": "Service info (this response)",
                "/transcribe": "HTTP POST endpoint for audio file transcription",
                "/stream": "WebSocket endpoint for real-time audio streaming"
            }
        })

    async def http_transcribe(self, request: Request):
        """Handle HTTP POST requests for transcription."""
        start_time = time.time()

        try:
            # Check content type and get parameters
            content_type = request.headers.get("content-type", "")
            params = {}

            if request.query_params:
                params = dict(request.query_params)

            # Read the audio data
            audio_data = await request.body()

            # Convert to the right format if needed
            if "audio/" in content_type:
                # Convert from encoded audio to raw PCM
                audio_data = await convert_audio_bytes(audio_data, content_type)

            # Process the audio with Whisper
            result = await self.asr_handle.remote(audio_data, options={"language": params.get("language")})

            # Add timing information
            result["processing_time_total"] = time.time() - start_time

            return JSONResponse(result)

        except Exception as e:
            logger.error(f"Error processing HTTP request: {str(e)}")
            return JSONResponse(
                {"error": str(e), "status": "error"},
                status_code=500
            )

    async def websocket_endpoint(self, websocket: WebSocket):
        """Handle WebSocket connections for streaming audio."""
        # Check if we can accept the connection
        if self.active_connections >= self.max_connections:
            await websocket.close(code=1008, reason="Too many connections")
            return

        await websocket.accept()
        self.active_connections += 1

        # Get client configuration
        config = await self.get_client_config(websocket)

        # Initialize audio buffer
        audio_buffer = []
        transcription_context = ""
        chunk_duration = config.get("chunk_duration", 5.0)  # Process in 5-second chunks
        chunk_size = int(chunk_duration * self.sample_rate)
        overlap_seconds = config.get("overlap", 0.5)  # Half-second overlap
        overlap_samples = int(overlap_seconds * self.sample_rate)

        try:
            logger.info(f"WebSocket connection established, config: {config}")

            while True:
                # Receive audio data from WebSocket
                data = await websocket.receive()

                if "bytes" in data:
                    # Process audio bytes
                    audio_chunk = np.frombuffer(data["bytes"], dtype=np.float32)
                    audio_buffer.extend(audio_chunk.tolist())

                    # Process when we have enough audio
                    if len(audio_buffer) >= chunk_size:
                        # Get audio chunk and convert to bytes
                        chunk = np.array(audio_buffer[:chunk_size], dtype=np.float32)

                        # Debug: save audio chunk if needed
                        if config.get("save_audio", False):
                            write_audio_file(chunk, f"chunk_{int(time.time())}.wav", self.sample_rate)

                        # Create options for transcription
                        options = {
                            "language": config.get("language"),
                            "prompt": transcription_context if config.get("use_context", True) else None
                        }

                        # Send to Whisper ASR
                        result = await self.asr_handle.remote(chunk, options=options)

                        # Get transcription result
                        if result and "text" in result and result["text"]:
                            # Update context with recent transcription
                            transcription_context = result["text"][-500:]  # Keep last 500 chars as context

                            # Send result back to client
                            await websocket.send_json(result)

                        # Slide window with overlap
                        audio_buffer = audio_buffer[chunk_size - overlap_samples:]

                elif "text" in data:
                    # Handle text commands from client
                    try:
                        message = json.loads(data["text"])

                        if message.get("command") == "reset":
                            # Reset buffer and context
                            audio_buffer = []
                            transcription_context = ""
                            await websocket.send_json({"status": "reset_complete"})

                        elif message.get("command") == "config":
                            # Update config
                            new_config = message.get("config", {})
                            config.update(new_config)
                            await websocket.send_json({"status": "config_updated"})
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON: {data['text']}")

                # Add a small delay to avoid tight loops
                await asyncio.sleep(0.001)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {str(e)}")
            try:
                await websocket.close(code=1011, reason=str(e))
            except:
                pass
        finally:
            self.active_connections -= 1

    async def get_client_config(self, websocket: WebSocket) -> Dict[str, Any]:
        """Get initial configuration from WebSocket client."""
        try:
            # Wait for initial config (with timeout)
            data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            config = json.loads(data)
            return config
        except (asyncio.TimeoutError, json.JSONDecodeError):
            # Default config if none provided
            return {
                "chunk_duration": 5.0,
                "overlap": 0.5,
                "language": None,
                "use_context": True,
                "save_audio": False
            }

    async def __call__(self, request: Request):
        """Entry point for Ray Serve."""
        return await self.app(request)
