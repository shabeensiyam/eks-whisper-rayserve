import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

# Configure logger
logger = logging.getLogger("whisper_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Create FastAPI app with documentation
app = FastAPI(
    title="Whisper ASR Service",
    description="Speech-to-text API using OpenAI's Whisper model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@serve.deployment(
    name="WhisperService",
    max_ongoing_requests=100,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 5,
        "min_replicas": 1,
        "max_replicas": 5,
    }
)
@serve.ingress(app)
class WhisperService:
    def __init__(self, whisper_asr: DeploymentHandle):
        """Initialize the service with a handle to the ASR model."""
        self.whisper_asr = whisper_asr
        self.active_connections: Dict[str, WebSocket] = {}
        logger.info("WhisperService initialized")

    @app.get("/")
    async def root(self):
        """API information endpoint."""
        return {
            "service": "Whisper ASR Service",
            "version": "1.0.0",
            "endpoints": {
                "/": "This information",
                "/transcribe": "Transcribe audio file via HTTP POST",
                "/stream": "WebSocket endpoint for real-time audio streaming",
                "/docs": "OpenAPI documentation",
                "/redoc": "ReDoc API documentation"
            }
        }

    @app.get("/health")
    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/transcribe")
    async def transcribe(
            self,
            file: UploadFile = File(...),
            model_size: str = Form("base"),
            language: Optional[str] = Form(None),
            task: str = Form("transcribe")
    ):
        """
        Transcribe audio file to text.

        - **file**: Audio file to transcribe
        - **model_size**: Whisper model size (tiny, base, small, medium, large)
        - **language**: Language code (e.g., en, fr, de) or None for auto-detection
        - **task**: Task to perform (transcribe or translate)
        """
        start_time = time.time()

        try:
            # Read audio file
            audio_data = await file.read()
            file_size = len(audio_data) / (1024 * 1024)  # Size in MB
            logger.info(f"Received audio file: {file.filename}, size: {file_size:.2f} MB")

            # Prepare options
            options = {
                "model_size": model_size,
                "language": language,
                "task": task,
            }

            # Process with Whisper ASR
            result = await self.whisper_asr.remote(audio_data, options)

            # Add total processing time
            result["total_processing_time"] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing audio: {str(e)}"}
            )

    @app.websocket("/stream")
    async def websocket_endpoint(self, websocket: WebSocket):
        """
        WebSocket endpoint for real-time audio streaming.

        Send configuration as JSON to initialize, then stream audio chunks.
        """
        await websocket.accept()
        connection_id = f"conn_{id(websocket)}"
        self.active_connections[connection_id] = websocket

        # Initialize buffers and state
        audio_buffer = []
        transcription_context = ""
        config = None

        try:
            # Get initial configuration
            config = await self._get_client_config(websocket)
            chunk_duration = config.get("chunk_duration", 5.0)
            sample_rate = config.get("sample_rate", 16000)
            chunk_size = int(chunk_duration * sample_rate)
            overlap_seconds = config.get("overlap", 0.5)
            overlap_samples = int(overlap_seconds * sample_rate)

            logger.info(f"WebSocket connection {connection_id} established with config: {config}")

            # Send acknowledgment
            await websocket.send_json({"status": "connected", "message": "Ready to receive audio"})

            while True:
                # Receive data from WebSocket
                data = await websocket.receive()

                if "bytes" in data:
                    # Process audio bytes
                    audio_chunk = np.frombuffer(data["bytes"], dtype=np.float32)
                    audio_buffer.extend(audio_chunk.tolist())

                    # Process when we have enough audio
                    if len(audio_buffer) >= chunk_size:
                        # Get chunk and convert to bytes for processing
                        chunk = np.array(audio_buffer[:chunk_size], dtype=np.float32)
                        # Convert to bytes (16-bit PCM)
                        chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()

                        # Prepare options
                        options = {
                            "model_size": config.get("model_size", "base"),
                            "language": config.get("language"),
                            "task": config.get("task", "transcribe"),
                            "initial_prompt": transcription_context if config.get("use_context", True) else None
                        }

                        # Process with Whisper ASR
                        result = await self.whisper_asr.remote(chunk_bytes, options)

                        # Update context if using context
                        if result and "text" in result and result["text"] and config.get("use_context", True):
                            transcription_context = result["text"][-500:]  # Keep last 500 chars

                        # Send result to client
                        await websocket.send_json(result)

                        # Slide window with overlap
                        audio_buffer = audio_buffer[chunk_size - overlap_samples:]

                elif "text" in data:
                    # Handle text commands from client
                    try:
                        message = json.loads(data["text"])
                        command = message.get("command", "")

                        if command == "reset":
                            # Reset buffers
                            audio_buffer = []
                            transcription_context = ""
                            await websocket.send_json({"status": "reset_complete"})

                        elif command == "config":
                            # Update config
                            new_config = message.get("config", {})
                            if config is None:
                                config = new_config
                            else:
                                config.update(new_config)

                            # Recalculate parameters if changed
                            if "chunk_duration" in new_config or "sample_rate" in new_config:
                                chunk_duration = config.get("chunk_duration", 5.0)
                                sample_rate = config.get("sample_rate", 16000)
                                chunk_size = int(chunk_duration * sample_rate)

                            if "overlap" in new_config:
                                overlap_seconds = config.get("overlap", 0.5)
                                overlap_samples = int(overlap_seconds * sample_rate)

                            await websocket.send_json({"status": "config_updated"})

                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON from client: {data['text']}")
                        await websocket.send_json({"status": "error", "message": "Invalid JSON"})

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {str(e)}")
            try:
                await websocket.send_json({"status": "error", "message": str(e)})
            except:
                pass
        finally:
            # Clean up connection
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

    async def _get_client_config(self, websocket: WebSocket) -> Dict[str, Any]:
        """Get initial configuration from WebSocket client."""
        try:
            # Wait for initial config with timeout
            data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            config = json.loads(data)
            logger.info(f"Received client config: {config}")
            return config
        except (asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get client config: {str(e)}, using defaults")
            # Default configuration
            return {
                "chunk_duration": 5.0,
                "sample_rate": 16000,
                "overlap": 0.5,
                "model_size": "base",
                "language": None,
                "task": "transcribe",
                "use_context": True
            }
