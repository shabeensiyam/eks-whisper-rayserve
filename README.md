# Whisper ASR Streaming Service

This project implements a scalable, distributed speech recognition service using OpenAI's Whisper model, Ray Serve, and FastAPI, deployable on Amazon EKS.

## Features

- Real-time audio transcription via WebSockets
- Batch audio file transcription via HTTP API
- Model size selection (tiny, base, small, medium, large)
- Language selection or auto-detection
- Integrated Swagger UI documentation
- Scalable architecture with autoscaling
- WebSocket streaming for real-time transcription
- HTTP API for file uploads
- Deployable to Kubernetes (EKS)

## Architecture

The service is built with a clean, modular architecture using Ray Serve:

1. **WhisperASR Component**: Core speech recognition using OpenAI's Whisper
2. **WhisperService Component**: FastAPI service layer with HTTP and WebSocket endpoints
3. **Application Composition**: Components linked together using Ray Serve dependency injection

## Local Development

### Prerequisites

- Python 3.10+
- FFmpeg installed (required for audio processing)
- GPU with CUDA (recommended for larger models)

### Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd whisper-streaming
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service locally:
```bash
python -m main
```

4. Access the service:
   - API documentation: http://localhost:8000/docs
   - WebSocket endpoint: ws://localhost:8000/stream
   - HTTP endpoint: http://localhost:8000/transcribe

## Docker

Build and run with Docker:

```bash
# Build the image
docker build -t whisper-streaming:latest .

# Run the container
docker run -p 8000:8000 -p 8265:8265 whisper-streaming:latest
```

## Deployment to EKS

1. Build and push the Docker image to a container registry:
```bash
# Build the image
docker build -t your-registry/whisper-streaming:latest .

# Push to registry
docker push your-registry/whisper-streaming:latest
```

2. Update the image reference in `whisper-rayservice.yaml`:
```yaml
image: your-registry/whisper-streaming:latest
```

3. Deploy to EKS:
```bash
kubectl apply -f whisper-rayservice.yaml
```

4. Access the service:
```bash
# Port-forward to access locally
kubectl port-forward svc/whisper-streaming-serve-svc 8000:8000
kubectl port-forward svc/whisper-streaming-head-svc 8265:8265

# Access the Ray dashboard
open http://localhost:8265

# Access the API
open http://localhost:8000/docs
```

## API Usage

### HTTP API

Use the `/transcribe` endpoint for batch audio transcription:

```bash
curl -X POST -F "file=@audio.wav" -F "model_size=base" -F "language=en" http://localhost:8000/transcribe
```

### WebSocket API

Use the `/stream` endpoint for real-time streaming:

1. Connect to the WebSocket endpoint: `ws://localhost:8000/stream`
2. Send initial configuration as JSON:
```json
{
  "model_size": "base",
  "language": "en",
  "chunk_duration": 5.0,
  "sample_rate": 16000,
  "overlap": 0.5,
  "use_context": true
}
```
3. Stream audio chunks as binary data
4. Receive transcription results as they become available

## Web Client

A simple HTML client is provided for testing:

1. Open `whisper_client.html` in a browser
2. Use the "Live Streaming" tab for WebSocket streaming
3. Use the "File Upload" tab for HTTP file uploads

## Configuration

Key configuration options:

- **Model Size**: Choose from tiny, base, small, medium, large (GPU RAM requirements increase with model size)
- **Language**: Specify language code or leave empty for auto-detection
- **Task**: transcribe or translate (to translate to English)
- **Chunk Duration**: For streaming, how many seconds of audio to process at once
- **Use Context**: Whether to use previous transcription as context for better continuity

## License

MIT