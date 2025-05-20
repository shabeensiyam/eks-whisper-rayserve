# Whisper Streaming on EKS with Ray Serve

This project implements a real-time audio transcription service using OpenAI's Whisper model, deployed with Ray Serve on Amazon EKS.

## Features

- Real-time audio transcription via WebSocket
- REST API for batch audio transcription
- Fixed-length chunking with overlap for continuous transcription
- Horizontal scaling with Ray Serve
- Kubernetes deployment with autoscaling

## Prerequisites

- Docker
- Kubernetes cluster (EKS recommended)
- kubectl configured to access your cluster
- GPU nodes in your cluster (for optimal performance)

## Getting Started

### Local Development

To run the application locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start a local Ray instance
ray start --head

# Start the server
python -m src.server

# Access the web client
# Open client/streaming_client.html in your browser
```

### Building and Deploying to EKS

1. Build and push the Docker image:

```bash
cd scripts
./build_image.sh
```

2. Deploy to EKS:

```bash
# Set your Hugging Face token if needed
export HUGGING_FACE_HUB_TOKEN=$(echo -n 'your-hugging-face-token' | base64)

# Deploy
cd scripts
./deploy.sh
```

3. Access the service:

```bash
# Forward the service port
kubectl port-forward svc/whisper-streaming-serve-svc 8000:8000

# Open the web client
# Open client/streaming_client.html in your browser
```

## API Reference

### WebSocket Endpoint

- **URL**: `/stream`
- **Protocol**: WebSocket
- **Purpose**: Real-time audio streaming and transcription

Initial configuration (send as JSON):

```json
{
  "language": "en",         // Optional, null for auto-detection
  "chunk_duration": 5.0,    // Audio chunk size in seconds
  "overlap": 0.5,           // Overlap between chunks in seconds
  "use_context": true       // Whether to use previous transcription as context
}
```

### HTTP Endpoint

- **URL**: `/transcribe`
- **Method**: POST
- **Content-Type**: audio/* (wav, mp3, etc.) or application/octet-stream
- **Purpose**: Batch transcription of audio files

Query parameters:
- `language`: Optional language code (e.g., "en", "fr")

## Architecture

This implementation uses:

- **OpenAI Whisper**: State-of-the-art speech recognition model
- **Ray Serve**: Framework for scalable model serving
- **WebSockets**: For real-time audio streaming
- **EKS**: Kubernetes orchestration for deployment

## Implementation Details

### Component Overview

1. **TranscriptionServer**: Handles WebSocket connections and HTTP requests, manages audio chunking
2. **WhisperASR**: Wraps the Whisper model for transcription, handles audio processing

### Chunking Strategy

Instead of using Voice Activity Detection (VAD), this implementation uses:
- Fixed-length audio chunks (configurable, default 5 seconds)
- Overlapping windows (default 0.5 seconds) to avoid cutting words
- Context passing between chunks for continuous transcription

### Scaling Configuration

- Each component can scale independently based on load
- WhisperASR replicas require GPU resources
- TranscriptionServer handles connection management

## Load Testing

The project includes a Locust-based load testing script to simulate multiple concurrent users:

```bash
cd scripts
locust -f locustfile.py --host=http://localhost:8000
```

## Monitoring

Ray Dashboard provides insights into the service:

```bash
kubectl port-forward svc/whisper-streaming-head-svc 8265:8265
# Access at http://localhost:8265
```

## License

MIT