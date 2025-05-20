#!/bin/bash
# Run the application locally for testing

# Start Ray locally if not already running
if ! ray status > /dev/null 2>&1; then
    echo "Starting Ray..."
    ray start --head --port=6379
fi

# Install dependencies if not in Docker
if [ ! -f "/.dockerenv" ]; then
    echo "Installing dependencies..."
    pip install -r ../requirements.txt
fi

# Change to the app directory
cd ..

# Run the application
echo "Starting Whisper streaming application..."
python -m src.server

echo "Application running! Access at http://localhost:8000"
echo "To stop, press Ctrl+C and run 'ray stop'"