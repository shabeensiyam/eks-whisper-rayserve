FROM rayproject/ray-ml:latest

WORKDIR /app

# System dependencies - ensuring FFmpeg is installed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY client/ /app/client/

# Set working directory for Ray Serve
WORKDIR /app

# Command to run when container starts
CMD ["python", "-m", "src.server"]