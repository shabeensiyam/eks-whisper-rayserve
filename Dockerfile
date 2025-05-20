FROM rayproject/ray-ml:2.9.2-py310

WORKDIR /app

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