#!/bin/bash
# Build and push the Docker image

# Configuration
REGISTRY="${REGISTRY:-your-registry}"  # Replace with your container registry or set env var
IMAGE_NAME="whisper-streaming"
IMAGE_TAG="latest"

# Full image name
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the image
echo "Building Docker image: ${FULL_IMAGE}"
docker build -t ${FULL_IMAGE} -f Dockerfile ..

# Push the image to the registry
read -p "Push image to registry ${REGISTRY}? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Pushing image to registry..."
    docker push ${FULL_IMAGE}
    echo "Done! Image is available at: ${FULL_IMAGE}"
else
    echo "Skipping push to registry. Image built locally: ${FULL_IMAGE}"
fi