#!/bin/bash
# Deploy the application to EKS

# Configuration
NAMESPACE="${NAMESPACE:-default}"
HF_TOKEN_SECRET_NAME="hf-token"
REGISTRY="${REGISTRY:-your-registry}"  # Replace with your container registry or set env var
IMAGE_NAME="whisper-streaming"
IMAGE_TAG="latest"

# Full image name
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH."
    exit 1
fi

# Check if connected to a Kubernetes cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Not connected to a Kubernetes cluster."
    echo "Please configure kubectl to connect to your EKS cluster."
    exit 1
fi

# Create namespace if it doesn't exist
kubectl get namespace ${NAMESPACE} > /dev/null 2>&1 || kubectl create namespace ${NAMESPACE}

# Check if HF token secret exists, create if needed
if ! kubectl get secret ${HF_TOKEN_SECRET_NAME} -n ${NAMESPACE} > /dev/null 2>&1; then
    echo "Creating Hugging Face token secret..."
    if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set."
        echo "Please set it with your base64-encoded Hugging Face token:"
        echo "export HUGGING_FACE_HUB_TOKEN=\$(echo -n 'your-token' | base64)"
        exit 1
    fi

    # Create secret from environment variable
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: ${HF_TOKEN_SECRET_NAME}
  namespace: ${NAMESPACE}
type: Opaque
data:
  token: ${HUGGING_FACE_HUB_TOKEN}
EOF
    echo "Secret created successfully."
fi

# Create a copy of the RayService YAML
TMP_YAML=$(mktemp)
cp ../deploy/whisper-rayservice.yaml ${TMP_YAML}

# Update image in RayService YAML file
echo "Updating image in RayService YAML..."
sed -i "s|your-registry/whisper-streaming:latest|${FULL_IMAGE}|g" ${TMP_YAML}

# Apply the RayService
echo "Deploying RayService..."
kubectl apply -f ${TMP_YAML}

# Apply the Kubernetes services
echo "Deploying Kubernetes services..."
kubectl apply -f ../deploy/kubernetes-services.yaml

echo "Waiting for RayService to be ready..."
kubectl wait --for=condition=Ready --timeout=600s rayservice/whisper-streaming -n ${NAMESPACE} 2>/dev/null || true

echo "Deployment complete! You can now access the service through:"
echo "- Ray Dashboard: kubectl port-forward svc/whisper-streaming-head-svc 8265:8265 -n ${NAMESPACE}"
echo "- Whisper ASR Service: kubectl port-forward svc/whisper-streaming-serve-svc 8000:8000 -n ${NAMESPACE}"
echo ""
echo "Then open the client in your browser: ../client/streaming_client.html"

# Clean up temp file
rm ${TMP_YAML}