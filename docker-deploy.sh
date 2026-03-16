#!/usr/bin/env bash
set -euo pipefail

IMAGE="docker.epilb.com/public/mirofish-en:2"

# Check Docker daemon
if ! docker info &>/dev/null; then
  echo "ERROR: Docker daemon is not running"
  exit 1
fi

# Check Docker Hub login
if ! docker buildx ls &>/dev/null; then
  echo "ERROR: Docker buildx not available"
  exit 1
fi

echo "Building: $IMAGE"
if ! docker build \
  --platform linux/amd64 \
  --tag "$IMAGE" \
  .; then
  echo ""
  echo "ERROR: Build failed"
  exit 1
fi
echo "✓ Built: $IMAGE"

echo ""
echo "Pushing: $IMAGE"
if ! docker push "$IMAGE"; then
  echo ""
  echo "ERROR: Push failed"
  exit 1
fi
echo "✓ Pushed: $IMAGE"
