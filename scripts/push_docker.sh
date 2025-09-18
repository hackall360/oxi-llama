#!/bin/sh

set -eu

export VERSION=${VERSION:-0.0.0}

docker build \
    --push \
    --platform=linux/arm64,linux/amd64 \
    --build-arg=VERSION \
    -f Dockerfile \
    -t ollama/ollama -t ollama/ollama:$VERSION \
    .
