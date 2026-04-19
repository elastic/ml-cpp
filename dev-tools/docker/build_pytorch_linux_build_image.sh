#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# Builds a Docker image that can be used to compile the machine learning
# C++ code for Linux with the latest build of the PyTorch release branch.
#
# This script is intended to be run regularly in order to flush out any
# issues we may have with the latest PyTorch release as early as possible.
# To avoid swamping the Docker registry with numerous images we re-use the
# same tag for the image.
#
# Optimizations:
#   1. Skip if viable/strict hasn't moved since the last build
#   2. sccache using a Docker secret for the GCS backend for fast incremental rebuilds

if [ "$(uname -m)" != x86_64 ] ; then
    echo "Native build images must be built on the correct hardware architecture"
    echo "Required: x86_64, Current: $(uname -m)"
    exit 1
fi

HOST=docker.elastic.co
ACCOUNT=ml-dev
REPOSITORY=ml-linux-dependency-build
VERSION=pytorch_latest
FULL_IMAGE="$HOST/$ACCOUNT/$REPOSITORY:$VERSION"
PYTORCH_BRANCH="${1:-viable/strict}"

set -e

cd "$(dirname "$0")"

# ---- Skip-if-unchanged check ----
# Compare the current viable/strict HEAD with the commit baked into the
# last published image.  If they match, there's nothing to rebuild.
echo "--- Checking if PyTorch ${PYTORCH_BRANCH} has changed since last build"
CURRENT_SHA=$(git ls-remote https://github.com/pytorch/pytorch.git "refs/heads/${PYTORCH_BRANCH}" 2>/dev/null | cut -f1)

if [ -n "$CURRENT_SHA" ]; then
    echo "Current ${PYTORCH_BRANCH} SHA: ${CURRENT_SHA}"

    # Try to read the label from the existing image (pull first if not local)
    docker pull "$FULL_IMAGE" 2>/dev/null || true
    PREVIOUS_SHA=$(docker inspect --format '{{index .Config.Labels "pytorch.commit"}}' "$FULL_IMAGE" 2>/dev/null || echo "")

    if [ -n "$PREVIOUS_SHA" ]; then
        echo "Previous build SHA:           ${PREVIOUS_SHA}"
        if [ "$CURRENT_SHA" = "$PREVIOUS_SHA" ]; then
            echo "PyTorch ${PYTORCH_BRANCH} unchanged — skipping rebuild"
            exit 0
        fi
        echo "SHA changed — rebuild needed"
    else
        echo "No previous build SHA found — full build required"
    fi
else
    echo "WARNING: could not fetch ${PYTORCH_BRANCH} SHA, proceeding with build"
fi

# ---- Build ----
DOCKER_DIR=$(docker info 2>/dev/null | grep '^ *Docker Root Dir' | awk -F: '{ print $2 }' | sed 's/^ *//')
echo "Building this image may require up to 50GB of space for Docker"
echo "Current space available in $DOCKER_DIR"
df -h "$DOCKER_DIR"

. ./prefetch_docker_image.sh
CONTEXT=pytorch_linux_image
prefetch_docker_base_image $CONTEXT/Dockerfile

echo "--- Building $FULL_IMAGE (branch: ${PYTORCH_BRANCH})"

export DOCKER_BUILDKIT=1

BUILD_ARGS="--build-arg pytorch_branch=${PYTORCH_BRANCH}"
if [ -n "$CURRENT_SHA" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg pytorch_commit=${CURRENT_SHA}"
fi

# Pass GCS credentials as a Docker secret for sccache.
# The key is injected by the Buildkite post-checkout hook into
# SCCACHE_GCS_KEY_FILE (a temp file with the service account JSON).
SECRET_ARGS=""
if [ -n "${SCCACHE_GCS_KEY_FILE:-}" ] && [ -f "$SCCACHE_GCS_KEY_FILE" ]; then
    SECRET_ARGS="--secret id=gcs_key,src=${SCCACHE_GCS_KEY_FILE}"
    echo "sccache: GCS credentials available — incremental build enabled"
else
    echo "sccache: no GCS credentials — full build (no cache)"
fi

docker build \
    --progress=plain \
    $BUILD_ARGS \
    $SECRET_ARGS \
    -t "$FULL_IMAGE" \
    "$CONTEXT"

# ---- Push ----
echo "--- Pushing $FULL_IMAGE"
echo "$DOCKER_REGISTRY_PASSWORD" | docker login -u "$DOCKER_REGISTRY_USERNAME" --password-stdin docker.elastic.co
docker push "$FULL_IMAGE"

echo "Build complete. PyTorch commit: ${CURRENT_SHA:-unknown}"
