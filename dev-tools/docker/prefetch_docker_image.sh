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

# Pull a docker image, retrying up to 5 times.
#
# Making sure the "FROM" image is present locally before building an image
# based on it removes the risk of a "docker build" failing due to transient
# Docker registry problems.
function prefetch_docker_image {
    IMAGE="$1"
    ATTEMPT=0
    MAX_RETRIES=5
  
    while true
    do
        ATTEMPT=$((ATTEMPT+1))
  
        if [ $ATTEMPT -gt $MAX_RETRIES ] ; then
            echo "Docker pull retries exceeded, aborting."
            exit 1
        fi
  
        if docker pull "$IMAGE" ; then
            echo "Docker pull of $IMAGE successful."
            break
        else
            echo "Docker pull of $IMAGE unsuccessful, attempt '$ATTEMPT'."
        fi
    done
}

# Similar to above, but taking a Dockerfile as an argument instead of the
# image name and prefetching the base image.
#
# The argument is the path to the Dockerfile to be built.
function prefetch_docker_base_image {
    DOCKERFILE="$2"
    IMAGE=$(grep '^FROM' "$DOCKERFILE" | awk '{ print $2 }' | head -1)
    prefetch_docker_image "$IMAGE"
}
