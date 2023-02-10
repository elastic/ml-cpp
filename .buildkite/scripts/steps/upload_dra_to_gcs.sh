#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Upload all artifacts, both platform-specific and all-platforms, to
# GCS, where release manager builds will download them from.
#

. ./dev-tools/docker/prefetch_docker_image.sh

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then 
    BUILD_SNAPSHOT=true 
fi

# The "branch" here selects which "$BRANCH.gradle" file of release manager is used
VERSION=$(awk -F= '/elasticsearchVersion/ {print $2}' gradle.properties)
MAJOR=$(echo $VERSION | awk -F. '{ print $1 }')
MINOR=$(echo $VERSION | awk -F. '{ print $2 }')
if [ -n "$(git ls-remote --heads origin $MAJOR.$MINOR)" ] ; then 
    BRANCH=$MAJOR.$MINOR 
elif [ -n "$(git ls-remote --heads origin $MAJOR.x)" ] ; then 
    BRANCH=$MAJOR.x 
else 
    BRANCH=main 
fi

if [ "$BUILD_SNAPSHOT" = false ] ; then 
    WORKFLOW=staging 
else 
    WORKFLOW=snapshot 
fi

# Allow other users access to read the artifacts so they are readable in the
# container
chmod a+r build/distributions/*

# Allow other users write access to create checksum files
chmod a+w build/distributions

# Variables named *_ACCESS_KEY & *_SECRET_KEY are redacted in BuildKite’s environment
# so we store the vault role and secret id values in them for security
VAULT_ACCESS_KEY=`vault read -field=role_id secret/ci/elastic-ml-cpp/gcs/creds/prelertartifacts`
VAULT_SECRET_KEY=`vault read -field=secret_id secret/ci/elastic-ml-cpp/gcs/creds/prelertartifacts`

IMAGE=docker.elastic.co/infra/release-manager:latest
prefetch_docker_image "$IMAGE"

# Generate checksum files and upload to GCS
docker run --rm \
  --name release-manager \
  -e VAULT_ADDR='https://secrets.elastic.co:8200' \
  -e VAULT_ROLE_ID=$VAULT_ACCESS_KEY \
  -e VAULT_SECRET_ID=$VAULT_SECRET_KEY \
  --mount type=bind,readonly=false,src="$PWD",target=/artifacts \
  "$IMAGE" \
    cli collect \
      --project ml-cpp \
      --branch "$BRANCH" \
      --version "$VERSION" \
      --commit `git rev-parse HEAD` \
      --workflow "$WORKFLOW" \
      --qualifier "$VERSION_QUALIFIER" \
      --artifact-set main
