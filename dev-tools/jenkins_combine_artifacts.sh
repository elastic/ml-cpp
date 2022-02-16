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

# The post-processing step of ML C++ CI does the following:
#
# 1. Download the platform-specific artifacts built by the the first phase
#    of the ML CI job.
# 2. Combine the platform-specific artifacts into an all-platforms bundle,
#    as used by the Elasticsearch build.
# 3. Upload the all-platforms bundle to S3, where day-to-day Elasticsearch
#    builds will download it from.
# 4. Upload all artifacts, both platform-specific and all-platforms, to
#    GCS, where release manager builds will download them from.
#

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

set +x

# Change directory to the directory containing this script
cd "$(dirname $0)"

# Obtain AWS credentials from Vault
. ./aws_creds_from_vault.sh

set -e

. docker/prefetch_docker_image.sh

cd ..
rm -rf build/distributions

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# The "branch" here selects which "$BRANCH.gradle" file of release manager is used
VERSION=$(cat gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
MAJOR=$(echo $VERSION | awk -F. '{ print $1 }')
MINOR=$(echo $VERSION | awk -F. '{ print $2 }')
if [ -n "$(git ls-remote --heads origin $MAJOR.$MINOR)" ] ; then
    BRANCH=$MAJOR.$MINOR
elif [ -n "$(git ls-remote --heads origin $MAJOR.x)" ] ; then
    BRANCH=$MAJOR.x
else
    # TODO: keep an eye on this in case it changes to main
    BRANCH=master
fi

# Jenkins sets BUILD_SNAPSHOT, but the Docker container requires a workflow that
# is staging or snapshot
if [ "$BUILD_SNAPSHOT" = false ] ; then
    WORKFLOW=staging
else
    WORKFLOW=snapshot
fi

# Download from S3, combine, and upload to S3 using the AWS credentials obtained
# above, and discarding the GCS credentials in the sub-shell before running
# anything that might log the environment
(unset GCS_VAULT_ROLE_ID GCS_VAULT_SECRET_ID && ./gradlew --info -Dbuild.version_qualifier=$VERSION_QUALIFIER -Dbuild.snapshot=$BUILD_SNAPSHOT uberUpload)

# Allow other users access to read the artifacts so they are readable in the
# container
chmod a+r build/distributions/*

# Allow other users write access to create checksum files
chmod a+w build/distributions

# Flip the Vault variables over to the GCS credentials
case $- in
    *x*)
        set +x
        REENABLE_X_OPTION=true
        ;;
    *)
        REENABLE_X_OPTION=false
        ;;
esac
export VAULT_ROLE_ID="$GCS_VAULT_ROLE_ID"
export VAULT_SECRET_ID="$GCS_VAULT_SECRET_ID"
unset GCS_VAULT_ROLE_ID GCS_VAULT_SECRET_ID
if [ "$REENABLE_X_OPTION" = true ] ; then
    set -x
fi

IMAGE=docker.elastic.co/infra/release-manager:latest
prefetch_docker_image "$IMAGE"

# Generate checksum files and upload to GCS
docker run --rm \
  --name release-manager \
  -e VAULT_ADDR \
  -e VAULT_ROLE_ID \
  -e VAULT_SECRET_ID \
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

