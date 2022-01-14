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

# The non-Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build nor a debug build, obtain credentials from Vault
#    for the accessing S3
# 2. Build and unit test the C++ on the native architecture
# 3. For Linux PR builds, also run some Java integration tests using the newly
#    built C++ code
# 4. If this is not a PR build nor a debug build, upload the builds to the
#    artifacts directory on S3 that subsequent Java builds will download the C++
#    components from
#
# On Linux all steps run in Docker containers that ensure OS dependencies
# are appropriate given the support matrix.
#
# On macOS the build runs on the native machine, but downloads dependencies
# that were previously built on a reference build server.  However, care still
# needs to be taken that the machines running this script are set up
# appropriately for generating builds for redistribution.

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
      --commit `git rev-parse HEAD` \
      --workflow "$WORKFLOW" \
      --qualifier "$VERSION_QUALIFIER" \
      --artifact-set main

