#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The non-Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build, obtain credentials from Vault for the accessing
#    S3
# 2. If this is a PR build, check the code style
# 3. Build and unit test the Linux version of the C++
# 4. Build the macOS version of the C++
# 5. If this is not a PR build, upload the builds to the artifacts directory on
#    S3 that subsequent Java builds will download the C++ components from
#
# The steps run in Docker containers that ensure OS dependencies
# are appropriate given the support matrix.
#
# The macOS build cannot be unit tested as it is cross-compiled.

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

# If this isn't a PR build then obtain credentials from Vault
if [ -z "$PR_AUTHOR" ] ; then
    set +x
    export VAULT_TOKEN=$(vault write -field=token auth/approle/login role_id="$VAULT_ROLE_ID" secret_id="$VAULT_SECRET_ID")

    AWS_CREDS=$(vault read -format=json -field=data aws-dev/creds/prelertartifacts)
    export ML_AWS_ACCESS_KEY=$(echo $AWS_CREDS | jq -r '.access_key')
    export ML_AWS_SECRET_KEY=$(echo $AWS_CREDS | jq -r '.secret_key')

    unset VAULT_TOKEN VAULT_ROLE_ID VAULT_SECRET_ID
    set -x
fi

set -e

# Change directory to the directory containing this script
cd $(dirname $0)

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Jenkins sets BUILD_SNAPSHOT, but our Docker scripts require SNAPSHOT
if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
fi

# Remove any old builds
rm -rf ../builds

# If this is a PR build then fail fast on style checks
if [ -n "$PR_AUTHOR" ] ; then
    ./docker_check_style.sh
fi

# Build and test Linux
./docker_test.sh linux

# Build macOS
./docker_build.sh macosx

# If this isn't a PR build then upload the artifacts
if [ -z "$PR_AUTHOR" ] ; then
    cd ..
    ./gradlew --info -b upload.gradle -Dbuild.snapshot=$BUILD_SNAPSHOT upload
fi

