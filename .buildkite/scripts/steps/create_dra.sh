#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# This post-processing step of ML C++ CI does the following:
#
# 1. Download the platform-specific artifacts built by the first phase
#    of the ML CI job.
# 2. Combine the platform-specific artifacts into an all-platforms bundle,
#    as used by the Elasticsearch build.

rm -rf build/distributions

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

VERSION=$(cat ${REPO_ROOT}/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
if [ "$BUILD_SNAPSHOT" = "true" ] ; then
    VERSION=${VERSION}-SNAPSHOT
fi
export VERSION

# Download artifacts from a previous build (TODO remove build specifier once integrated with branch pipeline),
# extract each, combine to 'uber' zip file, and upload to BuildKite's artifact store.
buildkite-agent artifact download "build/distributions/*" --build 01865abd-bfbd-4f5e-b87c-6f2b07fef27e
rm -rf build/temp
mkdir -p build/temp
for it in darwin-aarch64 darwin-x86_64 linux-aarch64 linux-x86_64 windows-x86_64; do unzip -o build/distributions/ml-cpp-${VERSION}-${it}.zip -d  build/temp;  done
(cd build/temp && zip ../distributions/ml-cpp-${VERSION}.zip -r platform)
buildkite-agent artifact upload "build/distributions/*"
