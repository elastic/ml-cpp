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
# 3. Upload the all-platforms bundle to S3, where day-to-day Elasticsearch
#    builds will download it from.

rm -rf build/distributions

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Download from S3, combine, and upload to BuildKite's artifact store.
./gradlew --info -Dbuild.version_qualifier=$VERSION_QUALIFIER -Dbuild.snapshot=$BUILD_SNAPSHOT buildUberZipFromDownloads buildDependenciesZipFromDownloads buildNoDependenciesZipFromDownloads buildDependencyReport
buildkite-agent artifact upload "build/distributions/*"
