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
# 3. Combine the platform-specific 3rd party dependencies into a 'deps' bundle.
# 4. Combine the platform-specific non 3rd party dependencies into a 'deps' bundle
# 4. Create a dependency report containing licensing info on the 3rd party dependencies.

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

# Download artifacts, either from earlier steps in this build or from the build that triggered this one.
# TODO For now test with a manually set build id...
BUILDKITE_TRIGGERED_FROM_BUILD_ID=01866fcf-3d1a-471d-b3b9-8a9a3d0c1ef6
if [[ -n "${BUILDKITE_TRIGGERED_FROM_BUILD_ID}" ]]; then
  buildkite-agent artifact download "build/distributions/*.zip" . --build ${BUILDKITE_TRIGGERED_FROM_BUILD_ID}
  buildkite-agent artifact download "build\\distributions\\*.zip" . --build ${BUILDKITE_TRIGGERED_FROM_BUILD_ID}
else
  buildkite-agent artifact download "build/distributions/*.zip" .
  buildkite-agent artifact download "build\\distributions\\*.zip" .
fi

# Extract each platform specific zip file & combine to an 'uber' zip file.
rm -rf build/temp
mkdir -p build/temp
for it in darwin-aarch64 darwin-x86_64 linux-aarch64 linux-x86_64 windows-x86_64; do
  unzip -o build/distributions/ml-cpp-${VERSION}-${it}.zip -d build/temp;
done
cd build/temp
zip ../distributions/ml-cpp-${VERSION}.zip -r platform

# Create a zip excluding dependencies from combined platform-specific C++ distributions
find . -path "**/libMl*" -o \
       -path "**/platform/darwin*/controller.app/Contents/MacOS/*" -o \
       -path "**/platform/linux*/bin/*" -o \
       -path "**/platform/windows*/bin/*.exe" -o \
       -path "**/ml-en.dict" -o \
       -path "**/Info.plist" -o \
       -path "**/date_time_zonespec.csv" -o \
       -path "**/licenses/**" | xargs zip ../distributions/ml-cpp-${VERSION}-nodeps.zip

# Create a zip of dependencies only from combined platform-specific C++ distributions
find . \( -path "**/libMl*" -o \
          -path "**/platform/darwin*/controller.app/Contents/MacOS/*" -o \
          -path "**/platform/linux*/bin/*" -o \
          -path "**/platform/windows*/bin/*.exe" -o \
          -path "**/ml-en.dict" -o \
          -path "**/Info.plist" -o \
          -path "**/date_time_zonespec.csv" -o \
          -path "**/licenses/**" \) -prune -o -print | xargs zip ../distributions/ml-cpp-${VERSION}-deps.zip

cd -

# Create a CSV report on 3rd party dependencies we redistribute
./3rd_party/dependency_report.sh --csv build/distributions/dependencies-${VERSION}.csv 

# Upload the newly created artifacts
buildkite-agent artifact upload "build/distributions/ml-cpp-${VERSION}.zip;build/distributions/ml-cpp-${VERSION}-deps.zip;build/distributions/ml-cpp-${VERSION}-nodeps.zip;build/distributions/dependencies-${VERSION}.csv"
