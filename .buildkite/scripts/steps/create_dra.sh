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
buildkite-agent artifact download "*.zip" build/distributions --build 01866fcf-3d1a-471d-b3b9-8a9a3d0c1ef6
ls -lR

rm -rf build/temp
mkdir -p build/temp
for it in darwin-aarch64 darwin-x86_64 linux-aarch64 linux-x86_64 windows-x86_64; do
  echo "Unzipping ml-cpp-${VERSION}-${it}.zip"
  unzip -o build/distributions/ml-cpp-${VERSION}-${it}.zip -d build/temp;
done
cd build/temp
echo "Zipping ml-cpp-${VERSION}.zip"
zip ../distributions/ml-cpp-${VERSION}.zip -r platform
ls -lR ..

# Create a zip excluding dependencies from combined platform-specific C++ distributions
echo "Creating nodeps archive"
find . -path "**/libMl*" -o \
       -path "**/platform/darwin*/controller.app/Contents/MacOS/*" -o \
       -path "**/platform/linux*/bin/*" -o \
       -path "**/platform/windows*/bin/*.exe" -o \
       -path "**/ml-en.dict" -o \
       -path "**/Info.plist" -o \
       -path "**/date_time_zonespec.csv" -o \
       -path "**/licenses/**" | xargs zip ../distributions/ml-cpp-${VERSION}-nodeps.zip
echo "rc = $?"

# Create a zip of dependencies only from combined platform-specific C++ distributions
echo "Creating nodeps archive"
find . \( -path "**/libMl*" -o \
          -path "**/platform/darwin*/controller.app/Contents/MacOS/*" -o \
          -path "**/platform/linux*/bin/*" -o \
          -path "**/platform/windows*/bin/*.exe" -o \
          -path "**/ml-en.dict" -o \
          -path "**/Info.plist" -o \
          -path "**/date_time_zonespec.csv" -o \
          -path "**/licenses/**" \) -prune -o -print | xargs zip ../distributions/ml-cpp-${VERSION}-deps.zip
echo "rc = $?"

cd -
pwd
ls -lR

# Create a CSV report on 3rd party dependencies we redistribute
echo "Creating dependency report"
./3rd_party/dependency_report.sh --csv build/distributions/dependencies-${VERSION}.csv 

buildkite-agent artifact upload "build/distributions/ml-cpp-${VERSION}.zip build/distributions/ml-cpp-${VERSION}-nodeps.zip build/distributions/ml-cpp-${VERSION}-deps.zip dependencies-${VERSION}.csv"
