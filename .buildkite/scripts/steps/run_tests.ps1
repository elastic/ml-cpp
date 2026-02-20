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

# Windows test step: downloads the test bundle from the build step, extracts
# it, and runs all test suites in parallel via CTest.

$ErrorActionPreference="Stop"

Set-Location -Path "$PSScriptRoot\..\..\.."

$BuildDir = "cmake-build-relwithdebinfo"
$BuildType = "RelWithDebInfo"
$TestBundle = "windows-x86_64-test-bundle.zip"

Write-Output "--- Downloading test bundle"
buildkite-agent artifact download $TestBundle .

Write-Output "--- Extracting test bundle"
& tar xzf $TestBundle
Remove-Item $TestBundle
Write-Output "Test bundle extracted"

Write-Output "--- Running tests"
$ErrorActionPreference="Continue"

$SourceDir = (Get-Location).Path
cmake "-DSOURCE_DIR=$SourceDir" "-DBUILD_DIR=$SourceDir\$BuildDir" "-DBUILD_TYPE=$BuildType" -P cmake/run-all-tests-parallel.cmake
$TestExitCode=$LastExitCode

$ErrorActionPreference="Stop"

# Upload test results
Write-Output "--- Uploading test results"
$TestResults = "windows-x86_64-unit_test_results.zip"
$OutFiles = Get-ChildItem -Path . -Include "*.out","*.junit" -File -Recurse -ErrorAction SilentlyContinue
if ($OutFiles) {
    Compress-Archive -Path ($OutFiles | Select-Object -ExpandProperty FullName) -DestinationPath $TestResults -ErrorAction SilentlyContinue
    if (Test-Path $TestResults) {
        buildkite-agent artifact upload $TestResults
    }
}

if ($TestExitCode -ne 0) {
    Exit $TestExitCode
}
