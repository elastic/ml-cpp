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
#
# The test bundle contains pre-built test executables and ALL DLLs.
# We prepend the DLL directories to PATH so the executables can find them
# regardless of which agent workspace we're on.

$ErrorActionPreference="Stop"

Set-Location -Path "$PSScriptRoot\..\..\.."

$BuildDir = "cmake-build-relwithdebinfo"
$BuildType = "RelWithDebInfo"
$TestBundle = "windows-x86_64-test-bundle.zip"

Write-Output "--- Downloading test bundle"
buildkite-agent artifact download "windows-x86_64-test-bundle.tar.gz" .

Write-Output "--- Extracting test bundle"
& tar xzf "windows-x86_64-test-bundle.tar.gz"
Remove-Item "windows-x86_64-test-bundle.tar.gz"
Write-Output "Test bundle extracted"

# Prepend DLL directories to PATH so test executables can find all libraries.
# This overrides the build-time paths that may point to a different agent.
$DllDirs = @()
if (Test-Path "build\distribution") {
    $DllDirs += (Get-ChildItem -Path "build\distribution" -Recurse -Filter "*.dll" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty DirectoryName -Unique)
}
if (Test-Path "$BuildDir\lib") {
    $DllDirs += (Get-ChildItem -Path "$BuildDir\lib" -Recurse -Filter "*.dll" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty DirectoryName -Unique)
}
if ($DllDirs.Count -gt 0) {
    $DllPath = ($DllDirs | Select-Object -Unique) -join ";"
    $Env:PATH = "$DllPath;$Env:PATH"
    Write-Output "Added $($DllDirs.Count) DLL directories to PATH"
}

# Also set CPP_SRC_HOME for resource file discovery
$Env:CPP_SRC_HOME = (Get-Location).Path

Write-Output "--- Running tests"
$ErrorActionPreference="Continue"

$SourceDir = (Get-Location).Path
cmake "-DSOURCE_DIR=$SourceDir" "-DBUILD_DIR=$SourceDir\$BuildDir" "-DBUILD_TYPE=$BuildType" -P cmake/run-all-tests-parallel.cmake
$TestExitCode=$LastExitCode

$ErrorActionPreference="Stop"

# Upload test results
Write-Output "--- Uploading test results"
$OutFiles = Get-ChildItem -Path . -Include "*.out","*.junit" -File -Recurse -ErrorAction SilentlyContinue
if ($OutFiles) {
    Compress-Archive -Path ($OutFiles | Select-Object -ExpandProperty FullName) -DestinationPath "windows-x86_64-unit_test_results.zip" -ErrorAction SilentlyContinue
    if (Test-Path "windows-x86_64-unit_test_results.zip") {
        buildkite-agent artifact upload "windows-x86_64-unit_test_results.zip"
    }
}

if ($TestExitCode -ne 0) {
    Exit $TestExitCode
}
