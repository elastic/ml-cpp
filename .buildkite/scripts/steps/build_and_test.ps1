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

# The Windows part of ML C++ CI does the following:
#
# Builds and unit tests the Windows version of the C++

$ErrorActionPreference="Stop"

# Change directory to the top level of the repo
Set-Location -Path "$PSScriptRoot\..\..\.."

# Ensure 3rd party dependencies are installed
& "dev-tools\download_windows_deps.ps1"

# Default to a snapshot build
if (!(Test-Path Env:BUILD_SNAPSHOT)) {
    $Env:BUILD_SNAPSHOT="true"
}

# Default to running tests
if (!(Test-Path Env:RUN_TESTS)) {
    $Env:RUN_TESTS="true"
}

# Default to no version qualifier
if (!(Test-Path Env:VERSION_QUALIFIER)) {
    $Env:VERSION_QUALIFIER=""
} elseif (Test-Path Env:PR_AUTHOR) {
    Write-Output "VERSION_QUALIFIER should not be set in PR builds: was $Env:VERSION_QUALIFIER"
    Exit 2
}

if ($Env:RUN_TESTS -eq "false") {
    $Tasks="clean", "buildZip", "buildZipSymbols"
} else {
    $Tasks="clean", "buildZip", "buildZipSymbols", "check"
}

if (Test-Path Env:ML_DEBUG) {
    $DebugOption="-Dbuild.ml_debug=$Env:ML_DEBUG"
} else {
    $DebugOption=""
}

# The exit code of the gradlew commands is checked explicitly, and their
# stderr is treated as an error by PowerShell without this
$ErrorActionPreference="Continue"

# Run the build and unit tests
# The | % { "$_" } at the end converts any error objects on stderr to strings
& ".\gradlew.bat" --info "-Dbuild.version_qualifier=$Env:VERSION_QUALIFIER" "-Dbuild.snapshot=$Env:BUILD_SNAPSHOT" $DebugOption $Tasks 2>&1 | % { "$_" }
if ($LastExitCode -ne 0) {
    Exit $LastExitCode
}

buildkite-agent artifact upload "build/distributions/*"
