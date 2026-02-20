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

# Windows build step: compiles libraries, installs, strips, packages, builds
# test executables, and uploads a test bundle artifact.

$ErrorActionPreference="Stop"

Set-Location -Path "$PSScriptRoot\..\..\.."

if ([System.Environment]::OSVersion.Version.Build -lt 20348) {
    & "dev-tools\download_windows_deps.ps1"
}

if (!(Test-Path Env:BUILD_SNAPSHOT)) {
    $Env:BUILD_SNAPSHOT="true"
}

if (!(Test-Path Env:VERSION_QUALIFIER)) {
    $Env:VERSION_QUALIFIER=""
} elseif (Test-Path Env:PR_AUTHOR) {
    Write-Output "VERSION_QUALIFIER should not be set in PR builds: was $Env:VERSION_QUALIFIER"
    Exit 2
}

if (Test-Path Env:ML_DEBUG) {
    $DebugOption="-Dbuild.ml_debug=$Env:ML_DEBUG"
} else {
    $DebugOption=""
}

# Build libraries, install, strip, package (no tests)
$ErrorActionPreference="Continue"
& ".\gradlew.bat" --info "-Dbuild.version_qualifier=$Env:VERSION_QUALIFIER" "-Dbuild.snapshot=$Env:BUILD_SNAPSHOT" $DebugOption clean compile strip buildZip buildZipSymbols 2>&1 | % { "$_" }
$BuildExitCode=$LastExitCode
$ErrorActionPreference="Stop"

if ($BuildExitCode -ne 0) {
    Write-Output "--- Build failed with exit code $BuildExitCode"
    Exit $BuildExitCode
}

# Build test executables
Write-Output "--- Building test executables"
$BuildDir = "cmake-build-relwithdebinfo"
$BuildType = "RelWithDebInfo"

$ErrorActionPreference="Continue"
& ".\gradlew.bat" --info "-Dbuild.version_qualifier=$Env:VERSION_QUALIFIER" "-Dbuild.snapshot=$Env:BUILD_SNAPSHOT" $DebugOption configure 2>&1 | % { "$_" }
$ErrorActionPreference="Stop"

# set_env.bat needs to run in cmd context for the cmake call
& cmd.exe /c "set_env.bat && cmake --build $BuildDir --config $BuildType -j $Env:NUMBER_OF_PROCESSORS -t build_tests"

# Create test bundle — a zip containing test executables and shared libs,
# preserving relative directory structure so it extracts in-place.
Write-Output "--- Creating test bundle"
$TestBundle = "windows-x86_64-test-bundle.zip"

# Collect files with their relative paths
$FilesToBundle = @()

# Test executables
Get-ChildItem -Path "$BuildDir\test" -Recurse -Filter "ml_test_*.exe" | ForEach-Object {
    $FilesToBundle += $_.FullName
}

# Our shared libraries in the build directory
Get-ChildItem -Path "$BuildDir\lib" -Recurse -Filter "Ml*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    $FilesToBundle += $_.FullName
}
Get-ChildItem -Path "$BuildDir\lib" -Recurse -Filter "Ml*.lib" -ErrorAction SilentlyContinue | ForEach-Object {
    $FilesToBundle += $_.FullName
}

# Installed distribution (DLLs needed at runtime)
if (Test-Path "build\distribution") {
    Get-ChildItem -Path "build\distribution" -Recurse -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        $FilesToBundle += $_.FullName
    }
}

Write-Output "Bundling $($FilesToBundle.Count) files into $TestBundle"

# Use tar to preserve directory structure relative to repo root
$RepoRoot = (Get-Location).Path
$RelativePaths = $FilesToBundle | ForEach-Object { $_.Replace("$RepoRoot\", "").Replace("\", "/") }
$FileList = Join-Path $Env:TEMP "test-bundle-files.txt"
$RelativePaths | Out-File -FilePath $FileList -Encoding utf8

& tar czf "$TestBundle" -T "$FileList"
$BundleSize = [math]::Round((Get-Item $TestBundle).Length / 1MB)
Write-Output "Test bundle: $TestBundle (${BundleSize}MB)"

buildkite-agent artifact upload $TestBundle
buildkite-agent artifact upload "build\distributions\*"
