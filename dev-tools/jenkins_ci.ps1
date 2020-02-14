#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build, obtain credentials from Vault for the accessing
#    S3
# 2. Build and unit test the Windows version of the C++
# 3. If this is not a PR build, upload the build to the artifacts directory on
#    S3 that subsequent Java builds will download the C++ components from

$ErrorActionPreference="Stop"

# If this isn't a PR build then obtain credentials from Vault
if (!(Test-Path Env:PR_AUTHOR)) {
    # Generate a Vault token
    $Env:VAULT_TOKEN=& vault write -field=token auth/approle/login "role_id=$Env:VAULT_ROLE_ID" "secret_id=$Env:VAULT_SECRET_ID"
    if ($LastExitCode -ne 0) {
        Exit $LastExitCode
    }

    $AwsCreds=& vault read -format=json -field=data aws-dev/creds/prelertartifacts
    if ($LastExitCode -ne 0) {
        Exit $LastExitCode
    }
    $Env:ML_AWS_ACCESS_KEY=(echo $AwsCreds | jq -r ".access_key")
    $Env:ML_AWS_SECRET_KEY=(echo $AwsCreds | jq -r ".secret_key")

    # Remove VAULT_* environment variables
    Remove-Item Env:VAULT_TOKEN
    Remove-Item Env:VAULT_ROLE_ID
    Remove-Item Env:VAULT_SECRET_ID
}

# Change directory to the top level of the repo
Set-Location -Path "$PSScriptRoot\.."

# Ensure 3rd party dependencies are installed
& "dev-tools\download_windows_deps.ps1"

# Default to a snapshot build
if (!(Test-Path Env:BUILD_SNAPSHOT)) {
    $Env:BUILD_SNAPSHOT="true"
}

# The exit code of the gradlew commands is checked explicitly, and their
# stderr is treated as an error by PowerShell without this
$ErrorActionPreference="Continue"

# Run the build and unit tests
# The | % { "$_" } at the end converts any error objects on stderr to strings
& ".\gradlew.bat" --info "-Dbuild.snapshot=$Env:BUILD_SNAPSHOT" clean buildZip buildZipSymbols check 2>&1 | % { "$_" }
if ($LastExitCode -ne 0) {
    Exit $LastExitCode
}

# If this isn't a PR build then upload the artifacts
if (!(Test-Path Env:PR_AUTHOR)) {
    # The | % { "$_" } at the end converts any error objects on stderr to strings
    & ".\gradlew.bat" --info -b "upload.gradle" "-Dbuild.snapshot=$Env:BUILD_SNAPSHOT" upload 2>&1 | % { "$_" }
    if ($LastExitCode -ne 0) {
        Exit $LastExitCode
    }
}

