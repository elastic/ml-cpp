#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build nor a debug build, obtain credentials from Vault
#    for the accessing S3
# 2. Build and unit test the Windows version of the C++
# 3. If this is not a PR build nor a debug build, upload the builds to the
#    artifacts directory on S3 that subsequent Java builds will download the C++
#    components from

# If this isn't a PR build or a debug build then obtain credentials from Vault
if (!(Test-Path Env:PR_AUTHOR) -And !(Test-Path Env:ML_DEBUG)) {
    # Generate a Vault token
    $Env:VAULT_TOKEN=& vault write -field=token auth/approle/login "role_id=$Env:VAULT_ROLE_ID" "secret_id=$Env:VAULT_SECRET_ID"
    if ($LastExitCode -ne 0) {
        Exit $LastExitCode
    }

    $Failures=0
    do {
        $AwsCreds=& vault read -format=json -field=data aws-dev/creds/prelertartifacts
        if ($LastExitCode -eq 0) {
            $Env:ML_AWS_ACCESS_KEY=(echo $AwsCreds | jq -r ".access_key")
            $Env:ML_AWS_SECRET_KEY=(echo $AwsCreds | jq -r ".secret_key")
        } else {
            $Failures++
            Write-Output "Attempt $Failures to get AWS credentials failed"
        }
    } while (($Failures -lt 3) -and [string]::IsNullOrEmpty($Env:ML_AWS_ACCESS_KEY))

    # Remove VAULT_* environment variables
    Remove-Item Env:VAULT_TOKEN
    Remove-Item Env:VAULT_ROLE_ID
    Remove-Item Env:VAULT_SECRET_ID

    if ([string]::IsNullOrEmpty($Env:ML_AWS_ACCESS_KEY) -or [string]::IsNullOrEmpty($Env:ML_AWS_SECRET_KEY)) {
        Write-Output "Exiting after failing to get AWS credentials $Failures times"
        Exit 1
    }
}

$ErrorActionPreference="Stop"

# Change directory to the top level of the repo
Set-Location -Path "$PSScriptRoot\.."

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

if (Test-Path Env:PR_AUTHOR) {
    if ($Env:RUN_TESTS -eq "false") {
        Write-Output "RUN_TESTS should not be false in PR builds"
        Exit 3
    }
    $Tasks="clean", "buildZip", "check"
} elseif ($Env:RUN_TESTS -eq "false") {
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

# If this isn't a PR build and isn't a debug build then upload the artifacts
if (!(Test-Path Env:PR_AUTHOR) -And !(Test-Path Env:ML_DEBUG)) {
    # The | % { "$_" } at the end converts any error objects on stderr to strings
    & ".\gradlew.bat" --info -b "upload.gradle" "-Dbuild.version_qualifier=$Env:VERSION_QUALIFIER" "-Dbuild.snapshot=$Env:BUILD_SNAPSHOT" upload 2>&1 | % { "$_" }
    if ($LastExitCode -ne 0) {
        Exit $LastExitCode
    }
}
