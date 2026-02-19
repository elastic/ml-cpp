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

# Downloads and configures sccache for use as a compiler cache with a GCS backend.
# Dot-source this script so that environment variables are set in the caller:
#   . dev-tools\setup_sccache.ps1

$ErrorActionPreference = "Stop"

$SccacheVersion = "v0.14.0"
$SccacheInstallDir = if ($Env:SCCACHE_INSTALL_DIR) { $Env:SCCACHE_INSTALL_DIR } else { "$Env:LOCALAPPDATA\sccache" }

function Install-Sccache {
    $existing = Get-Command sccache -ErrorAction SilentlyContinue
    if ($existing) {
        $Script:SccachePath = $existing.Source
        Write-Host "sccache already installed: $Script:SccachePath"
        return
    }

    $zipName = "sccache-${SccacheVersion}-x86_64-pc-windows-msvc.zip"
    $url = "https://github.com/mozilla/sccache/releases/download/${SccacheVersion}/${zipName}"
    $tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "sccache-install"

    if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
    New-Item -ItemType Directory -Path $tmpDir | Out-Null

    Write-Host "Downloading sccache ${SccacheVersion} for Windows x86_64..."
    Invoke-WebRequest -Uri $url -OutFile "$tmpDir\$zipName" -UseBasicParsing

    Expand-Archive -Path "$tmpDir\$zipName" -DestinationPath $tmpDir -Force

    $binary = Get-ChildItem -Path $tmpDir -Recurse -Filter "sccache.exe" | Select-Object -First 1
    if (-not $binary) {
        throw "sccache.exe not found after extraction"
    }

    if (-not (Test-Path $SccacheInstallDir)) {
        New-Item -ItemType Directory -Path $SccacheInstallDir | Out-Null
    }
    Copy-Item -Path $binary.FullName -Destination "$SccacheInstallDir\sccache.exe" -Force
    Remove-Item -Recurse -Force $tmpDir

    if ($Env:PATH -notlike "*$SccacheInstallDir*") {
        $Env:PATH = "$SccacheInstallDir;$Env:PATH"
    }

    $Script:SccachePath = "$SccacheInstallDir\sccache.exe"
    Write-Host "sccache installed: $Script:SccachePath"
}

function Configure-GcsBackend {
    if (-not $Env:SCCACHE_GCS_BUCKET) {
        $Env:SCCACHE_GCS_BUCKET = "elastic-ml-cpp-sccache"
    }
    if (-not $Env:SCCACHE_GCS_KEY_PREFIX) {
        $Env:SCCACHE_GCS_KEY_PREFIX = "windows-x86_64"
    }
    $Env:SCCACHE_GCS_RW_MODE = "READ_WRITE"

    if ($Env:SCCACHE_GCS_KEY_PATH) {
        Write-Host "sccache GCS auth: service account key at $Env:SCCACHE_GCS_KEY_PATH"
    } elseif ($Env:GOOGLE_APPLICATION_CREDENTIALS) {
        $Env:SCCACHE_GCS_KEY_PATH = $Env:GOOGLE_APPLICATION_CREDENTIALS
        Write-Host "sccache GCS auth: GOOGLE_APPLICATION_CREDENTIALS"
    } else {
        Write-Host "sccache GCS auth: using instance metadata / workload identity"
    }

    Write-Host "sccache GCS config: bucket=$Env:SCCACHE_GCS_BUCKET prefix=$Env:SCCACHE_GCS_KEY_PREFIX"
}

function Update-CmakeFlags {
    $launcherFlags = "-DCMAKE_CXX_COMPILER_LAUNCHER=$Script:SccachePath -DCMAKE_C_COMPILER_LAUNCHER=$Script:SccachePath"
    if ($Env:CMAKE_FLAGS) {
        $Env:CMAKE_FLAGS = "$Env:CMAKE_FLAGS $launcherFlags"
    } else {
        $Env:CMAKE_FLAGS = $launcherFlags
    }
    Write-Host "sccache: CMAKE_FLAGS updated with compiler launcher"
}

function Start-SccacheServer {
    & $Script:SccachePath --stop-server 2>$null
    & $Script:SccachePath --start-server
    Write-Host "sccache server started"
}

# --- Main ---
Install-Sccache
Configure-GcsBackend
Update-CmakeFlags
Start-SccacheServer

$Env:SCCACHE_PATH = $Script:SccachePath
Write-Host "sccache setup complete"
