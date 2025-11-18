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
$ErrorActionPreference="Stop"
# TODO: Fix the windows build and use the latest archive
# $Archive="usr-x86_64-windows-2016-16.zip"
$Archive="usr-x86_64-windows-2016-15.zip"
$Destination="C:\"
# If PyTorch is not version 2.7.1 then we need the latest download
if (!(Test-Path "$Destination\usr\local\include\pytorch\torch\csrc\api\include\torch\version.h") -Or
    !(Select-String -Path "$Destination\usr\local\include\pytorch\torch\csrc\api\include\torch\version.h" -Pattern "2.7.1" -Quiet)) {
    Remove-Item "$Destination\usr" -Recurse -Force -ErrorAction Ignore
    $ZipSource="https://storage.googleapis.com/elastic-ml-public/dependencies/$Archive"
    $ZipDestination="$Env:TEMP\$Archive"
    Write-Output "Downloading dependencies at time"
    Get-Date -Format yyyy-MM-ddTHH:mm:ss.ffffK
    (New-Object Net.WebClient).DownloadFile($ZipSource, $ZipDestination)
    Add-Type -assembly "system.io.compression.filesystem"
    [IO.Compression.ZipFile]::ExtractToDirectory($ZipDestination, $Destination)
    Remove-Item $ZipDestination
}
