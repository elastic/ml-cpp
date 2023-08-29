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
$Archive="usr-x86_64-windows-2016-10.zip"
$Destination="C:\"
if (!(Test-Path "$Destination\usr\local\include\boost-1_83\boost\unordered\detail\prime_fmod.hpp")) {
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
