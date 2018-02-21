#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#
$ErrorActionPreference="Stop"
$Archive="usr-x86_64-windows-2012_r2-5.zip"
$Destination="C:\"
if (!(Test-Path "$Destination\usr\local\lib\boost_system-vc141-mt-1_65_1.dll")) {
    Remove-Item "$Destination\usr" -Recurse -Force -ErrorAction Ignore
    $ZipSource="https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/$Archive"
    $ZipDestination="$env:TEMP\$Archive"
    (New-Object Net.WebClient).DownloadFile($ZipSource, $ZipDestination)
    Add-Type -assembly "system.io.compression.filesystem"
    [IO.Compression.ZipFile]::ExtractToDirectory($ZipDestination, $Destination)
    Remove-Item $ZipDestination
}
