#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

$ErrorActionPreference="Stop"

# Change directory to the top level of the repo
Set-Location -Path "$PSScriptRoot\.."

# Ensure 3rd party dependencies are installed
& "dev-tools\download_windows_deps.ps1"
