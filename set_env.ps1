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

# Set up a build environment, to ensure repeatable builds

# Initialize the Visual Studio command prompt environment variables
& 'C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64

# Set CPP_SRC_HOME to be an absolute path to this script's location, as
# different builds will come from different repositories and go to different
# staging areas
cd $PSScriptRoot
$env:CPP_SRC_HOME = git rev-parse --show-toplevel

# Assume the drive letter where our 3rd party dependencies are installed under
# \usr\local is the current drive at the time this script is run
$env:ROOT = (get-location).Drive.Name + ":"

$env:PATH="C:\Program Files\CMake\bin;$env:CPP_SRC_HOME\build\distribution\platform\windows-x86_64\bin;$env:PATH"

$env:INCLUDE=""
$env:LIBPATH=""

