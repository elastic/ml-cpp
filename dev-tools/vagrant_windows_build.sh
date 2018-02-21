#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Builds the machine learning C++ code for Windows in a Vagrant VM.
#
# The output .zip files are then copied out of the VM to the location in
# the current repository that they'd be in had they been built outside of
# the VM.

# Default to a snapshot build
if [ "$SNAPSHOT" = no ] ; then
    SNAPSHOT_PROP=-Dbuild.snapshot=false
else
    SNAPSHOT_PROP=-Dbuild.snapshot=true
fi

set -e

# The build needs to be done with the directory containing the Vagrantfile being
# the current directory.
MY_DIR=`dirname "$BASH_SOURCE"`
TOOLS_DIR=`cd "$MY_DIR" && pwd`

cd "$TOOLS_DIR/.."
rm -f build/distributions/*windows-x86_64.zip

VAGRANT_DIR="$TOOLS_DIR/vagrant/windows"
cd "$VAGRANT_DIR"

vagrant up

# The code is made available in the VM via a VirtualBox synced folder.  However,
# the Visual C++ compiler doesn't work on a case sensitive file system and
# the VirtualBox synced folder is case sensitive, so we copy it to a normal
# folder for building.  If the normal folder exists on the VM it's removed
# first.  This means that the VM can be reused without having to be provisioned
# on every build.  (Provisioning the VM for the first time is very slow.)
vagrant ssh <<EOF
rm -rf /cygdrive/c/machine-learning-cpp
exit
EOF

vagrant ssh -c cmd.exe <<EOF
xcopy /h /i /e /q C:\machine-learning-cpp-synced C:\machine-learning-cpp
cd C:\machine-learning-cpp
C:\tools\Gradle\bin\gradle $SNAPSHOT_PROP clean assemble
xcopy /i /e C:\machine-learning-cpp\build\distributions\*windows-x86_64.zip C:\machine-learning-cpp-synced\build\distributions
exit
EOF

vagrant halt

# This will cause this script to fail if the VM part of the build failed
cd "$TOOLS_DIR/../build/distributions"
ls *windows-x86_64.zip

