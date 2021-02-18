#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Script to download 3rd party dependencies built on a reference macOS
# build server to a CI machine.

set -e

if [ `uname` != Darwin ] ; then
    echo "This script is intended for use on macOS"
    exit 1
fi

DEST=/usr

case `uname -m` in

    arm64)
        ARCHIVE=local-arm64-apple-macosx11.1-2.tar.bz2
        ;;

    *)
        echo "No archive is available for this architecture:" `uname -m 2>&1`
        exit 2
        ;;

esac

URL="https://s3-eu-west-1.amazonaws.com/prelert-artifacts/dependencies/$ARCHIVE"

echo "Downloading dependencies from $URL"
cd "$TMPDIR" && curl -s -S --retry 5 -O "$URL"
echo "Extracting dependencies from $ARCHIVE"
cd "$DEST" && tar -jmxf "$TMPDIR/$ARCHIVE"
echo "Cleaning up dependency archive"
rm -f "$TMPDIR/$ARCHIVE"

