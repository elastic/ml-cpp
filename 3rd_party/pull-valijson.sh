#!/bin/bash
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

# Script to get the appropriate version of Valijson, if not already present.
#
# Valijson must only be used in test code, _not_ any form of redistributable code.


cd `dirname "$BASH_SOURCE"`

if [ ! -d "valijson" ] ; then
    git -c advice.detachedHead=false clone --depth=1 --branch=v1.0.2 https://github.com/tristanpenman/valijson.git || { rm -rf valijson; exit 1; }
    # A tag can be moved, or a clone intercepted: check what was actually cloned.
    # Keep this in step with the COMMIT in pull-valijson.cmake.
    ACTUAL_COMMIT=`cd valijson && git rev-parse HEAD`
    if [ "$ACTUAL_COMMIT" != "0b4771e273a065d437814baf426bcfcafec0f434" ] ; then
        echo "Valijson v1.0.2 is at commit '$ACTUAL_COMMIT', expected 0b4771e273a065d437814baf426bcfcafec0f434" >&2
        rm -rf valijson
        exit 1
    fi
fi

