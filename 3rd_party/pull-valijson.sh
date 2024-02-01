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
# If updating this script ensure the license information is correct in the
# licenses sub-directory.

cd `dirname "$BASH_SOURCE"`

if [ ! -d "valijson" ] ; then
    git -c advice.detachedHead=false clone --depth=1 --branch=v1.0.2 git@github.com:tristanpenman/valijson.git
fi

