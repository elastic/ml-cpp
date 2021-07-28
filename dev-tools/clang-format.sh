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

# Reformats Ml native source code, using clang-format,  to ensure consistency.

# Ensure $CPP_SRC_HOME is set
CPP_SRC_HOME=${CPP_SRC_HOME:-`git rev-parse --show-toplevel`}

# Ensure clang-format is available
which clang-format > /dev/null 2>&1
if [ $? != 0 ] ; then
    echo "ERROR: The clang-format code formatter is not available. Exiting."
    exit 1
fi

REQUIRED_CLANG_FORMAT_VERSION=5.0.1
FOUND_CLANG_FORMAT_VERSION=$(expr "`clang-format --version`" : ".* \([0-9].[0-9].[0-9]\)")

if [ -z "${FOUND_CLANG_FORMAT_VERSION}" ] ; then
    echo "ERROR: Required clang-format major version ${REQUIRED_CLANG_FORMAT_VERSION} not found."
    echo "       Could not determine clang-format version."
    exit 2
fi

if [ "${REQUIRED_CLANG_FORMAT_VERSION}" != "${FOUND_CLANG_FORMAT_VERSION}" ] ; then
    echo "ERROR: Required clang-format major version ${REQUIRED_CLANG_FORMAT_VERSION} not found."
    echo "       Detected clang-format version ${FOUND_CLANG_FORMAT_VERSION}"
    exit 3
fi

find $CPP_SRC_HOME \( -name 3rd_party -o -name build-setup \) -prune -o \( -name \*.cc -o -name \*.h \) -exec clang-format -i {} \;
