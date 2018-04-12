#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Reformats Ml native source code, using clang-format,  to ensure consistency.

# Ensure $CPP_SRC_HOME is set
if [ -z "$CPP_SRC_HOME" ] ; then
    echo '$CPP_SRC_HOME is not set'
    exit 1
fi

# Ensure clang-format is available
which clang-format > /dev/null 2>&1

if [ $? != 0 ] ; then
    echo "ERROR: The clang-format code formatter is not available. Exiting."
    exit 1;
fi

CLANG_FORMAT_MAJOR_VERSION=5
CLANG_FORMAT_VERSION=$(expr "`clang-format --version`" : ".* \(${CLANG_FORMAT_MAJOR_VERSION}.[0-9].[0-9]\) ")

if [ -z ${CLANG_FORMAT_VERSION} ]; then
    echo "ERROR: Require clang-format major version ${CLANG_FORMAT_MAJOR_VERSION}"
    exit 2
fi

find $CPP_SRC_HOME \( -name 3rd_party -o -name build-setup \) -prune -o \( -name \*.cc -o -name \*.h \) -exec clang-format -i {} \;
