#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

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

FILES=()

if [ "x$1" = "x--all" ] ; then
    # Batch mode - check everything and only display errors
    INFILES=`find $CPP_SRC_HOME \( -name 3rd_party -o -name build-setup \) -prune -o \( -name \*.cc -o -name \*.h \) -print`
    for FILE in ${INFILES}; do
        if ! cmp -s ${FILE} <(clang-format ${FILE}); then
            FILES+=("${FILE}")
        fi
    done
else
    # Local mode - check changed files only and report which files are checked
    INFILES=`git diff --name-only --diff-filter=ACMRT | grep -v 3rd_party | grep -E "\.(cc|h)$"`
    for FILE in ${INFILES}; do
        FQFILE=${CPP_SRC_HOME}/${FILE}
        echo "Checking: ${FILE}"
        if ! cmp -s ${FQFILE} <(clang-format ${FQFILE}); then
            FILES+=("${FILE}")
        fi
    done
fi

if [ -n "${FILES}" ] ; then
    echo "A format error has been detected within the following files:"
    printf "%s\n" "${FILES[@]}"
    exit 4
fi

