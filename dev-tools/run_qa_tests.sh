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

# Runs some QA tests using C++ artifacts from a local Ivy repo.
# The elasticsearch fork and branch that are tested are based on the author
# and branches of the current PR, as recorded in the $PR_AUTHOR,
# $PR_SOURCE_BRANCH and $PR_TARGET_BRANCH environment variables.
#
# This is designed to run on a BuildKite worker where all required versions of
# Java are installed in the BuildKite user's home directory.
#
# Arguments:
# $1 = Where to clone the elasticsearch repo
# $2 = Path to local Ivy repo

set -e

function isCloneTargetValid {
    FORK_TO_CHECK="$1"
    BRANCH_TO_CHECK="$2"
    echo "Checking for '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch"
    if [ -n "$(git ls-remote --heads "https://github.com/$FORK_TO_CHECK/elasticsearch.git" "$BRANCH_TO_CHECK" 2>/dev/null)" ]; then
        echo "Will use '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch for QA tests"
        return 0
    fi
    return 1
}

SELECTED_FORK=elastic
SELECTED_BRANCH=main

function pickCloneTarget {

    if isCloneTargetValid "$PR_AUTHOR" "$PR_SOURCE_BRANCH" ; then
        SELECTED_FORK="$PR_AUTHOR"
        SELECTED_BRANCH="$PR_SOURCE_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$SELECTED_FORK" "$PR_SOURCE_BRANCH" ; then
        SELECTED_BRANCH="$PR_SOURCE_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$SELECTED_FORK" "$PR_TARGET_BRANCH" ; then
        SELECTED_BRANCH="$PR_TARGET_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$SELECTED_FORK" "$SELECTED_BRANCH" ; then
        return 0
    fi

    return 1
}

pickCloneTarget

cd $1
IVY_REPO_URL="file://$2"

export SELECTED_FORK
export SELECTED_BRANCH
export IVY_REPO_URL

git clone git@github.com:elastic/qaf.git qaf
cd qaf
pip install -e .

export LOCAL_DEPLOYMENT_NAME=default-${VERSION}
qaf deployments create --stack-version ${RAW_VERSION} --plan 3-nodes --distribution-source GITHUB
qaf deployments start

cd ..
git clone git@github.com:elastic/qaf-tests.git qaf-tests
cd qaf-tests
USE_LOCAL_QAF=y make setup
source .venv/bin/activate

.buildkite/scripts/setup-qaf-tests.sh
.buildkite/scripts/pytest.sh tests/misc/notebooks
