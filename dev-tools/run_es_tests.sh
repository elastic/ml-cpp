#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Runs some Elasticsearch CI tests using C++ artifacts from a local Ivy repo.
# The elasticsearch fork and branch that are tested are based on the author
# and branches of the current PR, as recorded in the $PR_AUTHOR,
# $PR_SOURCE_BRANCH and $PR_TARGET_BRANCH environment variables.
#
# This is designed to run on an elasticsearch-ci Jenkins worker where all
# required versions of Java are installed in the Jenkins user's home directory.
#
# Arguments:
# $1 = Where to clone the elasticsearch repo
# $2 = Path to local Ivy repo

set -e

function isCloneTargetValid {
    FORK_TO_CHECK="$1"
    BRANCH_TO_CHECK="$2"
    echo "Checking for '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch"
    if [ -n "$(git ls-remote --heads "git@github.com:$FORK_TO_CHECK/elasticsearch.git" "$BRANCH_TO_CHECK" 2>/dev/null)" ]; then
        echo "Will use '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch for ES integration tests"
        return 0
    fi
    return 1
}

SELECTED_FORK=elastic
SELECTED_BRANCH=master

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

cd "$1"
rm -rf elasticsearch
git clone -b "$SELECTED_BRANCH" "git@github.com:${SELECTED_FORK}/elasticsearch.git" --depth=1
cd elasticsearch

export ES_BUILD_JAVA="$(grep "^ES_BUILD_JAVA" .ci/java-versions.properties | awk -F= '{ print $2 }' | xargs echo)"
if [ -z "$ES_BUILD_JAVA" ]; then
    echo "Unable to set JAVA_HOME, ES_BUILD_JAVA not present in .ci/java-versions.properties"
    exit 1
fi

# On aarch64:
# - openjdk is built with a 64KB page size
# - adoptopenjdk is built with a 4KB page size
# It's necessary to use use the one that matches the page size of the
# distribution that it's running on, which is:
# - 4KB for Ubuntu, Debian and SLES
# - 64KB for RHEL and CentOS
# We still disable the CDS archive as an extra measure to avoid warnings related
# to mismatched page sizes that can cause problems with Java (both for Gradle
# and the tests themselves).  This should cease to be necessary in Java 17 and
# above.
if [ `uname -m` = aarch64 ] ; then
    case `getconf PAGE_SIZE` in
        4096)
            export ES_BUILD_JAVA=adopt$ES_BUILD_JAVA
            ;;

        65536)
            export ES_BUILD_JAVA=$(echo $ES_BUILD_JAVA | sed 's/^adopt//')
            ;;

        *)
            echo "Unexpected page size:" `getconf PAGE_SIZE 2>&1`
            exit 2
            ;;
    esac
    export GRADLE_OPTS=-Xshare:off
    export EXTRA_TEST_OPTS="-Dtests.jvm.argline=-Xshare:off"
fi

echo "Setting JAVA_HOME=$HOME/.java/$ES_BUILD_JAVA"
export JAVA_HOME="$HOME/.java/$ES_BUILD_JAVA"
export RUNTIME_JAVA_HOME="$JAVA_HOME"

# For the ES build we need to:
# 1. Convince it that this is not part of a PR build, becuase it will get
#    confused that the PR is an ml-cpp PR rather than an elasticsearch PR
# 2. Set GIT_BRANCH to point at the elasticsearch branch, not the ml-cpp branch
# 3. Set GIT_COMMIT to point at the elasticsearch commit, not the ml-cpp commit
# 4. Set GIT_PREVIOUS_COMMIT the same as GIT_COMMIT as there are no changes to
#    Elasticsearch code in the current ML PR
unset ROOT_BUILD_CAUSE_GHPRBCAUSE
export GIT_BRANCH="$SELECTED_BRANCH"
export GIT_COMMIT="$(git rev-parse HEAD)"
export GIT_PREVIOUS_COMMIT="$GIT_COMMIT"

IVY_REPO_URL="file://$2"
./gradlew -Dbuild.ml_cpp.repo="$IVY_REPO_URL" :x-pack:plugin:ml:qa:native-multi-node-tests:javaRestTest $EXTRA_TEST_OPTS
./gradlew -Dbuild.ml_cpp.repo="$IVY_REPO_URL" :x-pack:plugin:yamlRestTest --tests "org.elasticsearch.xpack.test.rest.XPackRestIT.test {p0=ml/*}" $EXTRA_TEST_OPTS
