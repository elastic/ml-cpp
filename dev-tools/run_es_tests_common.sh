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

# Common setup for running Elasticsearch integration tests using C++ artifacts
# from a local Ivy repo.  Clones the appropriate elasticsearch fork/branch,
# configures the Java and Gradle environment, then executes the Gradle commands
# passed as remaining arguments.
#
# The elasticsearch fork and branch that are tested are based on the author
# and branches of the current PR, as recorded in the $PR_AUTHOR,
# $PR_SOURCE_BRANCH and $PR_TARGET_BRANCH environment variables.
#
# This is designed to run on a Buildkite worker where all required versions of
# Java are installed in the Buildkite user's home directory.
#
# Arguments:
# $1 = Where to clone the elasticsearch repo
# $2 = Path to local Ivy repo
# $3... = Gradle commands to run.  Each argument is a complete command line
#         passed to ./gradlew with $GRADLE_JVM_OPTS, the Ivy repo property,
#         and $EXTRA_TEST_OPTS automatically appended.

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
SELECTED_BRANCH=main

function pickCloneTarget {

    if isCloneTargetValid "$GITHUB_PR_OWNER" "$GITHUB_PR_BRANCH" ; then
        SELECTED_FORK="$GITHUB_PR_OWNER"
        SELECTED_BRANCH="$GITHUB_PR_BRANCH"
        return 0
    fi

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

CLONE_DIR="$1"
IVY_REPO_PATH="$2"
shift 2

pickCloneTarget

cd "$CLONE_DIR"
rm -rf elasticsearch
git clone -b "$SELECTED_BRANCH" "git@github.com:${SELECTED_FORK}/elasticsearch.git" --depth=1
cd elasticsearch

if [ -z "${BUILDKITE}" ]; then
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
  # There's a link "jdk<version>" pointing to the appropriate JDK on each CI worker,
  # so strip any specifics from what was specified in .ci/java-versions.properties.
  if [ `uname -m` = aarch64 ] ; then
      export ES_BUILD_JAVA=$(echo $ES_BUILD_JAVA | sed 's/.*jdk/jdk/')
  fi

  echo "Setting JAVA_HOME=$HOME/.java/$ES_BUILD_JAVA"
  export JAVA_HOME="$HOME/.java/$ES_BUILD_JAVA"
fi

# For the ES build we need to:
# 1. Convince it that this is not part of a PR build, because it will get
#    confused that the PR is an ml-cpp PR rather than an elasticsearch PR
# 2. Set GIT_BRANCH to point at the elasticsearch branch, not the ml-cpp branch
# 3. Set GIT_COMMIT to point at the elasticsearch commit, not the ml-cpp commit
# 4. Set GIT_PREVIOUS_COMMIT the same as GIT_COMMIT as there are no changes to
#    Elasticsearch code in the current ML PR
unset ROOT_BUILD_CAUSE_GHPRBCAUSE
export GIT_BRANCH="$SELECTED_BRANCH"
export GIT_COMMIT="$(git rev-parse HEAD)"
export GIT_PREVIOUS_COMMIT="$GIT_COMMIT"

IVY_REPO_URL="file://$IVY_REPO_PATH"

ML_CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT_SCRIPT="$ML_CPP_ROOT/dev-tools/gradle-build-cache-init.gradle"
GRADLE_CACHE_DIR="$HOME/.gradle/caches/build-cache-1"
CACHE_ARGS=""
if [ -f "$INIT_SCRIPT" ]; then
    CACHE_ARGS="--build-cache --init-script $INIT_SCRIPT"
fi

# Restore Gradle build cache from GCS if credentials are available.
# This lets ephemeral CI agents reuse compilation outputs from prior builds.
CACHE_KEY="gradle-build-cache-$(uname -m)"
GCS_CACHE_PATH=""
if [ -n "${GRADLE_BUILD_CACHE_GCS_BUCKET:-}" ] && [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    GCS_CACHE_PATH="gs://${GRADLE_BUILD_CACHE_GCS_BUCKET}/${CACHE_KEY}.tar.gz"
    if command -v gsutil &>/dev/null; then
        if command -v gcloud &>/dev/null; then
            gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS" 2>/dev/null || true
        fi
        echo "--- Restoring Gradle build cache from $GCS_CACHE_PATH"
        mkdir -p "$GRADLE_CACHE_DIR"
        if gsutil -q stat "$GCS_CACHE_PATH" 2>/dev/null; then
            gsutil cp "$GCS_CACHE_PATH" /tmp/gradle-cache.tar.gz \
                && tar xzf /tmp/gradle-cache.tar.gz -C "$HOME/.gradle/caches/" \
                && rm -f /tmp/gradle-cache.tar.gz \
                && echo "Gradle build cache restored ($(du -sh "$GRADLE_CACHE_DIR" 2>/dev/null | cut -f1))" \
                || echo "Warning: failed to restore Gradle build cache, continuing without it"
        else
            echo "No cached Gradle build cache found, will build from scratch"
        fi
    else
        echo "gsutil not found, skipping Gradle build cache restore"
    fi
fi

for GRADLE_CMD in "$@" ; do
    eval ./gradlew $GRADLE_JVM_OPTS $CACHE_ARGS -Dbuild.ml_cpp.repo="$IVY_REPO_URL" $GRADLE_CMD $EXTRA_TEST_OPTS
done

# Upload Gradle build cache to GCS for future builds.
if [ -n "$GCS_CACHE_PATH" ] && [ -d "$GRADLE_CACHE_DIR" ] && command -v gsutil &>/dev/null; then
    echo "--- Uploading Gradle build cache to $GCS_CACHE_PATH"
    CACHE_SIZE=$(du -sm "$GRADLE_CACHE_DIR" 2>/dev/null | cut -f1)
    if [ "${CACHE_SIZE:-0}" -gt 0 ] && [ "${CACHE_SIZE:-0}" -lt 4096 ]; then
        tar czf /tmp/gradle-cache.tar.gz -C "$HOME/.gradle/caches/" build-cache-1 \
            && gsutil -o "GSUtil:parallel_composite_upload_threshold=50M" cp /tmp/gradle-cache.tar.gz "$GCS_CACHE_PATH" \
            && rm -f /tmp/gradle-cache.tar.gz \
            && echo "Gradle build cache uploaded (${CACHE_SIZE}M)" \
            || echo "Warning: failed to upload Gradle build cache"
    else
        echo "Skipping cache upload (size=${CACHE_SIZE:-0}M, expected 1-4095M)"
    fi
fi
