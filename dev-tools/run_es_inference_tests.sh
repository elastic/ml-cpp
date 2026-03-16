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

# Runs Elasticsearch inference integration tests using C++ artifacts from a
# local Ivy repo.  These tests exercise the pytorch_inference process via the
# inference API default endpoints (ELSER, E5, rerank) and semantic text.
#
# This script mirrors run_es_tests.sh but targets inference-specific test
# suites.  It is designed to run as a separate Buildkite step in parallel
# with run_es_tests.sh.
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
        echo "Will use '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch for ES inference tests"
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

pickCloneTarget

cd "$1"
rm -rf elasticsearch
git clone -b "$SELECTED_BRANCH" "git@github.com:${SELECTED_FORK}/elasticsearch.git" --depth=1
cd elasticsearch

if [ -z "${BUILDKITE}" ]; then
  export ES_BUILD_JAVA="$(grep "^ES_BUILD_JAVA" .ci/java-versions.properties | awk -F= '{ print $2 }' | xargs echo)"
  if [ -z "$ES_BUILD_JAVA" ]; then
      echo "Unable to set JAVA_HOME, ES_BUILD_JAVA not present in .ci/java-versions.properties"
      exit 1
  fi

  if [ `uname -m` = aarch64 ] ; then
      export ES_BUILD_JAVA=$(echo $ES_BUILD_JAVA | sed 's/.*jdk/jdk/')
  fi

  echo "Setting JAVA_HOME=$HOME/.java/$ES_BUILD_JAVA"
  export JAVA_HOME="$HOME/.java/$ES_BUILD_JAVA"
fi

unset ROOT_BUILD_CAUSE_GHPRBCAUSE
export GIT_BRANCH="$SELECTED_BRANCH"
export GIT_COMMIT="$(git rev-parse HEAD)"
export GIT_PREVIOUS_COMMIT="$GIT_COMMIT"

IVY_REPO_URL="file://$2"

# Default inference endpoints: ELSER, E5, rerank.  These deploy real
# prepacked models via the local MlModelServer and exercise the full
# pytorch_inference pipeline including graph validation.
./gradlew $GRADLE_JVM_OPTS -Dbuild.ml_cpp.repo="$IVY_REPO_URL" \
  :x-pack:plugin:inference:qa:inference-service-tests:javaRestTest \
  --tests "org.elasticsearch.xpack.inference.DefaultEndPointsIT" \
  --tests "org.elasticsearch.xpack.inference.TextEmbeddingCrudIT" \
  $EXTRA_TEST_OPTS

# Semantic text YAML REST tests.  Most use mock services but the suites
# include tests that exercise the default ELSER 2 endpoint for indexing
# and querying, which deploy real models.
./gradlew $GRADLE_JVM_OPTS -Dbuild.ml_cpp.repo="$IVY_REPO_URL" \
  :x-pack:plugin:inference:yamlRestTest \
  --tests "org.elasticsearch.xpack.inference.InferenceRestIT.test {p0=inference/30_semantic_text_inference/*}" \
  --tests "org.elasticsearch.xpack.inference.InferenceRestIT.test {p0=inference/40_semantic_text_query/*}" \
  $EXTRA_TEST_OPTS
