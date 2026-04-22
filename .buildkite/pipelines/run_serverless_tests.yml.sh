#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Pipeline: trigger the elasticsearch-serverless validation pipeline to build
# a Docker image incorporating custom ml-cpp artifacts from this build, then
# run E2E tests against MKI QA.
#
# The triggered pipeline uses $BUILDKITE_TRIGGERED_FROM_BUILD_ID to download
# ml-cpp artifacts from this build via buildkite-agent, sets up a local Ivy
# repo, and passes -Dbuild.ml_cpp.repo to the Gradle Docker build.
#
# This avoids cloning elasticsearch-serverless or needing AWS credentials
# in the ml-cpp PR pipeline.

SAFE_MESSAGE=$(printf '%s' "${BUILDKITE_MESSAGE}" | head -1 | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
PR_NUM="${BUILDKITE_PULL_REQUEST}"
if [ -z "${PR_NUM}" ] || [ "${PR_NUM}" = "false" ]; then
  PR_NUM="manual"
fi

# Extract PR metadata once for reuse by all resolution steps below.
PR_AUTHOR_FORK="$(expr "${BUILDKITE_BRANCH:-}" : '\(.*\):.*' 2>/dev/null || true)"
PR_SOURCE="$(expr "${BUILDKITE_BRANCH:-}" : '.*:\(.*\)' 2>/dev/null || true)"
PR_TARGET="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

ML_CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=dev-tools/pick_elasticsearch_clone_target.sh
source "${ML_CPP_ROOT}/dev-tools/pick_elasticsearch_clone_target.sh"
export PR_AUTHOR="${PR_AUTHOR_FORK}"
export PR_SOURCE_BRANCH="${PR_SOURCE}"
export PR_TARGET_BRANCH="${PR_TARGET}"

# --- Resolve elasticsearch-serverless branch ---
# Same fork/branch *idea* as pick_elasticsearch_clone_target.sh (different repo);
# ES submodule SHA below uses that script directly.
# The trigger step can only use branches on elastic/elasticsearch-serverless,
# so if a matching branch is found on a fork but not on elastic/, we warn.
SERVERLESS_BRANCH="main"

check_serverless_branch() {
  local repo="$1" branch="$2"
  [ -n "$branch" ] && git ls-remote --heads "https://github.com/${repo}/elasticsearch-serverless.git" "$branch" 2>/dev/null | grep -q .
}

if [ -n "${ES_SERVERLESS_BRANCH:-}" ]; then
  SERVERLESS_BRANCH="${ES_SERVERLESS_BRANCH}"
  echo "Using explicit ES_SERVERLESS_BRANCH override: $SERVERLESS_BRANCH" >&2
else
  if [ -n "$PR_AUTHOR_FORK" ] && check_serverless_branch "$PR_AUTHOR_FORK" "$PR_SOURCE"; then
    if check_serverless_branch "elastic" "$PR_SOURCE"; then
      SERVERLESS_BRANCH="$PR_SOURCE"
      echo "Found '$PR_SOURCE' on both $PR_AUTHOR_FORK and elastic; using elastic/" >&2
    else
      echo "WARNING: Found '$PR_SOURCE' on $PR_AUTHOR_FORK/elasticsearch-serverless but not on elastic/." >&2
      echo "The trigger step can only use branches on elastic/elasticsearch-serverless." >&2
      echo "Push the branch to elastic/ or set ES_SERVERLESS_BRANCH explicitly." >&2
    fi
  elif check_serverless_branch "elastic" "$PR_SOURCE"; then
    SERVERLESS_BRANCH="$PR_SOURCE"
  elif [ "$PR_TARGET" != "main" ] && check_serverless_branch "elastic" "$PR_TARGET"; then
    SERVERLESS_BRANCH="$PR_TARGET"
  fi
fi
echo "Resolved elasticsearch-serverless branch: $SERVERLESS_BRANCH" >&2

# --- Resolve ES submodule commit (same fork/branch rules as run_es_tests_common.sh) ---
pickCloneTarget || true
ES_COMMIT="$(elasticsearch_selected_branch_head_sha)"
ES_COMMIT="${ES_COMMIT:-HEAD}"
echo "Resolved elasticsearch submodule: ${SELECTED_FORK}/${SELECTED_BRANCH} -> ${ES_COMMIT}" >&2

# --- Resolve ES PR number ---
# The serverless pipeline's PR-specific tests step looks up labels from the
# ES PR. First tries the ml-cpp PR author's matching ES PR (coordinated
# changes), then falls back to any recent open ES PR.
ES_PR_NUM=""
if [ -z "${ELASTICSEARCH_PR_NUMBER:-}" ]; then
  if [ -n "$PR_AUTHOR_FORK" ] && [ -n "$PR_SOURCE" ]; then
    ES_PR_NUM=$(curl -s "https://api.github.com/repos/elastic/elasticsearch/pulls?head=${PR_AUTHOR_FORK}:${PR_SOURCE}&state=open&per_page=1" 2>/dev/null \
      | python3 -c "import sys,json; prs=json.load(sys.stdin); print(prs[0]['number'] if prs else '')" 2>/dev/null || true)
  fi
  if [ -z "$ES_PR_NUM" ]; then
    ES_PR_NUM=$(curl -s "https://api.github.com/repos/elastic/elasticsearch/pulls?state=open&per_page=1" 2>/dev/null \
      | python3 -c "import sys,json; prs=json.load(sys.stdin); print(prs[0]['number'] if prs else '')" 2>/dev/null || true)
  fi
fi
ES_PR_NUM="${ELASTICSEARCH_PR_NUMBER:-${ES_PR_NUM}}"
if [ -z "$ES_PR_NUM" ]; then
  echo "WARNING: Could not resolve an ES PR number. The serverless PR-specific tests step may fail." >&2
fi
echo "Using ES submodule commit: $ES_COMMIT, ES PR number: $ES_PR_NUM" >&2

cat <<EOL
steps:
  - label: ":package: Upload ml-cpp deps artifact"
    key: "upload_ml_cpp_deps"
    command: 'buildkite-agent artifact upload dev-tools/minimal.zip'
    depends_on:
      - "build_test_linux-x86_64-RelWithDebInfo"
      - "build_test_linux-aarch64-RelWithDebInfo"
    agents:
      provider: aws
      instanceType: m6i.xlarge
      imagePrefix: core-amazonlinux-2023
      diskSizeGb: 100
      diskName: '/dev/xvda'

  - label: ":docker: :serverless: Build serverless image with custom ml-cpp"
    depends_on: "upload_ml_cpp_deps"
    async: false
    trigger: elasticsearch-serverless-es-pr-check
    build:
      branch: "${SERVERLESS_BRANCH}"
      message: "ml-cpp PR #${PR_NUM}: ${SAFE_MESSAGE}"
      env:
        UPDATE_SUBMODULE: "false"
        ML_CPP_BUILD_ID: "${BUILDKITE_BUILD_ID}"
        # ml-cpp repo commit at trigger time; serverless folds this into IMAGE_TAG
        # with ML_CPP_BUILD_ID so Docker tags never collide with stock builds.
        ML_CPP_COMMIT: "${BUILDKITE_COMMIT}"
        ELASTICSEARCH_SUBMODULE_COMMIT: "${ES_COMMIT}"
        ELASTICSEARCH_PR_NUMBER: "${ES_PR_NUM}"
EOL
