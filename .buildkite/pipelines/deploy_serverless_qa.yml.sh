#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Pipeline: build a serverless Docker image with custom ml-cpp and deploy it
# to the QA environment for interactive use. Unlike run_serverless_tests.yml.sh,
# this does NOT run E2E tests -- it just gets the environment running so the
# developer can interact with it (deploy models, run queries, kubectl, etc.).
#
# The deployment stays up for 1 hour by default. Set KEEP_DEPLOYMENT=true
# (via the Buildkite UI) to keep it longer. The build annotations will
# contain the URL and encrypted credentials for accessing the deployment.

SAFE_MESSAGE=$(printf '%s' "${BUILDKITE_MESSAGE}" | head -1 | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
PR_NUM="${BUILDKITE_PULL_REQUEST}"
if [ -z "${PR_NUM}" ] || [ "${PR_NUM}" = "false" ]; then
  PR_NUM="manual"
fi

# Extract PR metadata once for reuse.
PR_AUTHOR_FORK="$(expr "${BUILDKITE_BRANCH:-}" : '\(.*\):.*' 2>/dev/null || true)"
PR_SOURCE="$(expr "${BUILDKITE_BRANCH:-}" : '.*:\(.*\)' 2>/dev/null || true)"
PR_TARGET="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

# --- Resolve elasticsearch-serverless branch ---
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
      echo "Push the branch to elastic/ or set ES_SERVERLESS_BRANCH explicitly." >&2
    fi
  elif check_serverless_branch "elastic" "$PR_SOURCE"; then
    SERVERLESS_BRANCH="$PR_SOURCE"
  elif [ "$PR_TARGET" != "main" ] && check_serverless_branch "elastic" "$PR_TARGET"; then
    SERVERLESS_BRANCH="$PR_TARGET"
  fi
fi
echo "Resolved elasticsearch-serverless branch: $SERVERLESS_BRANCH" >&2

# --- Resolve ES submodule commit ---
ES_COMMIT=""
if [ -n "$PR_AUTHOR_FORK" ] && [ -n "$PR_SOURCE" ]; then
  ES_COMMIT=$(git ls-remote --heads "https://github.com/${PR_AUTHOR_FORK}/elasticsearch.git" "$PR_SOURCE" 2>/dev/null | awk '{print $1}')
  if [ -n "$ES_COMMIT" ]; then
    echo "Using ES commit from ${PR_AUTHOR_FORK}/elasticsearch:${PR_SOURCE}" >&2
  fi
fi
if [ -z "$ES_COMMIT" ]; then
  ES_COMMIT=$(git ls-remote --heads "https://github.com/elastic/elasticsearch.git" main 2>/dev/null | awk '{print $1}')
  ES_COMMIT="${ES_COMMIT:-HEAD}"
fi

echo "Deploying to serverless QA with custom ml-cpp from PR #${PR_NUM}" >&2

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

  - label: ":rocket: Deploy custom ml-cpp to serverless QA"
    depends_on: "upload_ml_cpp_deps"
    async: false
    trigger: elasticsearch-serverless-deploy-qa
    build:
      branch: "${SERVERLESS_BRANCH}"
      message: "ml-cpp PR #${PR_NUM}: ${SAFE_MESSAGE}"
      env:
        ML_CPP_BUILD_ID: "${BUILDKITE_BUILD_ID}"
        ELASTICSEARCH_SUBMODULE_COMMIT: "${ES_COMMIT}"
        KEEP_DEPLOYMENT: "${KEEP_DEPLOYMENT:-false}"
        REGION_ID: "${REGION_ID:-aws-eu-west-1}"
EOL
