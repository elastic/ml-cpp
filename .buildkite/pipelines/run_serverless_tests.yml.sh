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

ML_CPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=dev-tools/serverless_buildkite_trigger_prepare.sh
source "${ML_CPP_ROOT}/dev-tools/serverless_buildkite_trigger_prepare.sh"

prepareMlCppServerlessTriggerContext "${BASH_SOURCE[0]}" || exit 1

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

assignServerlessQaTriggerEnvYamlEscapes

cat <<EOL
steps:
$(emitServerlessUploadMlCppDepsStepYaml)
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
        KEEP_DEPLOYMENT: "${KEEP_DEPLOYMENT_SAFE}"
        REGION_ID: "${REGION_ID_SAFE}"
        PROJECT_TYPE: "${PROJECT_TYPE_SAFE}"
EOL
