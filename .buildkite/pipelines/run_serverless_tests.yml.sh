#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Pipeline: build a custom ml-cpp, package it into a serverless ES Docker
# image, push the image, and trigger E2E tests against MKI QA.
#
# Depends on the linux x86_64 build step having already produced
# build/distributions/ml-cpp-*-linux-x86_64.zip as a Buildkite artifact.

SAFE_MESSAGE=$(printf '%s' "${BUILDKITE_MESSAGE}" | head -1 | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
PR_NUM="${BUILDKITE_PULL_REQUEST}"
if [ -z "${PR_NUM}" ] || [ "${PR_NUM}" = "false" ]; then
  PR_NUM="manual"
fi
IMAGE_TAG="ml-cpp-pr-${PR_NUM}-${BUILDKITE_BUILD_NUMBER}"

cat <<EOL
steps:
  - label: ":docker: Build serverless Docker image with custom ml-cpp"
    key: "serverless_docker_build"
    command:
      - 'buildkite-agent artifact download "build/*" . --step build_test_linux-x86_64-RelWithDebInfo'
      - '.buildkite/scripts/steps/build_serverless_docker.sh'
    depends_on: "build_test_linux-x86_64-RelWithDebInfo"
    agents:
      provider: aws
      instanceType: m6i.4xlarge
      imagePrefix: core-amazonlinux-2023
      diskSizeGb: 250
      diskName: '/dev/xvda'
    env:
      IMAGE_TAG: "${IMAGE_TAG}"
      ES_SERVERLESS_BRANCH: "${ES_SERVERLESS_BRANCH:-main}"
    notify:
      - github_commit_status:
          context: "Serverless Docker Build"

  - label: ":aws: :serverless: E2E tests (new project)"
    trigger: elasticsearch-serverless-e2e-tests-qa
    depends_on: "serverless_docker_build"
    build:
      message: "ml-cpp PR #${PR_NUM}: ${SAFE_MESSAGE}"
      env:
        DEPLOY_ID: "ml-cpp-pr-${PR_NUM}-\${BUILDKITE_BUILD_NUMBER}"
        REGION_ID: "aws-eu-west-1"
        IMAGE_OVERRIDE: "docker.elastic.co/elasticsearch-ci/elasticsearch-serverless:${IMAGE_TAG}"

  - label: ":aws: :serverless: E2E tests (upgraded project)"
    trigger: elasticsearch-serverless-e2e-tests-qa
    depends_on: "serverless_docker_build"
    build:
      message: "ml-cpp PR #${PR_NUM} (upgrade): ${SAFE_MESSAGE}"
      env:
        DEPLOY_ID: "ml-cpp-pr-${PR_NUM}-upgrade-\${BUILDKITE_BUILD_NUMBER}"
        REGION_ID: "aws-eu-west-1"
        IMAGE_OVERRIDE: "docker.elastic.co/elasticsearch-ci/elasticsearch-serverless:${IMAGE_TAG}"
        UPGRADED_PROJECT: true
EOL
