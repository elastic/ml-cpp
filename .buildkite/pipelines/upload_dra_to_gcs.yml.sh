#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Stage DRA artifacts and trigger unified-release DRA processing via the
# elastic/dra-prep-buildkite-plugin (replaces the Release Manager Docker step).
#

. .buildkite/scripts/common/base.sh

STACK_VERSION=$(awk -F= '/^elasticsearchVersion/ {print $2}' gradle.properties | xargs echo)

if [ -n "${VERSION_QUALIFIER:-}" ] ; then
    STACK_VERSION="${STACK_VERSION}-${VERSION_QUALIFIER}"
fi

if [ "${BUILD_SNAPSHOT:-true}" = "false" ] ; then
    DRA_WORKFLOW=staging
else
    DRA_WORKFLOW=snapshot
    STACK_VERSION="${STACK_VERSION}-SNAPSHOT"
fi

cat <<EOL
steps:
  - label: ":package: DRA Prep"
    key: "dra-prep"
    depends_on: create_dra_artifacts
    command: ".buildkite/scripts/stage_artifacts.sh"
    env:
      DRA_WORKFLOW: "${DRA_WORKFLOW}"
    agents:
      provider: gcp
    plugins:
      - elastic/dra-prep#v0.1.5:
          product_id: "ml-cpp"
          stack_version: "${STACK_VERSION}"
          workflow: "${DRA_WORKFLOW}"

  - label: ":pipeline: Trigger DRA processing"
    trigger: "unified-release-dra-processing"
    async: true
    depends_on: "dra-prep"
    build:
      env:
        DRA_PRODUCT_ID: "ml-cpp"
        DRA_STACK_VERSION: "${STACK_VERSION}"
        DRA_WORKFLOW: "${DRA_WORKFLOW}"
EOL
