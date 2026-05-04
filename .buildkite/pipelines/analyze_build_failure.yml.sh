#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

EXTRA_FLAGS=""
if [ "${ML_ANALYZE_PREVIOUS:-}" = "true" ]; then
    EXTRA_FLAGS=" --find-previous-failure"
fi

cat <<EOL
steps:
  - label: "Analyze build failure :mag:"
    key: "analyze_build_failure"
    command:
        - |
            set -eu
            # Step-level if/build.state is evaluated at pipeline upload time, so it cannot
            # reliably gate on the final build outcome. Skip at job start when the build already
            # succeeded, except for the lightweight "find previous failure" pipeline.
            # If sibling steps are still running, BUILDKITE_BUILD_STATE is often "running" here
            # (not "passed"); analyze_build_failure.py then exits without Claude when the API
            # shows no failed/timed_out script jobs yet. Failures only in steps that start after
            # this job cannot be analyzed without widening depends_on.
            bs="\${BUILDKITE_BUILD_STATE:-}"
            if [ "\$bs" = "passed" ] && [ "\${ML_ANALYZE_PREVIOUS:-}" != "true" ]; then
              echo "Build state is passed; skipping failure analysis."
              exit 0
            fi
            python3 dev-tools/analyze_build_failure.py --pipeline \$BUILDKITE_PIPELINE_SLUG --build \$BUILDKITE_BUILD_NUMBER${EXTRA_FLAGS}
EOL

# Emit depends_on dynamically — ML_BUILD_STEP_KEYS and ML_TEST_STEP_KEYS are
# comma-separated lists set by the pipeline generator (branch builds expose
# both; PR pipelines may only set ML_BUILD_STEP_KEYS). In analyze-previous
# mode there are no build/test steps so this block is skipped.
DEPENDS_ON_KEYS=()
if [ -n "${ML_BUILD_STEP_KEYS:-}" ]; then
    IFS=',' read -ra STEP_KEYS <<< "$ML_BUILD_STEP_KEYS"
    DEPENDS_ON_KEYS+=("${STEP_KEYS[@]}")
fi
if [ -n "${ML_TEST_STEP_KEYS:-}" ]; then
    IFS=',' read -ra STEP_KEYS <<< "$ML_TEST_STEP_KEYS"
    DEPENDS_ON_KEYS+=("${STEP_KEYS[@]}")
fi
if [ "${#DEPENDS_ON_KEYS[@]}" -gt 0 ]; then
    echo '    depends_on:'
    seen=" "
    for key in "${DEPENDS_ON_KEYS[@]}"; do
        [ -z "$key" ] && continue
        case "$seen" in
            *" ${key} "*) continue ;;
        esac
        seen+=" ${key} "
        echo "        - \"${key}\""
    done
fi

cat <<'EOL'
    allow_dependency_failure: true
    soft_fail: true
    agents:
      image: "python:3"
EOL
