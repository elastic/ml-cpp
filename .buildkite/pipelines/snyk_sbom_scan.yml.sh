#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Scan the third-party libraries that ship in the ml-cpp artifacts with Snyk.
# The Snyk GitHub integration only scans the Dockerfiles (build images), so the
# source-built libraries that actually ship (Boost, libtorch, ...) are checked
# here from an SBOM generated from 3rd_party/licenses/*INFO.csv.
#
# soft_fail while the baseline is being understood: findings turn the step
# yellow without failing the build.

cat <<'EOL'
steps:
  - label: "Snyk scan of shipped third-party libraries :snyk:"
    key: "snyk_sbom_scan"
    command:
        - ".buildkite/scripts/steps/snyk-sbom-scan.sh"
    soft_fail: true
    agents:
      image: "python:3.11-bookworm"
    notify:
      - github_commit_status:
          context: "Snyk scan of shipped third-party libraries"
EOL
