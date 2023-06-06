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
# Upload all artifacts, both platform-specific and all-platforms, to
# GCS, where release manager builds will download them from.
#

. .buildkite/scripts/common/base.sh

cat <<EOL
steps:
  - label: "Upload DRA artifacts to GCS :gcloud:"
    key: "upload_dra_artifacts_to_gcs"
    depends_on: create_dra_artifacts
    command:
      - 'buildkite-agent artifact download "build/distributions/*" .'
      - 'echo "${RED}This step is disabled until BuildKite migration is complete${NOCOLOR}"'
      #- '.buildkite/scripts/steps/upload_dra_to_gcs.sh'
    agents:
      provider: gcp
EOL
