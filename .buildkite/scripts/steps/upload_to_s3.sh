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
# If this isn't a PR build and isn't a debug build then upload the artifacts.
# Experience indicates that BuildKite always sets BUILDKITE_PULL_REQUEST to
# be the PR number or "false". Hence we explicitly check for "false" here.
if [[ x"$BUILDKITE_PULL_REQUEST" = xfalse && -z "$ML_DEBUG" ]] ; then
    . .buildkite/scripts/common/aws_creds_from_vault.sh
    echo 'Uploading artifacts to S3'
    ./gradlew --info -Dbuild.version_qualifier=$VERSION_QUALIFIER -Dbuild.snapshot=$BUILD_SNAPSHOT upload
fi

