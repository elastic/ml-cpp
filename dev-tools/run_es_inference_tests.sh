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

# Runs Elasticsearch inference integration tests that exercise the
# pytorch_inference process via inference API default endpoints (ELSER, E5,
# rerank) and semantic text.  Designed to run as a separate Buildkite step
# in parallel with run_es_tests.sh.
#
# Arguments:
# $1 = Where to clone the elasticsearch repo
# $2 = Path to local Ivy repo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_es_tests_common.sh" "$1" "$2" \
    ':x-pack:plugin:inference:qa:inference-service-tests:javaRestTest --tests "org.elasticsearch.xpack.inference.DefaultEndPointsIT" --tests "org.elasticsearch.xpack.inference.TextEmbeddingCrudIT"' \
    ':x-pack:plugin:inference:yamlRestTest --tests "org.elasticsearch.xpack.inference.InferenceRestIT.test {p0=inference/30_semantic_text_inference/*}" --tests "org.elasticsearch.xpack.inference.InferenceRestIT.test {p0=inference/40_semantic_text_query/*}"'
