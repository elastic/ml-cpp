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

# Runs the core Elasticsearch ML integration tests: native multi-node Java
# REST tests and the ML YAML REST tests.
#
# When ES_TEST_SUITE is set to "javaRestTest" or "yamlRestTest", only that
# suite is run.  Otherwise both suites are run sequentially.
#
# Arguments:
# $1 = Where to clone the elasticsearch repo
# $2 = Path to local Ivy repo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${ES_TEST_SUITE:-}" in
  javaRestTest)
    exec "$SCRIPT_DIR/run_es_tests_common.sh" "$1" "$2" \
        ':x-pack:plugin:ml:qa:native-multi-node-tests:javaRestTest'
    ;;
  yamlRestTest)
    exec "$SCRIPT_DIR/run_es_tests_common.sh" "$1" "$2" \
        ':x-pack:plugin:yamlRestTest' \
        --tests 'org.elasticsearch.xpack.test.rest.XPackRestIT.test {p0=ml/*}'
    ;;
  *)
    exec "$SCRIPT_DIR/run_es_tests_common.sh" "$1" "$2" \
        ':x-pack:plugin:ml:qa:native-multi-node-tests:javaRestTest' \
        '---' \
        ':x-pack:plugin:yamlRestTest' \
        --tests 'org.elasticsearch.xpack.test.rest.XPackRestIT.test {p0=ml/*}'
    ;;
esac
