#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

cat <<EOL
steps:
  - label: "Java :java: Integration Tests for aarch64 :hammer:"
    key: "java_integration_tests_aarch64"
    command:
      - 'sudo mkdir -p /usr/lib/jvm/amazon-corretto-21 && wget -O - https://corretto.aws/downloads/latest/amazon-corretto-21-aarch64-linux-jdk.tar.gz | sudo tar xvzf - -C /usr/lib/jvm/amazon-corretto-21 --strip=1'
      - 'ls -lR /usr/lib/jvm'
      # - 'buildkite-agent artifact download "build/*" . --step build_test_linux-aarch64-RelWithDebInfo'
      - '.buildkite/scripts/steps/run_es_tests.sh || (cd ../elasticsearch && find x-pack -name logs | xargs tar cvzf logs.tgz && buildkite-agent artifact upload logs.tgz && false)'
    # depends_on: "build_test_linux-aarch64-RelWithDebInfo"
    agents:
      provider: aws
      instanceType: m6g.2xlarge
      imagePrefix: ci-amazonlinux-2-aarch64
      diskSizeGb: 100
      diskName: '/dev/xvda'
    env:
      JAVA_HOME: "/usr/lib/jvm/amazon-corretto-21"
      IVY_REPO: "../ivy"
      GRADLE_JVM_OPTS: "-Dorg.gradle.jvmargs=-Xmx16g"
    notify:
      - github_commit_status:
          context: "Java Integration Tests for aarch64"
EOL
