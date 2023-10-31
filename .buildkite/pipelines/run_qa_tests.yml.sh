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
  - label: "Appex QA Tests :test_tube:"
    key: "appex_qa_tests"
    command: 
      - "sudo apt-get -y install openjdk-17-jre openjdk-17-jdk"
      - 'buildkite-agent artifact download "build/*" . --step build_test_linux-x86_64-RelWithDebInfo'
      - '.buildkite/scripts/steps/run_qa_tests.sh || (cd ../elasticsearch && find x-pack -name logs | xargs tar cvzf logs.tgz && buildkite-agent artifact upload logs.tgz && false)'
    depends_on: "build_test_linux-x86_64-RelWithDebInfo"
    agents:
      "cpu": "16",
      "ephemeralStorage": "20G",
      "memory": "64G",
      "image": "docker.elastic.co/employees/dolaru/qaf:latest"
    env:
      IVY_REPO: "../ivy"
      GRADLE_JVM_OPTS: "-Dorg.gradle.jvmargs=-Xmx16g"
    notify:
      - github_commit_status:
          context: "QA Tests"
EOL
