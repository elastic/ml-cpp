#!/usr/bin/env python
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

email_notification = {
  "label": "Queue an :email: notification for the pipeline",
  "depends_on": None,
  "command": ".buildkite/scripts/steps/send_email_notification.sh",
}

slack_notification = {
  "label": "Queue a :slack: notification for the pipeline",
  "depends_on": None,
  "command": ".buildkite/scripts/steps/send_slack_notification.sh",
}

format_and_validation = {
  "label": "Upload clang-format validation",
  "depends_on": None,
  "command": ".buildkite/pipelines/format_and_validation.yml.sh | buildkite-agent pipeline upload",
}

upload_to_s3 = {
  "label": "Upload artifact uploader pipeline",
  "depends_on": None,
  "command": ".buildkite/pipelines/upload_to_s3.yml.sh | buildkite-agent pipeline upload",
}

run_es_tests = {
  "label": "Upload ES tests runner pipeline",
  "depends_on": None,
  "command": ".buildkite/pipelines/run_es_tests.yml.sh | buildkite-agent pipeline upload",
}

build_windows = {
  "label": "Upload build pipeline for Windows",
  "depends_on": None,
  "command": "python3 .buildkite/pipelines/build_windows.json.py | buildkite-agent pipeline upload",
  "agents": {
    "image": "python",
  }
}

debug_windows = {
  "label": "Upload build pipeline for Windows",
  "depends_on": None,
  "command": "python3 .buildkite/pipelines/build_windows.json.py --action=debug | buildkite-agent pipeline upload",
  "agents": {
    "image": "python",
  }
}

build_linux = {
  "label": "Upload build pipeline for Linux",
  "depends_on": None,
  "command": "python3 .buildkite/pipelines/build_linux.json.py | buildkite-agent pipeline upload",
  "agents": {
    "image": "python",
  }
}

debug_linux = {
  "label": "Upload build pipeline for Linux",
  "depends_on": None,
  "command": "python3 .buildkite/pipelines/build_linux.json.py --action=debug | buildkite-agent pipeline upload",
  "agents": {
    "image": "python",
  }
}

build_macos = {
  "label": "Upload build pipeline for MacOS",
  "depends_on": None,
  "command": "python3 .buildkite/pipelines/build_macos.json.py | buildkite-agent pipeline upload",
  "agents": {
    "image": "python",
  }
}

debug_macos = {
  "label": "Upload build pipeline for MacOS",
  "depends_on": None,
  "command": "python3 .buildkite/pipelines/build_macos.json.py --action=debug | buildkite-agent pipeline upload",
  "agents": {
    "image": "python",
  }
}

class PipelineStep(list):
  def append_with_extended_os(self, step: dict, extended_os=False):
    self.append(step)
