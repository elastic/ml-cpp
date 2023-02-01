#!/usr/bin/env python
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

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

class PipelineStep(list):
    #def append_step(self, step: dict):
    #  self.append(step)

  def generate_step(self, label, action, platform, snapshot, candidate):
    label = f"Upload {action} pipeline for {label}"
    command = f"python3 .buildkite/pipelines/build_{platform}.json.py --action={action} --snapshot={snapshot} --candidate={candidate}| buildkite-agent pipeline upload"
    template = {
      "label": label,
      "depends_on": None,
      "command": command,
      "agents": {
        "image": "python",
      }
    }
    return template

