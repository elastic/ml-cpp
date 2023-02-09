#!/usr/bin/env python
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

class PipelineStep(list):
  def generate_step(self, label, command):
    command = command + " | buildkite-agent pipeline upload"
    step = {
      "label": label,
      "depends_on": None,
      "command": command,
      "agents": {
        "image": "python",
      }
    }
    return step

  def generate_step_template(self, platform, action, snapshot, candidate):
    platform_lower = platform.lower()
    platform_emoji = ":"+platform_lower+":"
    label = f"Upload {action} pipeline for {platform} {platform_emoji}"
    command = f"python3 .buildkite/pipelines/build_{platform_lower}.json.py --action={action} --snapshot={snapshot} --candidate={candidate}"
    return self.generate_step(label, command)
