#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# THROWAWAY (do not merge): run the Orka distrust script on a real
# ml-macos-14-arm-004.orkasi VM.
cat <<EOL
steps:
  - label: "Orka distrust script check (macOS aarch64, image -004)"
    key: "orka_vm_check"
    command: ".buildkite/scripts/steps/orka_vm_check.sh"
    timeout_in_minutes: 20
    agents:
      provider: "orka"
      image: "ml-macos-14-arm-004.orkasi"
EOL
