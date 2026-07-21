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
# Run the Sandbox2 attack-defense integration test against freshly built
# controller and pytorch_inference binaries. Linux only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$(uname -s)" != "Linux" ]; then
  echo "Sandbox2 attack-defense test is Linux-only; skipping"
  exit 0
fi

if [ ! -e /proc/sys/kernel/unprivileged_userns_clone ] && [ "$(id -u)" -ne 0 ]; then
  echo "Skipping Sandbox2 attack-defense test: user namespaces not available"
  exit 0
fi

cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to run Sandbox2 attack-defense tests" >&2
  exit 1
fi

exec python3 "$ROOT/test/test_sandbox2_attack_defense.py" "$@"
