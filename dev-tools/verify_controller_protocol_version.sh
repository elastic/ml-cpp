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

# Shared helper for the ML C++ packaging paths.
#
# Elasticsearch's verifyControllerProtocolVersion gate rejects native controller
# bundles that do not contain a 'controller-protocol.version' marker. Failing
# fast at packaging time turns what would otherwise be an opaque downstream
# Elasticsearch build failure into a clear, local error.
#
# Usage: verify_controller_protocol_version <zip-file>
#   Returns non-zero if the marker is absent. If 'unzip' is unavailable the
#   check is skipped with a warning so images without it can still build.
verify_controller_protocol_version() {
    local zip_file="$1"
    if ! command -v unzip >/dev/null 2>&1 ; then
        echo "WARNING: unzip not available; skipping controller-protocol.version check for ${zip_file}" >&2
        return 0
    fi
    # Capture the full listing before matching. Piping 'unzip -l' straight into
    # 'grep -q' lets grep close the pipe as soon as it matches, which can deliver
    # SIGPIPE to 'unzip'; under 'set -o pipefail' (used by several callers) that
    # makes the pipeline non-zero and a present marker gets reported as missing.
    # A here-string avoids the pipe entirely.
    local listing
    if ! listing=$(unzip -l "$zip_file") ; then
        echo "ERROR: failed to list ${zip_file}" >&2
        return 1
    fi
    if ! grep -q 'controller-protocol\.version' <<< "$listing" ; then
        echo "ERROR: controller-protocol.version missing from ${zip_file}" >&2
        return 1
    fi
}
