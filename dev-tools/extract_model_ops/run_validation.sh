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
# Wrapper that ensures the Python virtual environment exists and then
# runs validate_allowlist.py.  All arguments are forwarded to the script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
VALIDATE_SCRIPT="${SCRIPT_DIR}/validate_allowlist.py"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found on PATH" >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR}..." >&2
    python3 -m venv "${VENV_DIR}"
fi

if [ ! -f "${VENV_DIR}/.requirements.stamp" ] || \
   [ "${REQUIREMENTS}" -nt "${VENV_DIR}/.requirements.stamp" ]; then
    echo "Installing/updating dependencies..." >&2
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip
    "${VENV_DIR}/bin/pip" install --quiet -r "${REQUIREMENTS}"
    touch "${VENV_DIR}/.requirements.stamp"
fi

exec "${VENV_DIR}/bin/python3" "${VALIDATE_SCRIPT}" "$@"
