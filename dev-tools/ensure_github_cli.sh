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
# Ensure the GitHub CLI (gh) is available on PATH. Used by automated PR flows
# (e.g. dev-tools/create_github_pull_request.sh) when the image does not
# pre-install gh (Wolfi: try apk; Linux: fall back to GitHub release tarball).
#
# Environment:
#   SKIP_GH_AUTO_INSTALL — set to true to skip and exit non-zero if gh is missing
#   GH_CLI_VERSION — pinned release for tarball fallback (default below)

set -euo pipefail

if command -v gh >/dev/null 2>&1; then
    exit 0
fi

if [[ "${SKIP_GH_AUTO_INSTALL:-}" == "true" ]]; then
    echo "ERROR: gh not found and SKIP_GH_AUTO_INSTALL=true" >&2
    exit 1
fi

echo "Installing GitHub CLI (gh)..." >&2

# Wolfi / Alpine-style images (ml-cpp version-bump uses release-eng Wolfi)
if command -v apk >/dev/null 2>&1; then
    if apk add --no-cache gh 2>/dev/null || apk add --no-cache github-cli 2>/dev/null; then
        command -v gh >/dev/null 2>&1 && exit 0
    fi
fi

OS=$(uname -s)
ARCH=$(uname -m)
if [[ "$OS" != Linux ]]; then
    echo "ERROR: gh not installed; on ${OS} install from https://cli.github.com/ (e.g. brew install gh)." >&2
    exit 1
fi

case "$ARCH" in
    x86_64) GH_ARCH=amd64 ;;
    aarch64 | arm64) GH_ARCH=arm64 ;;
    *)
        echo "ERROR: unsupported Linux machine type for gh tarball: ${ARCH}" >&2
        exit 1
        ;;
esac

GH_CLI_VERSION="${GH_CLI_VERSION:-2.63.2}"
PREFIX="${GH_CLI_INSTALL_PREFIX:-/usr/local}"
BIN_DIR="${PREFIX}/bin"
if ! mkdir -p "$BIN_DIR" 2>/dev/null || [[ ! -w "$BIN_DIR" ]]; then
    echo "ERROR: cannot write gh to ${BIN_DIR}; install gh manually or run as a user that can write there." >&2
    exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT

URL="https://github.com/cli/cli/releases/download/v${GH_CLI_VERSION}/gh_${GH_CLI_VERSION}_linux_${GH_ARCH}.tar.gz"
if ! curl -fsSL "$URL" -o "${TMP}/gh.tgz"; then
    echo "ERROR: failed to download gh ${GH_CLI_VERSION} from GitHub releases (set GH_CLI_VERSION?)." >&2
    exit 1
fi

tar -xzf "${TMP}/gh.tgz" -C "${TMP}"
GH_BIN=$(find "${TMP}" -path '*/bin/gh' -type f | head -1)
if [[ -z "${GH_BIN}" ]]; then
    echo "ERROR: gh binary not found in release archive." >&2
    exit 1
fi

install -m 0755 "${GH_BIN}" "${BIN_DIR}/gh"
hash -r 2>/dev/null || true
echo "Installed gh to ${BIN_DIR}/gh" >&2

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh still not on PATH after install (ensure ${BIN_DIR} is on PATH)." >&2
    exit 1
fi
