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

# Downloads and configures sccache for use as a compiler cache with a GCS backend.
#
# This script is sourced (not executed) so that it can export environment
# variables into the calling shell. It sets:
#   SCCACHE_PATH       - absolute path to the sccache binary
#   CMAKE_FLAGS        - appends compiler launcher flags (if CMAKE_FLAGS is already set)
#   SCCACHE_GCS_BUCKET - GCS bucket name for the cache
#   SCCACHE_GCS_KEY_PREFIX - per-platform prefix within the bucket
#   SCCACHE_GCS_RW_MODE   - read-write mode
#
# Prerequisites:
#   SCCACHE_GCS_BUCKET must be set (or defaults to elastic-ml-cpp-sccache)
#   For GCS auth, one of:
#     - GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON key
#     - SCCACHE_GCS_KEY_PATH pointing to a service account JSON key
#     - Running on a GCE instance with a service account attached
#
# Usage:
#   source dev-tools/setup_sccache.sh    # downloads + configures
#   cmake -B build ... -DCMAKE_CXX_COMPILER_LAUNCHER=$SCCACHE_PATH ...

set -e

SCCACHE_VERSION="v0.14.0"
SCCACHE_INSTALL_DIR="${SCCACHE_INSTALL_DIR:-/usr/local/bin}"

detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Linux)
            case "$arch" in
                x86_64)  echo "x86_64-unknown-linux-musl" ;;
                aarch64) echo "aarch64-unknown-linux-musl" ;;
                *)       echo "UNSUPPORTED"; return 1 ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                arm64|aarch64) echo "aarch64-apple-darwin" ;;
                x86_64)        echo "x86_64-apple-darwin" ;;
                *)             echo "UNSUPPORTED"; return 1 ;;
            esac
            ;;
        *)
            echo "UNSUPPORTED"; return 1
            ;;
    esac
}

install_sccache() {
    if command -v sccache &>/dev/null; then
        SCCACHE_PATH=$(command -v sccache)
        echo "sccache already installed: $SCCACHE_PATH ($(sccache --version))"
        return 0
    fi

    local platform
    platform=$(detect_platform) || { echo "ERROR: Unsupported platform for sccache"; return 1; }

    local tarball="sccache-${SCCACHE_VERSION}-${platform}.tar.gz"
    local url="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${tarball}"
    local tmpdir=$(mktemp -d)

    echo "Downloading sccache ${SCCACHE_VERSION} for ${platform}..."
    curl -fsSL "$url" -o "${tmpdir}/${tarball}"
    tar xzf "${tmpdir}/${tarball}" -C "${tmpdir}"

    local binary="${tmpdir}/sccache-${SCCACHE_VERSION}-${platform}/sccache"
    if [ ! -f "$binary" ]; then
        echo "ERROR: sccache binary not found after extraction"
        rm -rf "$tmpdir"
        return 1
    fi

    chmod +x "$binary"

    if [ -w "$SCCACHE_INSTALL_DIR" ]; then
        cp "$binary" "$SCCACHE_INSTALL_DIR/sccache"
    else
        mkdir -p "$HOME/.local/bin"
        cp "$binary" "$HOME/.local/bin/sccache"
        SCCACHE_INSTALL_DIR="$HOME/.local/bin"
        export PATH="$HOME/.local/bin:$PATH"
    fi

    rm -rf "$tmpdir"
    SCCACHE_PATH="${SCCACHE_INSTALL_DIR}/sccache"
    echo "sccache installed: $SCCACHE_PATH ($(sccache --version))"
}

configure_gcs_backend() {
    export SCCACHE_GCS_BUCKET="${SCCACHE_GCS_BUCKET:-elastic-ml-cpp-sccache}"

    local arch=$(uname -m | sed 's/arm64/aarch64/')
    local os=$(uname -s | tr 'A-Z' 'a-z')
    export SCCACHE_GCS_KEY_PREFIX="${SCCACHE_GCS_KEY_PREFIX:-${os}-${arch}}"

    export SCCACHE_GCS_RW_MODE="READ_WRITE"

    if [ -n "$SCCACHE_GCS_KEY_PATH" ]; then
        echo "sccache GCS auth: service account key at $SCCACHE_GCS_KEY_PATH"
    elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        export SCCACHE_GCS_KEY_PATH="$GOOGLE_APPLICATION_CREDENTIALS"
        echo "sccache GCS auth: GOOGLE_APPLICATION_CREDENTIALS at $GOOGLE_APPLICATION_CREDENTIALS"
    else
        echo "sccache GCS auth: using instance metadata / workload identity"
    fi

    echo "sccache GCS config: bucket=$SCCACHE_GCS_BUCKET prefix=$SCCACHE_GCS_KEY_PREFIX"
}

append_cmake_flags() {
    local launcher_flags="-DCMAKE_CXX_COMPILER_LAUNCHER=${SCCACHE_PATH} -DCMAKE_C_COMPILER_LAUNCHER=${SCCACHE_PATH}"

    if [ -n "$CMAKE_FLAGS" ]; then
        export CMAKE_FLAGS="${CMAKE_FLAGS} ${launcher_flags}"
    else
        export CMAKE_FLAGS="${launcher_flags}"
    fi
    echo "sccache: CMAKE_FLAGS updated with compiler launcher"
}

start_server() {
    "$SCCACHE_PATH" --stop-server &>/dev/null || true
    "$SCCACHE_PATH" --start-server
    echo "sccache server started"
}

show_stats() {
    echo "=== sccache stats ==="
    "$SCCACHE_PATH" --show-stats
    echo "===================="
}

# --- Main ---
install_sccache
configure_gcs_backend
append_cmake_flags
start_server

echo "sccache setup complete"
