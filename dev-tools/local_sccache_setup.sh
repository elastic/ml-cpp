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

# Sets up sccache for local development builds with optional GCS shared cache.
#
# Usage:
#   # Local disk cache only (default, zero config):
#   source dev-tools/local_sccache_setup.sh
#
#   # With GCS shared cache (read-only, uses CI-populated cache):
#   source dev-tools/local_sccache_setup.sh --gcs
#
#   # Then build as usual:
#   cmake -B cmake-build-relwithdebinfo
#   cmake --build cmake-build-relwithdebinfo -j$(nproc)
#
# sccache is auto-detected by CMakeLists.txt, so after installing it once
# you don't strictly need this script. It's provided for:
#   - Installing sccache if not already present
#   - Configuring the GCS shared cache backend
#   - Starting the sccache server with the right settings
#
# The GCS shared cache stores compilation results from CI builds. When
# enabled, your first local build after pulling changes will get cache
# hits for any files that CI has already compiled — typically giving
# near-instant rebuilds for upstream merges.

set -e

USE_GCS=false
GCS_RW_MODE="READ_ONLY"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gcs)
            USE_GCS=true
            shift
            ;;
        --gcs-rw)
            USE_GCS=true
            GCS_RW_MODE="READ_WRITE"
            shift
            ;;
        -h|--help)
            echo "Usage: source dev-tools/local_sccache_setup.sh [--gcs] [--gcs-rw]"
            echo ""
            echo "Options:"
            echo "  --gcs      Enable GCS shared cache (read-only)"
            echo "  --gcs-rw   Enable GCS shared cache (read-write, requires service account)"
            echo ""
            echo "Without flags, uses a local disk cache at ~/.cache/sccache"
            return 0 2>/dev/null || exit 0
            ;;
        *)
            echo "Unknown option: $1"
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

install_sccache() {
    if command -v sccache &>/dev/null; then
        SCCACHE_PATH=$(command -v sccache)
        echo "sccache already installed: $SCCACHE_PATH ($(sccache --version 2>&1 | head -1))"
        return 0
    fi

    local os=$(uname -s)
    case "$os" in
        Darwin)
            if command -v brew &>/dev/null; then
                echo "Installing sccache via Homebrew..."
                brew install sccache
                SCCACHE_PATH=$(command -v sccache)
            else
                echo "ERROR: sccache not found. Install with: brew install sccache"
                return 1
            fi
            ;;
        Linux)
            if command -v cargo &>/dev/null; then
                echo "Installing sccache via cargo..."
                cargo install sccache
                SCCACHE_PATH="$HOME/.cargo/bin/sccache"
            else
                echo "ERROR: sccache not found. Install with one of:"
                echo "  cargo install sccache"
                echo "  sudo apt install sccache    # Debian/Ubuntu"
                echo "  sudo dnf install sccache    # Fedora/RHEL"
                return 1
            fi
            ;;
        *)
            echo "ERROR: Unsupported platform. Install sccache manually:"
            echo "  https://github.com/mozilla/sccache/releases"
            return 1
            ;;
    esac

    if [ -z "$SCCACHE_PATH" ]; then
        SCCACHE_PATH=$(command -v sccache 2>/dev/null || true)
    fi

    if [ -z "$SCCACHE_PATH" ]; then
        echo "ERROR: sccache installation failed"
        return 1
    fi

    echo "sccache installed: $SCCACHE_PATH ($(sccache --version 2>&1 | head -1))"
}

configure_local_cache() {
    export SCCACHE_DIR="${SCCACHE_DIR:-$HOME/.cache/sccache}"
    export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-10G}"
    mkdir -p "$SCCACHE_DIR"
    echo "sccache local cache: $SCCACHE_DIR (max ${SCCACHE_CACHE_SIZE})"
}

configure_gcs_cache() {
    export SCCACHE_GCS_BUCKET="${SCCACHE_GCS_BUCKET:-elastic-ml-cpp-sccache}"
    export SCCACHE_GCS_RW_MODE="$GCS_RW_MODE"

    local arch=$(uname -m | sed 's/arm64/aarch64/')
    local os=$(uname -s | tr 'A-Z' 'a-z')
    export SCCACHE_GCS_KEY_PREFIX="${SCCACHE_GCS_KEY_PREFIX:-${os}-${arch}}"

    if [ -n "$SCCACHE_GCS_KEY_PATH" ]; then
        echo "sccache GCS auth: service account key at $SCCACHE_GCS_KEY_PATH"
    elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        export SCCACHE_GCS_KEY_PATH="$GOOGLE_APPLICATION_CREDENTIALS"
        echo "sccache GCS auth: GOOGLE_APPLICATION_CREDENTIALS"
    elif command -v gcloud &>/dev/null; then
        local adc="$HOME/.config/gcloud/application_default_credentials.json"
        if [ -f "$adc" ]; then
            export SCCACHE_GCS_KEY_PATH="$adc"
            echo "sccache GCS auth: gcloud application default credentials"
        else
            echo "sccache GCS auth: run 'gcloud auth application-default login' first"
            echo "Falling back to local-only cache"
            return 1
        fi
    else
        echo "No GCS credentials found. Falling back to local-only cache"
        echo "To enable GCS, either:"
        echo "  1. Run 'gcloud auth application-default login'"
        echo "  2. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
        return 1
    fi

    echo "sccache GCS config: bucket=$SCCACHE_GCS_BUCKET prefix=$SCCACHE_GCS_KEY_PREFIX mode=$SCCACHE_GCS_RW_MODE"
}

start_server() {
    "$SCCACHE_PATH" --stop-server &>/dev/null || true
    "$SCCACHE_PATH" --start-server
    echo "sccache server started"
}

# --- Main ---
install_sccache || return 1 2>/dev/null || exit 1

if [ "$USE_GCS" = true ]; then
    configure_gcs_cache || configure_local_cache
else
    configure_local_cache
fi

start_server

export SCCACHE_PATH
echo ""
echo "sccache is ready. CMake will auto-detect it on next configure."
echo "To reconfigure an existing build directory:"
echo "  cmake -B cmake-build-relwithdebinfo -DCMAKE_CXX_COMPILER_LAUNCHER=$SCCACHE_PATH"
echo ""
echo "View cache stats:   sccache --show-stats"
echo "Stop server:        sccache --stop-server"
