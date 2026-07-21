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
# Shared helpers for ml-cpp release-eng version bump scripts.

set -euo pipefail

version_bump_trim_value() {
    local s=$1
    s="${s//$'\r'/}"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

configure_git() {
    git config user.name elasticsearchmachine
    git config user.email 'infra-root+elasticsearchmachine@elastic.co'
}

sed_inplace() {
    local script=$1
    local target=$2
    local tmp
    tmp=$(mktemp "${target}.sedtmp.XXXXXX")
    if ! sed "${script}" "$target" >"$tmp"; then
        rm -f "$tmp"
        return 1
    fi
    mv "$tmp" "$target"
}

github_repo_slug() {
    local url
    url=$(git remote get-url origin 2>/dev/null || true)
    if [[ "$url" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        return 0
    fi
    echo "ERROR: could not parse owner/repo from git remote url: ${url:-empty}" >&2
    return 1
}

version_bump_set_buildkite_meta() {
    local key="$1"
    local value="$2"
    if [[ "${BUILDKITE:-}" != "true" ]]; then
        return 0
    fi
    if ! command -v buildkite-agent >/dev/null 2>&1; then
        echo "WARNING: BUILDKITE=true but buildkite-agent not in PATH; skipping meta-data ${key}=${value}" >&2
        return 0
    fi
    buildkite-agent meta-data set "$key" "$value"
}

version_bump_set_buildkite_meta_changed() {
    version_bump_set_buildkite_meta "ml_cpp_version_bump_changed" "$1"
}

version_bump_set_noop_meta() {
    local noop="$1"
    version_bump_set_buildkite_meta "ml_cpp_version_bump_noop" "$noop"
}

version_bump_set_pr_url_meta() {
    local url="${1:-}"
    if [[ -z "${url}" ]]; then
        return 0
    fi
    version_bump_set_buildkite_meta "ml_cpp_version_bump_pr_url" "$url"
}

read_elasticsearch_version_from_file() {
    local file=$1
    grep '^elasticsearchVersion=' "$file" | head -1 | cut -d= -f2 | tr -d '[:space:]' || true
}

read_elasticsearch_version_from_ref() {
    local ref=$1
    git show "${ref}:gradle.properties" | grep '^elasticsearchVersion=' | head -1 | cut -d= -f2 | tr -d '[:space:]' || true
}

# Copy named helper files from \p src_dir into a fresh temp directory and print
# that path.  Patch/minor bump pipelines check out the *target* release branch
# tip after starting from main; without a snapshot, later subprocesses would
# execute older copies of these helpers from the release branch (e.g. a
# create_github_pull_request.sh that does not accept --label).
#
# Usage: dest=$(version_bump_snapshot_helpers "$SCRIPT_DIR" file1 file2 ...)
version_bump_snapshot_helpers() {
    local src_dir="$1"
    shift
    if [[ $# -lt 1 ]]; then
        echo "ERROR: version_bump_snapshot_helpers requires at least one file name" >&2
        return 1
    fi
    local dest name
    dest="$(mktemp -d "${TMPDIR:-/tmp}/ml-cpp-version-bump-helpers.XXXXXX")"
    for name in "$@"; do
        if [[ ! -f "${src_dir}/${name}" ]]; then
            echo "ERROR: missing helper to snapshot: ${src_dir}/${name}" >&2
            rm -rf "${dest}"
            return 1
        fi
        # Handle cp/chmod failure explicitly: under `set -e` an aborted copy would
        # skip the caller's cleanup trap (not yet installed) and leak ${dest}.
        if ! cp "${src_dir}/${name}" "${dest}/${name}" || ! chmod +x "${dest}/${name}"; then
            echo "ERROR: failed to stage helper ${name} into ${dest}" >&2
            rm -rf "${dest}"
            return 1
        fi
    done
    printf '%s' "$dest"
}
