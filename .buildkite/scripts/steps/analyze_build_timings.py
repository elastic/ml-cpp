#!/usr/bin/env python3
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""
Analyse build+test timings for the current snapshot build and compare
against recent history.  Produces a Buildkite annotation with a summary
table and flags any regressions.
"""

import json
import math
import os
import subprocess
import sys
import urllib.request
import urllib.error

PIPELINE_SLUG = "ml-cpp-snapshot-builds"
ORG_SLUG = "elastic"
API_BASE = f"https://api.buildkite.com/v2/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}"
HISTORY_COUNT = 14

PLATFORM_MAP = {
    "Windows": "windows_x86_64",
    "MacOS": "macos_aarch64",
    "linux-x86_64": "linux_x86_64",
    "linux-aarch64": "linux_aarch64",
}


def api_get(path, token):
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"API error {e.code} for {url}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)


def extract_timings(build_data):
    """Extract per-platform build+test timings from a build's jobs."""
    timings = {}
    for job in build_data.get("jobs", []):
        name = job.get("name") or ""
        if "Build & test" not in name:
            continue
        if "debug" in name.lower():
            continue
        started = job.get("started_at")
        finished = job.get("finished_at")
        if not started or not finished:
            continue

        for pattern, key in PLATFORM_MAP.items():
            if pattern in name:
                from datetime import datetime, timezone
                fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                t_start = datetime.strptime(started, fmt).replace(tzinfo=timezone.utc)
                t_end = datetime.strptime(finished, fmt).replace(tzinfo=timezone.utc)
                mins = (t_end - t_start).total_seconds() / 60.0
                timings[key] = round(mins, 1)
                break
    return timings


def mean_stddev(values):
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(variance)


def annotate(markdown, style="info"):
    """Create a Buildkite annotation."""
    cmd = ["buildkite-agent", "annotate", "--style", style, "--context", "build-timings"]
    proc = subprocess.run(cmd, input=markdown.encode(), capture_output=True)
    if proc.returncode != 0:
        print(f"buildkite-agent annotate failed: {proc.stderr.decode()}", file=sys.stderr)


def main():
    token = os.environ.get("BUILDKITE_API_READ_TOKEN", "")
    if not token:
        print("BUILDKITE_API_READ_TOKEN not set, skipping timing analysis", file=sys.stderr)
        sys.exit(0)

    build_number = os.environ.get("BUILDKITE_BUILD_NUMBER", "")
    branch = os.environ.get("BUILDKITE_BRANCH", "main")

    # Fetch current build
    current = api_get(f"/builds/{build_number}", token)
    current_timings = extract_timings(current)
    current_date = current.get("created_at", "")[:10]

    if not current_timings:
        print("No build+test timings found for current build")
        sys.exit(0)

    # Fetch historical builds for the same branch
    history_data = api_get(
        f"/builds?branch={branch}&state=passed&per_page={HISTORY_COUNT + 1}", token
    )

    # Exclude the current build from history
    history_builds = [
        b for b in history_data if str(b.get("number")) != str(build_number)
    ][:HISTORY_COUNT]

    # Collect historical timings per platform
    history = {key: [] for key in PLATFORM_MAP.values()}
    for build in history_builds:
        full_build = api_get(f"/builds/{build['number']}", token)
        timings = extract_timings(full_build)
        for key, val in timings.items():
            history[key].append(val)

    # Build the summary table
    platforms = ["linux_x86_64", "linux_aarch64", "macos_aarch64", "windows_x86_64"]
    platform_labels = {
        "linux_x86_64": "Linux x86_64",
        "linux_aarch64": "Linux aarch64",
        "macos_aarch64": "macOS aarch64",
        "windows_x86_64": "Windows x86_64",
    }

    lines = []
    lines.append(f"### Build Timing Analysis — {current_date} (build #{build_number})")
    lines.append("")
    lines.append("| Platform | Current (min) | Avg (min) | Std Dev | Delta | Status |")
    lines.append("|----------|:------------:|:---------:|:-------:|:-----:|:------:|")

    has_regression = False
    for plat in platforms:
        cur = current_timings.get(plat)
        hist = history.get(plat, [])
        avg, sd = mean_stddev(hist)

        if cur is None:
            lines.append(f"| {platform_labels[plat]} | — | {avg:.1f} | {sd:.1f} | — | — |")
            continue

        delta = cur - avg
        delta_pct = (delta / avg * 100) if avg > 0 else 0
        sign = "+" if delta >= 0 else ""

        if avg > 0 and sd > 0 and cur > avg + 2 * sd:
            status = ":rotating_light: Regression"
            has_regression = True
        elif avg > 0 and cur < avg - sd:
            status = ":rocket: Faster"
        else:
            status = ":white_check_mark: Normal"

        lines.append(
            f"| {platform_labels[plat]} | **{cur:.1f}** | {avg:.1f} | {sd:.1f} "
            f"| {sign}{delta:.1f} ({sign}{delta_pct:.0f}%) | {status} |"
        )

    n_hist = len(history_builds)
    lines.append("")
    lines.append(f"_Compared against {n_hist} recent `{branch}` builds._")

    markdown = "\n".join(lines)
    print(markdown)

    style = "warning" if has_regression else "info"
    annotate(markdown, style)


if __name__ == "__main__":
    main()
