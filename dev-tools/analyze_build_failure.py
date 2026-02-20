#!/usr/bin/env python3
"""Analyze a Buildkite build failure using Claude and post a diagnosis.

Fetches logs from failed build steps, sends them to the Anthropic Claude API
with repository context, and posts the analysis as a Buildkite annotation.

Usage:
    # Analyze the current build (in CI)
    python3 dev-tools/analyze_build_failure.py

    # Analyze a specific build
    python3 dev-tools/analyze_build_failure.py --pipeline ml-cpp-snapshot-builds --build 5819

    # Dry run (print to stdout, don't annotate)
    python3 dev-tools/analyze_build_failure.py --pipeline ml-cpp-snapshot-builds --build 5819 --dry-run

Environment:
    BUILDKITE_TOKEN / BUILDKITE_API_READ_TOKEN   Buildkite API token
    ANTHROPIC_API_KEY                             Claude API key
    BUILDKITE_PIPELINE_SLUG                       Current pipeline (set by Buildkite)
    BUILDKITE_BUILD_NUMBER                        Current build number (set by Buildkite)
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

BUILDKITE_ORG = "elastic"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_LOG_CHARS = 30000
MAX_RESPONSE_TOKENS = 2048

KNOWN_FAILURE_PATTERNS = """
Known transient/infrastructure failures:
- "Unable to download toolchain" / Adoptium JDK download failure: transient, retry usually fixes it
- "Exceeded maximum artifact size limit of 10 GiB": artifact_paths glob is too broad
- "sccache: error: couldn't connect to server": sccache server not running, check setup_sccache.sh
- CKMostCorrelatedTest/testScale timeout: CPU contention on low-core machines, check test parallelism
- CMultiFileDataAdderTest collision: test isolation bug with temp file naming

Known compilation patterns:
- "redefinition of" in unity builds: file needs SKIP_UNITY_BUILD_INCLUSION or unity disabled for library
- boost/unordered_map.hpp conflicts: remove from PCH list
- "mspdbsrv.exe" errors on Windows: switch from /Zi to /Z7
"""

SYSTEM_PROMPT = """You are a CI build failure analyst for the elastic/ml-cpp repository.
This is a C++ codebase that builds on Linux (x86_64, aarch64), macOS (aarch64), and Windows (x86_64).
Build system: CMake with Boost, uses Docker for Linux builds, Gradle for macOS/Windows, Buildkite for CI.

Your job is to:
1. Identify the root cause of the failure from the build log
2. Classify it as: code bug, test failure, infrastructure/transient, configuration issue, or dependency issue
3. Suggest a specific fix or workaround
4. If it's transient, say so clearly — don't over-diagnose

Be concise and actionable. Use markdown formatting.
Format your response as:

### Root Cause
<1-2 sentences>

### Classification
<one of: code bug | test failure | infrastructure/transient | configuration | dependency>

### Suggested Fix
<specific actionable steps>

### Confidence
<high | medium | low> — <brief justification>
"""


def get_env_or_file(env_var, file_path):
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    if file_path:
        p = Path(file_path).expanduser()
        if p.exists():
            return p.read_text().strip()
    return None


def buildkite_get(path, token):
    url = f"https://api.buildkite.com/v2/organizations/{BUILDKITE_ORG}/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_job_log(log_url, token):
    """Fetch the raw log for a Buildkite job."""
    req = urllib.request.Request(
        log_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "text/plain",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError:
        return None


def truncate_log(log_text, max_chars=MAX_LOG_CHARS):
    """Keep the last max_chars of the log (the end usually has the error)."""
    if not log_text or len(log_text) <= max_chars:
        return log_text
    return f"... [truncated {len(log_text) - max_chars} chars] ...\n" + log_text[-max_chars:]


def call_claude(api_key, prompt):
    body = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    for block in result.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return "No analysis generated."


def main():
    parser = argparse.ArgumentParser(description="Analyze Buildkite build failures with Claude")
    parser.add_argument("--pipeline", default=os.environ.get("BUILDKITE_PIPELINE_SLUG"))
    parser.add_argument("--build", type=int, default=int(os.environ.get("BUILDKITE_BUILD_NUMBER", "0")))
    parser.add_argument("--dry-run", action="store_true", help="Print analysis without annotating")
    args = parser.parse_args()

    if not args.pipeline or not args.build:
        print("Error: --pipeline and --build required", file=sys.stderr)
        sys.exit(1)

    bk_token = (get_env_or_file("BUILDKITE_TOKEN", "~/.buildkite/token")
                or get_env_or_file("BUILDKITE_API_READ_TOKEN", ""))
    claude_key = get_env_or_file("ANTHROPIC_API_KEY", "~/.elastic/claude_api_key")

    if not bk_token:
        print("Error: No Buildkite token available", file=sys.stderr)
        sys.exit(1)
    if not claude_key:
        print("Error: No Anthropic API key available", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {args.pipeline} build #{args.build}...")

    build = buildkite_get(f"pipelines/{args.pipeline}/builds/{args.build}", bk_token)

    if build.get("state") == "passed":
        print("Build passed — nothing to analyze.")
        sys.exit(0)

    failed_jobs = [
        j for j in build.get("jobs", [])
        if j.get("type") == "script" and j.get("state") == "failed"
    ]

    if not failed_jobs:
        print("No failed steps found.")
        sys.exit(0)

    print(f"Found {len(failed_jobs)} failed step(s)")

    all_analyses = []

    for job in failed_jobs:
        step_key = job.get("step_key", "unknown")
        step_label = job.get("name", step_key)
        raw_log_url = job.get("raw_log_url", "")

        print(f"\nAnalyzing: {step_label} ({step_key})")

        log_text = get_job_log(raw_log_url, bk_token) if raw_log_url else None
        if not log_text:
            print(f"  Could not fetch log, skipping")
            continue

        log_excerpt = truncate_log(log_text)

        prompt = f"""Analyze this CI build failure.

**Pipeline**: {args.pipeline}
**Build**: #{args.build}
**Branch**: {build.get('branch', 'unknown')}
**Failed step**: {step_label} (key: {step_key})

{KNOWN_FAILURE_PATTERNS}

**Build log (last {MAX_LOG_CHARS} chars)**:
```
{log_excerpt}
```

Analyze the root cause and suggest a fix."""

        try:
            analysis = call_claude(claude_key, prompt)
        except Exception as e:
            analysis = f"Failed to get analysis: {e}"

        print(f"\n{analysis}")
        all_analyses.append(f"## {step_label}\n\n{analysis}")

    if not all_analyses:
        print("No analyses generated.")
        sys.exit(0)

    full_annotation = f"# 🔍 Build Failure Analysis\n\n"
    full_annotation += f"*Pipeline*: `{args.pipeline}` | *Build*: #{args.build} | *Branch*: `{build.get('branch', '?')}`\n\n"
    full_annotation += "\n\n---\n\n".join(all_analyses)
    full_annotation += "\n\n---\n*Analysis generated by Claude. Verify before acting.*"

    if not args.dry_run:
        try:
            subprocess.run(
                ["buildkite-agent", "annotate",
                 "--style", "error",
                 "--context", "build-failure-analysis"],
                input=full_annotation.encode(),
                check=True,
            )
            print("\nAnnotation posted to Buildkite.")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"\nCould not post annotation: {e}", file=sys.stderr)
            print("Full analysis printed above.")


if __name__ == "__main__":
    main()
