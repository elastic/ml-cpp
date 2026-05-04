#!/usr/bin/env python3
"""Analyze a Buildkite build failure using Claude and post a diagnosis.

Fetches logs from failed build steps, sends them to the Anthropic Claude API
with repository context, and posts the analysis as a Buildkite annotation,
Buildkite build metadata (for the GitHub Actions PR comment workflow),
and optionally to Slack.

Usage:
    # Analyze the current build (in CI)
    python3 dev-tools/analyze_build_failure.py

    # Analyze a specific build
    python3 dev-tools/analyze_build_failure.py --pipeline ml-cpp-snapshot-builds --build 5819

    # Find and analyze the most recent failed build for the current branch
    # (used by "buildkite analyze" PR comment — no rebuild needed)
    python3 dev-tools/analyze_build_failure.py --find-previous-failure

    # Dry run (print to stdout, don't annotate or post to Slack/GitHub)
    python3 dev-tools/analyze_build_failure.py --pipeline ml-cpp-snapshot-builds --build 5819 --dry-run

Environment:
    BUILDKITE_TOKEN / BUILDKITE_API_READ_TOKEN   Buildkite API token
    ANTHROPIC_API_KEY                             Claude API key
    SLACK_WEBHOOK_URL                             Slack incoming webhook (optional)
    BUILDKITE_PIPELINE_SLUG                       Current pipeline (set by Buildkite)
    BUILDKITE_BUILD_NUMBER                        Current build number (set by Buildkite)
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

BUILDKITE_ORG = "elastic"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_LOG_CHARS = 30000
MAX_RESPONSE_TOKENS = 2048
BK_HTTP_TIMEOUT_SEC = 45
BK_LOG_FETCH_TIMEOUT_SEC = 120

FAILED_JOB_STATES = frozenset({"failed", "timed_out"})

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
    """GET a JSON resource from the Buildkite REST API."""
    url = f"https://api.buildkite.com/v2/organizations/{BUILDKITE_ORG}/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=BK_HTTP_TIMEOUT_SEC) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"Buildkite HTTP {e.code} for {path}: {body[:500]}") from e
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
        raise RuntimeError(f"Buildkite request failed for {path}: {e}") from e


def find_previous_failed_build(pipeline, token, branch=None, exclude_build=None):
    """Find the most recent failed build for a pipeline, optionally filtered by branch."""
    params = {"state": "failed", "per_page": "5"}
    if branch:
        params["branch"] = branch
    query = urllib.parse.urlencode(params)
    try:
        builds = buildkite_get(f"pipelines/{pipeline}/builds?{query}", token)
    except RuntimeError:
        return None
    for build in builds:
        if exclude_build and build.get("number") == exclude_build:
            continue
        return build
    return None


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
        with urllib.request.urlopen(req, timeout=BK_LOG_FETCH_TIMEOUT_SEC) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        return None


ERROR_PATTERNS = re.compile(
    r"(?i)"
    r"(?:^|\s)error(?:\s|:|\[|C\d)"    # "error:", "error C2338", "error[E"
    r"|fatal error"
    r"|^#error\b"
    r"|FAILED"
    r"|\*\*\* \d+ failure"              # Boost.Test: *** N failure(s) detected
    r"|: fatal:"                         # linker fatal
    r"|ninja: build stopped"
    r"|make.*\*\*\*"                     # make: *** [target] Error
    r"|CMake Error"
    r"|assertion failed"
    r"|LINK : fatal"                     # MSVC linker
    r"|unresolved external"
    r"|cannot find -l"                   # linker: cannot find library
    r"|undefined reference"
    r"|Segmentation fault"
    r"|signal \d+"
    r"|exit code \d+"
    r"|Exit status: \d+(?!.*exit code 0)"
)

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")
BK_TIMESTAMP = re.compile(r"_bk;t=\d+")

def redact_secrets(text):
    """Best-effort removal of credentials from CI log excerpts before external APIs."""
    if not text:
        return text
    out = text
    out = re.sub(r"(?i)(authorization:\s*bearer\s+)\S+", r"\1<redacted>", out)
    out = re.sub(r"(?i)(x-api-key:\s*)\S+", r"\1<redacted>", out)
    out = re.sub(r"(?i)(x-anthropic-api-key:\s*)\S+", r"\1<redacted>", out)
    out = re.sub(r"\bsk-ant-api\d{3}-[\w-]{20,}\b", "<redacted>", out)
    out = re.sub(r"\bghp_[A-Za-z0-9]{36,}\b", "<redacted>", out)
    out = re.sub(r"\bgho_[A-Za-z0-9]{36}\b", "<redacted>", out)
    out = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "<redacted>", out)
    out = re.sub(r"\bxox[baprs]-[A-Za-z0-9-]+\b", "<redacted>", out)
    out = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "<redacted>", out)
    out = re.sub(r"\bAROA[0-9A-Z]{16}\b", "<redacted>", out)
    out = re.sub(r"(?i)(aws_secret_access_key|aws_session_token)\s*=\s*\S+", r"\1=<redacted>", out)
    out = re.sub(r"(?i)(password|secret|token|apikey)\s*=\s*\S+", r"\1=<redacted>", out)
    out = re.sub(r"https?://[^:]+:[^@\s]+@", "https://<redacted>:<redacted>@", out)
    out = re.sub(
        r"(?is)(-----BEGIN [A-Z ]*PRIVATE KEY-----)(.*?)(-----END [A-Z ]*PRIVATE KEY-----)",
        r"\1[REDACTED PRIVATE KEY]\3",
        out,
    )
    return out


def strip_terminal_noise(log_text):
    """Remove ANSI escapes and Buildkite timestamp markers."""
    text = ANSI_ESCAPE.sub("", log_text)
    return BK_TIMESTAMP.sub("", text)


def extract_error_context(log_text, context_lines=10, max_chars=MAX_LOG_CHARS):
    """Extract error-relevant sections from a build log.

    Scans every line for error patterns and collects matching lines with
    surrounding context.  Always appends the tail of the log (which
    typically contains the build summary / exit code).  The combined
    output is capped at *max_chars*.
    """
    if not log_text:
        return log_text

    log_text = strip_terminal_noise(log_text)
    log_text = redact_secrets(log_text)
    lines = log_text.splitlines()

    if len(log_text) <= max_chars:
        return log_text

    # Find line indices that match error patterns.
    error_indices = set()
    for i, line in enumerate(lines):
        if ERROR_PATTERNS.search(line):
            error_indices.add(i)

    # Expand each match with context_lines before/after, merging overlaps.
    include = set()
    for idx in sorted(error_indices):
        for j in range(max(0, idx - context_lines), min(len(lines), idx + context_lines + 1)):
            include.add(j)

    # Always include the last 80 lines (build summary / exit info).
    tail_start = max(0, len(lines) - 80)
    for j in range(tail_start, len(lines)):
        include.add(j)

    # Build the excerpt, inserting "..." markers for skipped regions.
    sections = []
    prev = -2
    for i in sorted(include):
        if i != prev + 1:
            sections.append("... [skipped] ...")
        sections.append(lines[i])
        prev = i

    excerpt = "\n".join(sections)

    # Final safety cap — if still too long, keep the head and tail.
    if len(excerpt) > max_chars:
        half = max_chars // 2
        excerpt = (excerpt[:half]
                   + f"\n... [trimmed {len(excerpt) - max_chars} chars] ...\n"
                   + excerpt[-half:])

    return excerpt


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


def post_to_slack(webhook_url, pipeline, build_number, branch, build_url, analyses):
    """Post a summary of the failure analysis to Slack."""
    # Slack uses mrkdwn, not full markdown — convert minimally
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Build Failure Analysis",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Pipeline:* `{pipeline}` | *Build:* <{build_url}|#{build_number}> | *Branch:* `{branch}`"
                ),
            },
        },
    ]

    for step_label, analysis in analyses:
        # Extract just the classification and root cause for a compact Slack message
        lines = analysis.split("\n")
        root_cause = ""
        classification = ""
        for i, line in enumerate(lines):
            if line.startswith("### Root Cause"):
                root_cause = lines[i + 1].strip() if i + 1 < len(lines) else ""
            elif line.startswith("### Classification"):
                classification = lines[i + 1].strip() if i + 1 < len(lines) else ""

        emoji = {
            "infrastructure/transient": ":cloud:",
            "code bug": ":bug:",
            "test failure": ":test_tube:",
            "configuration": ":gear:",
            "dependency": ":package:",
        }.get(classification, ":warning:")

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji} *{step_label}*\n>{root_cause}\n_Classification: {classification}_",
            },
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"<{build_url}|View build> | Analysis by Claude — verify before acting",
            }
        ],
    })

    payload = json.dumps({"blocks": blocks}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print("Slack notification posted.")
            else:
                print(f"Slack returned status {resp.status}", file=sys.stderr)
    except Exception as e:
        print(f"Could not post to Slack: {e}", file=sys.stderr)



def main():
    parser = argparse.ArgumentParser(description="Analyze Buildkite build failures with Claude")
    parser.add_argument("--pipeline", default=os.environ.get("BUILDKITE_PIPELINE_SLUG"))
    parser.add_argument("--build", type=int, default=int(os.environ.get("BUILDKITE_BUILD_NUMBER", "0")))
    parser.add_argument("--find-previous-failure", action="store_true",
                        help="Find and analyze the most recent failed build for the current branch")
    parser.add_argument("--dry-run", action="store_true", help="Print analysis without annotating or posting to Slack")
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

    if args.find_previous_failure:
        branch = os.environ.get("BUILDKITE_BRANCH")
        print(f"Searching for previous failed build on branch '{branch}'...")
        prev = find_previous_failed_build(args.pipeline, bk_token, branch, args.build)
        if not prev:
            print(f"No previous failed build found for branch '{branch}' — nothing to analyze.")
            sys.exit(0)
        args.build = prev["number"]
        print(f"Found failed build #{args.build}: {prev.get('web_url', '')}")

    print(f"Analyzing {args.pipeline} build #{args.build}...")

    try:
        build = buildkite_get(f"pipelines/{args.pipeline}/builds/{args.build}", bk_token)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print("Could not fetch build from Buildkite — exiting without analysis.", file=sys.stderr)
        sys.exit(0)

    if build.get("state") == "passed":
        print("Build passed — nothing to analyze.")
        sys.exit(0)

    failed_jobs = [
        j for j in build.get("jobs", [])
        if j.get("type") == "script" and j.get("state") in FAILED_JOB_STATES
    ]

    if not failed_jobs:
        print("No failed steps found.")
        sys.exit(0)

    print(f"Found {len(failed_jobs)} failed step(s)")

    slack_webhook = get_env_or_file("SLACK_WEBHOOK_URL", "")
    build_url = build.get("web_url", f"https://buildkite.com/{BUILDKITE_ORG}/{args.pipeline}/builds/{args.build}")

    all_analyses = []
    slack_analyses = []

    for job in failed_jobs:
        step_key = job.get("step_key", "unknown")
        step_label = job.get("name", step_key)
        raw_log_url = job.get("raw_log_url", "")

        print(f"\nAnalyzing: {step_label} ({step_key})")

        log_text = get_job_log(raw_log_url, bk_token) if raw_log_url else None
        if not log_text:
            print(f"  Could not fetch log, skipping")
            continue

        log_excerpt = extract_error_context(log_text)

        if args.dry_run:
            analysis = (
                "_Dry run:_ Claude API not called. Log excerpt (redacted) below.\n\n"
                f"```\n{log_excerpt}\n```"
            )
        elif not claude_key:
            analysis = (
                "*Analysis skipped:* `ANTHROPIC_API_KEY` is not set. "
                "Configure the key on the Buildkite pipeline to enable Claude failure analysis."
            )
        else:
            prompt = f"""Analyze this CI build failure.

**Pipeline**: {args.pipeline}
**Build**: #{args.build}
**Branch**: {build.get('branch', 'unknown')}
**Failed step**: {step_label} (key: {step_key})

{KNOWN_FAILURE_PATTERNS}

**Build log (error-relevant sections extracted from full log)**:
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
        slack_analyses.append((step_label, analysis))

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

        # Store analysis as build metadata so that the GitHub Actions
        # workflow (post-build-analysis.yml) can fetch it and post a
        # PR comment using the built-in GITHUB_TOKEN.
        annotation_body = "\n\n---\n\n".join(all_analyses)
        try:
            # buildkite-agent accepts the value as argv or via stdin (see
            # `buildkite-agent meta-data set --help`). Pass bytes on stdin so
            # large analyses need no temp file.
            subprocess.run(
                ["buildkite-agent", "meta-data", "set", "build-failure-analysis"],
                input=annotation_body.encode("utf-8"),
                check=True,
            )
            print("Analysis saved as build metadata.")
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            print(f"Could not save build metadata: {e}", file=sys.stderr)

        if slack_webhook:
            post_to_slack(
                slack_webhook, args.pipeline, args.build,
                build.get("branch", "?"), build_url, slack_analyses,
            )
        else:
            print("No SLACK_WEBHOOK_URL set, skipping Slack notification.")


if __name__ == "__main__":
    main()
