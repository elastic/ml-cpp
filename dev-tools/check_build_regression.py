#!/usr/bin/env python3
"""Check for build timing regressions in a PR build.

Compares step durations from the current build against a rolling baseline
computed from recent nightly builds in Elasticsearch.  Flags steps whose
duration exceeds the baseline mean + 2*stddev.

Usage:
    # Check a specific build
    python3 dev-tools/check_build_regression.py --pipeline ml-cpp-pr-builds --build 1234

    # In CI — uses BUILDKITE_PIPELINE_SLUG and BUILDKITE_BUILD_NUMBER
    python3 dev-tools/check_build_regression.py

Environment:
    BUILDKITE_TOKEN / BUILDKITE_API_READ_TOKEN   Buildkite API read token
    ES_ENDPOINT              Elasticsearch endpoint (or ~/.elastic/serverless-endpoint)
    ES_API_KEY               Elasticsearch API key (or ~/.elastic/serverless-api-key)
    BUILDKITE_PIPELINE_SLUG  Current pipeline slug (set by Buildkite)
    BUILDKITE_BUILD_NUMBER   Current build number (set by Buildkite)
"""

import argparse
import json
import math
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

BUILDKITE_ORG = "elastic"
INDEX_NAME = "buildkite-build-timings"
BASELINE_DAYS = 30
THRESHOLD_STDDEVS = 2.0
MIN_BASELINE_SAMPLES = 5


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


def es_query(endpoint, api_key, body):
    url = f"{endpoint}/{INDEX_NAME}/_search"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_baseline_stats(es_endpoint, es_api_key):
    """Get mean and stddev of duration per step_key from recent nightly builds."""
    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gte": f"now-{BASELINE_DAYS}d"}}},
                    {"terms": {"pipeline": [
                        "ml-cpp-snapshot-builds", "ml-cpp-debug-build"
                    ]}},
                    {"term": {"state": "passed"}},
                    {"terms": {"step_type": ["build_test", "build_cross", "java_it"]}},
                ]
            }
        },
        "aggs": {
            "by_step": {
                "terms": {"field": "step_key", "size": 50},
                "aggs": {
                    "duration_stats": {
                        "extended_stats": {"field": "duration_seconds"}
                    }
                }
            }
        }
    }
    result = es_query(es_endpoint, es_api_key, body)
    baselines = {}
    for bucket in result["aggregations"]["by_step"]["buckets"]:
        stats = bucket["duration_stats"]
        if stats["count"] >= MIN_BASELINE_SAMPLES:
            baselines[bucket["key"]] = {
                "mean": stats["avg"],
                "stddev": stats["std_deviation"],
                "min": stats["min"],
                "max": stats["max"],
                "count": stats["count"],
            }
    return baselines


def parse_ts(ts_str):
    if not ts_str:
        return None
    from datetime import datetime
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def get_build_step_durations(pipeline_slug, build_number, bk_token):
    """Get step durations from a specific build via Buildkite API."""
    build = buildkite_get(
        f"pipelines/{pipeline_slug}/builds/{build_number}", bk_token
    )
    steps = {}
    for job in build.get("jobs", []):
        if job.get("type") != "script":
            continue
        step_key = job.get("step_key")
        if not step_key:
            continue
        started = parse_ts(job.get("started_at"))
        finished = parse_ts(job.get("finished_at"))
        if not started or not finished:
            continue
        steps[step_key] = {
            "duration": (finished - started).total_seconds(),
            "state": job.get("state", ""),
            "label": job.get("name", ""),
        }
    return steps


def format_duration(seconds):
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    if seconds >= 60:
        return f"{seconds/60:.1f}m"
    return f"{seconds:.0f}s"


def main():
    parser = argparse.ArgumentParser(description="Check for build timing regressions")
    parser.add_argument("--pipeline", default=os.environ.get("BUILDKITE_PIPELINE_SLUG"))
    parser.add_argument("--build", type=int, default=int(os.environ.get("BUILDKITE_BUILD_NUMBER", "0")))
    parser.add_argument("--threshold", type=float, default=THRESHOLD_STDDEVS,
                        help="Number of standard deviations above mean to flag (default: 2.0)")
    parser.add_argument("--annotate", action="store_true",
                        help="Post results as a Buildkite annotation")
    args = parser.parse_args()

    if not args.pipeline or not args.build:
        print("Error: --pipeline and --build required (or set BUILDKITE_PIPELINE_SLUG/BUILDKITE_BUILD_NUMBER)",
              file=sys.stderr)
        sys.exit(1)

    bk_token = (get_env_or_file("BUILDKITE_TOKEN", "~/.buildkite/token")
                or get_env_or_file("BUILDKITE_API_READ_TOKEN", ""))
    es_endpoint = get_env_or_file("ES_ENDPOINT", "~/.elastic/serverless-endpoint")
    es_api_key = get_env_or_file("ES_API_KEY", "~/.elastic/serverless-api-key")

    if not bk_token:
        print("Error: No Buildkite token available", file=sys.stderr)
        sys.exit(1)
    if not es_endpoint or not es_api_key:
        print("Error: ES_ENDPOINT and ES_API_KEY required", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {args.pipeline} build #{args.build} against {BASELINE_DAYS}-day baseline...")
    print(f"Threshold: mean + {args.threshold}σ\n")

    baselines = get_baseline_stats(es_endpoint, es_api_key)
    if not baselines:
        print("Warning: no baseline data available, skipping regression check")
        sys.exit(0)

    steps = get_build_step_durations(args.pipeline, args.build, bk_token)
    if not steps:
        print("Warning: no step timings found in this build")
        sys.exit(0)

    regressions = []
    results_lines = []

    for step_key, step_data in sorted(steps.items()):
        if step_key not in baselines:
            continue
        if step_data["state"] != "passed":
            continue

        baseline = baselines[step_key]
        duration = step_data["duration"]
        threshold = baseline["mean"] + args.threshold * baseline["stddev"]
        deviation = ((duration - baseline["mean"]) / baseline["stddev"]
                     if baseline["stddev"] > 0 else 0)

        status = "✅"
        if duration > threshold:
            status = "🔴"
            regressions.append({
                "step_key": step_key,
                "duration": duration,
                "mean": baseline["mean"],
                "stddev": baseline["stddev"],
                "deviation": deviation,
                "label": step_data["label"],
            })
        elif duration > baseline["mean"] + baseline["stddev"]:
            status = "🟡"

        results_lines.append(
            f"  {status} {step_key}: {format_duration(duration)} "
            f"(baseline: {format_duration(baseline['mean'])} ± {format_duration(baseline['stddev'])}, "
            f"{deviation:+.1f}σ)"
        )

    print("Step timings vs baseline:")
    for line in results_lines:
        print(line)

    annotation_body = ""
    if regressions:
        print(f"\n⚠️  {len(regressions)} regression(s) detected:")
        annotation_body = f"### ⚠️ Build Timing Regressions ({len(regressions)} steps)\n\n"
        annotation_body += "| Step | Duration | Baseline | Deviation |\n"
        annotation_body += "|------|----------|----------|-----------|\n"
        for r in regressions:
            print(f"  🔴 {r['step_key']}: {format_duration(r['duration'])} "
                  f"(expected {format_duration(r['mean'])} ± {format_duration(r['stddev'])}, "
                  f"{r['deviation']:+.1f}σ)")
            annotation_body += (
                f"| {r['step_key']} | {format_duration(r['duration'])} "
                f"| {format_duration(r['mean'])} ± {format_duration(r['stddev'])} "
                f"| {r['deviation']:+.1f}σ |\n"
            )
    else:
        print("\n✅ No regressions detected")
        annotation_body = "### ✅ No build timing regressions detected\n"

    if args.annotate:
        import subprocess
        subprocess.run(
            ["buildkite-agent", "annotate", "--style",
             "warning" if regressions else "success",
             "--context", "build-timing-regression"],
            input=annotation_body.encode(),
            check=False,
        )

    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
