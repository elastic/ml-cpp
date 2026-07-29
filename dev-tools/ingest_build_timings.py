#!/usr/bin/env python3
"""Ingest Buildkite build step timings into Elasticsearch.

Fetches step-level timing data from ml-cpp Buildkite pipelines and indexes
it into the `buildkite-build-timings` index for anomaly detection and
regression tracking.

Usage:
    # Backfill last 30 builds from all nightly pipelines
    python3 dev-tools/ingest_build_timings.py --backfill 30

    # Ingest a specific build
    python3 dev-tools/ingest_build_timings.py --pipeline ml-cpp-snapshot-builds --build 5819

    # Ingest the latest build (for use as a post-build step)
    python3 dev-tools/ingest_build_timings.py --pipeline ml-cpp-snapshot-builds --latest

Environment:
    BUILDKITE_TOKEN          Buildkite API read token
    ES_ENDPOINT              Elasticsearch endpoint (or read from ~/.elastic/serverless-endpoint)
    ES_API_KEY               Elasticsearch API key (or read from ~/.elastic/serverless-api-key)
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

BUILDKITE_ORG = "elastic"
NIGHTLY_PIPELINES = ["ml-cpp-snapshot-builds", "ml-cpp-debug-build"]
INDEX_NAME = "buildkite-build-timings"

STEP_KEY_TO_PLATFORM = {
    "build_test_linux-x86_64": "linux-x86_64",
    "build_test_linux-aarch64": "linux-aarch64",
    "build_test_macos-aarch64": "macos-aarch64",
    "build_test_Windows-x86_64": "windows-x86_64",
    "build_macos_x86_64_cross": "macos-x86_64-cross",
    "java_integration_tests_linux-x86_64": "linux-x86_64",
    "java_integration_tests_linux-aarch64": "linux-aarch64",
    "check_style": "linux-x86_64",
    "clone_eigen": "linux-x86_64",
    "create_dra_artifacts": "linux-x86_64",
    "upload_dra_artifacts": "linux-x86_64",
    "upload_dra_artifacts_to_gcs": "linux-x86_64",
}


def get_env_or_file(env_var, file_path):
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    p = Path(file_path).expanduser()
    if p.exists():
        return p.read_text().strip()
    return None


def buildkite_get(path, token):
    url = f"https://api.buildkite.com/v2/organizations/{BUILDKITE_ORG}/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def parse_ts(ts_str):
    if not ts_str:
        return None
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def classify_step(step_key, label):
    if not step_key:
        return "infrastructure"
    if step_key.startswith("build_test_"):
        return "build_test"
    if step_key.startswith("build_macos"):
        return "build_cross"
    if step_key.startswith("java_integration_tests"):
        return "java_it"
    if step_key == "check_style":
        return "check_style"
    if step_key == "clone_eigen":
        return "dependency"
    if "dra" in step_key:
        return "release"
    if "analyze" in step_key:
        return "analytics"
    return "other"


def infer_platform(step_key):
    if not step_key:
        return "unknown"
    for prefix, platform in STEP_KEY_TO_PLATFORM.items():
        if step_key.startswith(prefix):
            return platform
    return "unknown"


def build_to_docs(build, pipeline_slug):
    docs = []
    for job in build.get("jobs", []):
        if job.get("type") != "script":
            continue
        step_key = job.get("step_key")
        if not step_key:
            continue

        created = parse_ts(job.get("created_at"))
        started = parse_ts(job.get("started_at"))
        finished = parse_ts(job.get("finished_at"))

        if not started or not finished:
            continue

        duration = (finished - started).total_seconds()
        agent_wait = (started - created).total_seconds() if created else None

        doc = {
            "@timestamp": job["finished_at"],
            "pipeline": pipeline_slug,
            "build_number": build["number"],
            "build_id": build["id"],
            "branch": build["branch"],
            "commit": build.get("commit", ""),
            "step_key": step_key,
            "step_label": job.get("name", ""),
            "platform": infer_platform(step_key),
            "step_type": classify_step(step_key, job.get("name", "")),
            "duration_seconds": round(duration, 1),
            "state": job.get("state", ""),
            "build_state": build.get("state", ""),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }
        if agent_wait is not None:
            doc["agent_wait_seconds"] = round(agent_wait, 1)

        docs.append(doc)

    return docs


def bulk_index(docs, es_endpoint, es_api_key):
    if not docs:
        return 0

    body_lines = []
    for doc in docs:
        doc_id = f"{doc['pipeline']}-{doc['build_number']}-{doc['step_key']}"
        body_lines.append(json.dumps({"index": {"_index": INDEX_NAME, "_id": doc_id}}))
        body_lines.append(json.dumps(doc))
    body = "\n".join(body_lines) + "\n"

    url = f"{es_endpoint}/_bulk"
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={
            "Authorization": f"ApiKey {es_api_key}",
            "Content-Type": "application/x-ndjson",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    errors = sum(1 for item in result.get("items", []) if item.get("index", {}).get("error"))
    indexed = len(result.get("items", [])) - errors
    if errors:
        for item in result["items"]:
            err = item.get("index", {}).get("error")
            if err:
                print(f"  Error: {err}", file=sys.stderr)
                break
    return indexed


def fetch_builds(pipeline_slug, token, count=30):
    builds = []
    page = 1
    while len(builds) < count:
        per_page = min(count - len(builds), 100)
        data = buildkite_get(
            f"pipelines/{pipeline_slug}/builds?per_page={per_page}&page={page}",
            token,
        )
        if not data:
            break
        builds.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return builds[:count]


def main():
    parser = argparse.ArgumentParser(description="Ingest Buildkite build timings into Elasticsearch")
    parser.add_argument("--pipeline", help="Pipeline slug (default: all nightly pipelines)")
    parser.add_argument("--build", type=int, help="Specific build number to ingest")
    parser.add_argument("--latest", action="store_true", help="Ingest only the latest build")
    parser.add_argument("--backfill", type=int, metavar="N", help="Backfill last N builds per pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print docs without indexing")
    args = parser.parse_args()

    bk_token = (get_env_or_file("BUILDKITE_TOKEN", "~/.buildkite/token")
                or get_env_or_file("BUILDKITE_API_READ_TOKEN", ""))
    es_endpoint = get_env_or_file("ES_ENDPOINT", "~/.elastic/serverless-endpoint")
    es_api_key = get_env_or_file("ES_API_KEY", "~/.elastic/serverless-api-key")

    if not bk_token:
        print("Error: BUILDKITE_TOKEN not set and ~/.buildkite/token not found", file=sys.stderr)
        sys.exit(1)
    if not es_endpoint and not args.dry_run:
        print("Error: ES_ENDPOINT not set and ~/.elastic/serverless-endpoint not found", file=sys.stderr)
        sys.exit(1)
    if not es_api_key and not args.dry_run:
        print("Error: ES_API_KEY not set and ~/.elastic/serverless-api-key not found", file=sys.stderr)
        sys.exit(1)

    pipelines = [args.pipeline] if args.pipeline else NIGHTLY_PIPELINES

    total_indexed = 0
    for pipeline_slug in pipelines:
        print(f"--- {pipeline_slug} ---")

        if args.build:
            builds = [buildkite_get(f"pipelines/{pipeline_slug}/builds/{args.build}", bk_token)]
        elif args.latest:
            builds = fetch_builds(pipeline_slug, bk_token, count=1)
        elif args.backfill:
            builds = fetch_builds(pipeline_slug, bk_token, count=args.backfill)
        else:
            builds = fetch_builds(pipeline_slug, bk_token, count=1)

        for build in builds:
            docs = build_to_docs(build, pipeline_slug)
            if not docs:
                print(f"  Build #{build['number']}: no step timings found")
                continue

            if args.dry_run:
                print(f"  Build #{build['number']} ({build['branch']}): {len(docs)} steps")
                for doc in docs:
                    print(f"    {doc['step_key']}: {doc['duration_seconds']}s ({doc['state']})")
            else:
                indexed = bulk_index(docs, es_endpoint, es_api_key)
                total_indexed += indexed
                print(f"  Build #{build['number']} ({build['branch']}): indexed {indexed}/{len(docs)} steps")

    if not args.dry_run:
        print(f"\nTotal indexed: {total_indexed} documents")


if __name__ == "__main__":
    main()
