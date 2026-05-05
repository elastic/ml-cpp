#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
"""Create a GitHub pull request using the REST API (no gh CLI required).

Environment — one of:
  GITHUB_TOKEN, VAULT_GITHUB_TOKEN, GH_TOKEN

On success, prints the PR HTML URL to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo (e.g. elastic/ml-cpp)")
    parser.add_argument("--base", required=True, help="base branch name")
    parser.add_argument("--head", required=True, help="head branch name")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    args = parser.parse_args()

    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("VAULT_GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
    )
    if not token:
        print(
            "ERROR: Set GITHUB_TOKEN, VAULT_GITHUB_TOKEN, or GH_TOKEN",
            file=sys.stderr,
        )
        return 1

    url = f"https://api.github.com/repos/{args.repo}/pulls"
    payload = json.dumps(
        {
            "title": args.title,
            "head": args.head,
            "base": args.base,
            "body": args.body,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "ml-cpp-version-bump",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode())
            html_url = out.get("html_url", "")
            if html_url:
                print(html_url)
            return 0
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"ERROR: GitHub API HTTP {e.code}: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
