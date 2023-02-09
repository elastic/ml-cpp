#!/usr/bin/env python
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import os

class Config:
    build_windows: bool = False
    build_macos: bool = False
    build_linux: bool = False
    action: str = "build"
    snapshot: str = "true"
    candidate: str = None

    def parse_comment(self):
        if "GITHUB_PR_COMMENT_VAR_ACTION" in os.environ:
            self.action = os.environ["GITHUB_PR_COMMENT_VAR_ACTION"]

        if "GITHUB_PR_COMMENT_VAR_SNAPSHOT" in os.environ:
            self.snapshot = os.environ["GITHUB_PR_COMMENT_VAR_SNAPSHOT"]

        if "GITHUB_PR_COMMENT_VAR_CANDIDATE" in os.environ:
            self.candidate = os.environ["GITHUB_PR_COMMENT_VAR_CANDIDATE"]

        if "GITHUB_PR_COMMENT_VAR_PLATFORM" in os.environ:
            csv_platform = os.environ["GITHUB_PR_COMMENT_VAR_PLATFORM"]
            for each in [ x.strip().lower() for x in csv_platform.split(",")]:
                if each == "windows":
                    self.build_windows = True
                elif each == "macos" or each == "mac":
                    self.build_macos = True
                elif each == "linux":
                    self.build_linux = True
        else:
            self.build_windows = True
            self.build_macos = True
            self.build_linux = True

    def parse_label(self):
        for label in [x.strip().lower() for x in os.environ["GITHUB_PR_LABELS"].split(",")]:
            if "ci:build-windows" == label:
                self.build_windows = True
            elif "ci:build-macos" == label:
                self.build_macos = True
            elif "ci:build-linux" == label:
                self.build_linux = True
            elif "ci:test-all" == label:
                self.build_windows = True
                self.build_macos = True
                self.build_linux = True

    def parse(self):
        """Parse Github label or Github comment passed through buildkite-pr-bot."""

        if "GITHUB_PR_TRIGGER_COMMENT" in os.environ:
            self.parse_comment()
        elif "GITHUB_PR_LABELS" in os.environ:
            self.parse_label()
        else:
            self.build_windows = True
            self.build_macos = True
            self.build_linux = True

