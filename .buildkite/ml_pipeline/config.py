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
import re

class Config:
    build_windows: bool = False
    build_macos: bool = False
    build_linux: bool = False
    build_aarch64: str = ""
    build_x86_64: str = ""
    run_qa_tests: bool = False
    action: str = "build"

    def parse_comment(self):
        if "GITHUB_PR_COMMENT_VAR_ACTION" in os.environ:
            self.action = os.environ["GITHUB_PR_COMMENT_VAR_ACTION"]
            self.run_qa_tests = self.action == "run_qa_tests"
            if self.run_qa_tests:
                self.action = "build"

        if "GITHUB_PR_COMMENT_VAR_ARCH" in os.environ and os.environ["GITHUB_PR_COMMENT_VAR_ARCH"] != "":
            csv_arch = os.environ["GITHUB_PR_COMMENT_VAR_ARCH"]
            for each in [ x.strip().lower() for x in csv_arch.split(",")]:
                if each == "aarch64":
                    self.build_aarch64 = "--build-aarch64"
                elif each == "x86_64":
                    self.build_x86_64 = "--build-x86_64"
        else:
            self.build_aarch64 = "--build-aarch64"
            self.build_x86_64 = "--build-x86_64"

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
        build_labels = ['ci:build-linux','ci:build-macos','ci:build-windows','ci:run-qa-tests','ci:build-aarch64','ci:build-x86_64']
        all_labels = [x.strip().lower() for x in os.environ["GITHUB_PR_LABELS"].split(",")]
        ci_labels = [label for label in all_labels if re.search("|".join(build_labels), label)]
        if not ci_labels:
            self.build_windows = True
            self.build_macos = True
            self.build_linux = True
            self.build_aarch64 = ""
            self.build_x86_64 = ""
            self.run_qa_tests = False
        else:
            for label in ci_labels:
                if "ci:build-aarch64" == label:
                    self.build_aarch64 = "--build-aarch64"
                if "ci:build-x86_64" == label:
                    self.build_x86_64 = "--build-x86_64"
                if "ci:build-windows" == label:
                    self.build_windows = True
                elif "ci:build-macos" == label:
                    self.build_macos = True
                elif "ci:build-linux" == label:
                    self.build_linux = True
                if "ci:run-qa-tests" == label:
                    self.build_windows = True
                    self.build_macos = True
                    self.build_linux = True
                    self.run_qa_tests = True

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
            self.build_aarch64 = "--build-aarch64"
            self.build_x86_64 = "--build-x86_64"
            self.run_qa_tests = False

