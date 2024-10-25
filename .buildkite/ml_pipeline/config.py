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
    run_pytorch_tests: bool = False
    action: str = "build"

    def parse_comment(self):
        """ Parse environment variables set from GitHub PR comments

        The GITHUB_PR_COMMENT_VAR_* environment variables can be populated from a matching group in the
        trigger_comment_regex defined in ".buildkite/pull-requests.json" (which parses comment fields on a GitHub PR)
        or they can be set manually, say from a trigger step in a Buildkite job, or from the environment variables
        passed to a job triggered manually on the ml-cpp PR pipeline from the Buildkite UI.
        """

        # If the GITHUB_PR_COMMENT_VAR_ACTION environment variable is set attemot to use its value to determine what
        # build action to take - one of build, debug, run_qa_tests, run_pytorch_tests. If either of the "run tests"
        # actions are set then this also implies "build"
        if "GITHUB_PR_COMMENT_VAR_ACTION" in os.environ:
            self.action = os.environ["GITHUB_PR_COMMENT_VAR_ACTION"]
            self.run_qa_tests = self.action == "run_qa_tests"
            self.run_pytorch_tests = self.action == "run_pytorch_tests"
            if self.run_pytorch_tests or self.run_qa_tests:
                self.action = "build"

        # If the ACTION is set to "run_qa_tests" then set some optional variables governing the ES branch to build, the
        # stack version to set and the subset of QA tests to run, depending on whether appropriate variables are set in
        # the environment.
        if self.run_qa_tests:
            if "GITHUB_PR_COMMENT_VAR_BRANCH" in os.environ:
                os.environ["ES_BRANCH"] = os.environ["GITHUB_PR_COMMENT_VAR_BRANCH"]

            if "GITHUB_PR_COMMENT_VAR_VERSION" in os.environ:
                os.environ["STACK_VERSION"] = os.environ["GITHUB_PR_COMMENT_VAR_VERSION"]

            if "GITHUB_PR_COMMENT_VAR_ARGS" in os.environ:
                os.environ["QAF_TESTS_TO_RUN"] = os.environ["GITHUB_PR_COMMENT_VAR_ARGS"]

        # If the GITHUB_PR_COMMENT_VAR_ARCH environment variable is set then   attemot to parse it
        # into comma separated values. If the values are one or both of "aarch64" or "x86_64" then set the member
        # variables self.build_aarch64, self.build_x86_64 accordingly. These values will be used to restrict the build
        # jobs to a particular achitecture.
        if "GITHUB_PR_COMMENT_VAR_ARCH" in os.environ:
            csv_arch = os.environ["GITHUB_PR_COMMENT_VAR_ARCH"]
            for each in [ x.strip().lower() for x in csv_arch.split(",")]:
                if each == "aarch64":
                    self.build_aarch64 = "--build-aarch64"
                elif each == "x86_64":
                    self.build_x86_64 = "--build-x86_64"
        else:
            self.build_aarch64 = "--build-aarch64"
            self.build_x86_64 = "--build-x86_64"

        # If the GITHUB_PR_COMMENT_VAR_PLATFORM environment variable is set to a non-empty string then attemot to parse it
        # into comma separated values. If the values are one or a combination of "windows", "mac(os)", "linux"  then set the member
        # variables self.build_windows, self.build_macos, self.build_linux accordingly. These values will be used to restrict the build
        # jobs to a particular platform.
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
        """ Parse labels set on GitHub PR comments."""

        build_labels = ['ci:build-linux','ci:build-macos','ci:build-windows','ci:run-qa-tests','ci:run-pytorch-tests','ci:build-aarch64','ci:build-x86_64']
        all_labels = [x.strip().lower() for x in os.environ["GITHUB_PR_LABELS"].split(",")]
        ci_labels = [label for label in all_labels if re.search("|".join(build_labels), label)]
        if not ci_labels:
            self.build_windows = True
            self.build_macos = True
            self.build_linux = True
            self.build_aarch64 = "--build-aarch64"
            self.build_x86_64 = "--build-x86_64"
            self.run_qa_tests = False
            self.run_pytorch_tests = False
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
                if "ci:run-pytorch-tests" == label:
                    self.build_windows = True
                    self.build_macos = True
                    self.build_linux = True
                    self.run_pytorch_tests = True
            if self.build_aarch64 == "" and self.build_x86_64 == "":
                self.build_aarch64 = "--build-aarch64"
                self.build_x86_64 = "--build-x86_64"

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

