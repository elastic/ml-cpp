#!/usr/bin/env Python

import os

class Config:
    build_windows: bool = False
    build_macos: bool = False
    build_linux: bool = False

    def parse_comment(self):
        if "GITHUB_PR_COMMENT_VAR_PLATFORM" in os.environ:
            csv_platform = os.environ["GITHUB_PR_COMMENT_VAR_PLATFORM"]
            for each in [ x.strip().lower() for x in csv_platform.split(",")]:
                if each == "windows":
                    self.build_windows = True
                elif each == "macos":
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

        for k, v in os.environ.items():
            print(f'{k}={v}')

        if "GITHUB_PR_TRIGGER_COMMENT" in os.environ:
            self.parse_comment()
        else if "GITHUB_PR_LABELS" in os.environ:
            self.parse_label()
        else:
            self.build_windows = True
            self.build_macos = True
            self.build_linux = True

