#!/bin/bash
#
env
if [[ "$GITHUB_PR_COMMENT_VAR_ACTION" == "debug" ]]; then export ML_DEBUG=1; fi
powershell -File .buildkite/scripts/steps/build_and_test.ps1
