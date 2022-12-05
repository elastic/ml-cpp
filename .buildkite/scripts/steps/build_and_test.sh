#!/bin/bash

# For now, re-use our existing CI scripts based on Docker
${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
