# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

ENV = env
REQUIRE = requirements.txt
BIN := $(ENV)/bin
SYS_VIRTUALENV := python3 -m venv

PIP := $(BIN)/pip
# TOX := $(BIN)/tox
PYTHON := $(BIN)/$(python)
TEST_RUNNER := $(BIN)/py.test
BUMPVERSION := bumpversion

REQUIREMENTS := $(shell find ./ -name $(REQUIRE))
REQUIREMENTS_LOG := .requirements.log
TEST_CODE := $(wildcard $(TESTDIR)/*.py)
TEST_RUNNER_PKGS = pytest
SRC_DIR := src
NB_DIR := notebooks
TEST_DIR := tests

TEST_NOTEBOOKS := \
	incremental_learning

PACKAGES := \
	incremental_learning

.DEFAULT_GOAL := help
.PHONY: all
.PHONY: test
.PHONY: test-jupyter
.PHONY: bump-minor
.PHONY: bump-major

all: env test test-notebooks ## Run all

env: $(REQUIREMENTS_LOG) src ## Install requirements and python packages

$(PIP):
	$(info Remember to source new environment  [ $(ENV) ])
	test -d $(ENV) || $(SYS_VIRTUALENV) $(ENV)

$(REQUIREMENTS_LOG): $(PIP) $(REQUIREMENTS)
	for f in $(REQUIREMENTS); do \
	  $(PIP) install -r $$f | tee -a $(REQUIREMENTS_LOG); \
	done

src: $(PIP) 
	$(PIP) install -e $(SRC_DIR)

bump-minor: $(SRC_DIR) ## Bump minor version of python packages
	cd $(SRC_DIR); $(BUMPVERSION) minor

bump-major: $(SRC_DIR) ## Bump major version of python packages
	cd $(SRC_DIR); $(BUMPVERSION) major

test: $(REQUIREMENTS_LOG) $(TEST_RUNNER) ## Run unit tests
	$(TEST_RUNNER) $(args) $(TEST_DIR)

test-notebooks: $(addprefix $(NB_DIR)/,$(TEST_NOTEBOOKS)) $(REQUIREMENTS_LOG) $(TEST_RUNNER) ## Run jupyter notebooks smoketests
	$(TEST_RUNNER) --nbsmoke-run $<

$(TEST_RUNNER):
	$(PIP) install $(TEST_RUNNER_PKGS) | tee -a $(REQUIREMENTS_LOG)

build-docker: ## Build docker image tagged IMAGE_NAME
ifdef IMAGE_NAME
	cd ..; \
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME) -f ./jupyter/docker/Dockerfile .
else
	@echo "You need to specify IMAGE_NAME:\nIMAGE_NAME=foobar make build-docker"
endif

run-docker: ## Run docker container for image IMAGE_NAME
ifdef IMAGE_NAME
	docker run -p 9999:9999 $(IMAGE_NAME):latest
else
	@echo "You need to specify IMAGE_NAME:\nIMAGE_NAME=foobar make run-docker"
endif

help: ## Print this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sed -n 's/^\(.*\): \(.*\)##\(.*\)/\1:\3/p' \
	| column -t  -s ':'

### Cleanup ##################################################################
.PHONY: clean clean-env clean-all clean-build clean-test

clean: clean-test clean-build

clean-env: clean  ## Delete environment
	-@rm -rf $(ENV)
	-@rm -rf $(REQUIREMENTS_LOG)
	-@rm -rf .tox

clean-all: clean clean-env

clean-build: ## Delete temporary files and directories
	@find $(SRC_DIR) -name '*.pyc' -delete
	@find $(SRC_DIR) -name '__pycache__' -delete
	@find $(SRC_DIR) -name '*.egg-info' -exec rm -r {} \;
	@find $(TEST_DIR) -name '*.pyc' -delete 2>/dev/null || true
	@find $(TEST_DIR) -name '__pycache__' -delete 2>/dev/null || true
	-@rm -rf __pycache__

clean-test: ## Delete temporary files for testing
	@find $(TEST_DIR) -name '.pytest_cache' -exec rm -r {} +
	@find $(TEST_DIR) -name '__pycache__' -exec rm -r {} +


clean-notebooks: ## Delete notebook checkpoints
	-@find $(NB_DIR) -type d -name '.ipynb_checkpoints' -delete
	