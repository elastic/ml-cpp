#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/defines.mk

TESTDIR = tests
ENV = env
REQUIRE = requirements.txt
BIN := $(ENV)/bin
SYS_VIRTUALENV := python3 -m venv

PIP := $(BIN)/pip
# TOX := $(BIN)/tox
PYTHON := $(BIN)/$(python)
TEST_RUNNER := $(BIN)/py.test
JUPYTER := $(BIN)/jupyter

REQUIREMENTS := $(shell find ./ -name $(REQUIRE))
REQUIREMENTS_LOG := .requirements.log
TEST_CODE := $(wildcard $(TESTDIR)/*.py)
TEST_RUNNER_PKGS = pytest
SRC_DIR := src/

TEST_NOTEBOOKS := \
	incremental_learning

.PHONY: all
.PHONY: test
.PHONY: test-jupyter

all: test

env: $(REQUIREMENTS_LOG) src

$(PIP):
	$(info Remember to source new environment  [ $(ENV) ])
	test -d $(ENV) || $(SYS_VIRTUALENV) $(ENV)

$(REQUIREMENTS_LOG): $(PIP) $(REQUIREMENTS)
	for f in $(REQUIREMENTS); do \
	  $(PIP) install -r $$f | tee -a $(REQUIREMENTS_LOG); \
	done

src: $(PIP) $(SRC_DIR)
	$(PIP) install -e $(SRC_DIR)

test: $(REQUIREMENTS_LOG) $(TEST_RUNNER)
	$(TEST_RUNNER) $(args) $(TESTDIR)

test-notebooks: $(addprefix notebooks/,$(TEST_NOTEBOOKS)) $(REQUIREMENTS_LOG) $(TEST_RUNNER) 
	$(TEST_RUNNER) --nbsmoke-run $<

$(TEST_RUNNER):
	$(PIP) install $(TEST_RUNNER_PKGS) | tee -a $(REQUIREMENTS_LOG)

### Cleanup ##################################################################
.PHONY: clean clean-env clean-all clean-build clean-test

clean: clean-test clean-build

clean-env: clean
	-@rm -rf $(ENV)
	-@rm -rf $(REQUIREMENTS_LOG)
	-@rm -rf .tox

clean-all: clean clean-env

clean-build:
	@find $(PKGDIR) -name '*.pyc' -delete
	@find $(PKGDIR) -name '__pycache__' -delete
	@find $(TESTDIR) -name '*.pyc' -delete 2>/dev/null || true
	@find $(TESTDIR) -name '__pycache__' -delete 2>/dev/null || true
	-@rm -rf $(EGG_INFO)
	-@rm -rf __pycache__

clean-test:
	-@rm -rf .pytest_cache
	-@rm -rf .cache
	-@rm -rf .coverage

clean-notebook:
	