#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/defines.mk

.PHONY: test
.PHONY: build
.PHONY: install

COMPONENTS= \
            3rd_party \
            lib \
            bin \

include $(CPP_SRC_HOME)/mk/toplevel.mk

