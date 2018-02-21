#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

include $(CPP_SRC_HOME)/mk/stdcpptest.mk

test:$(TARGET)
	$(PRE_TEST_CMDS)
	$(TEST_CMDS)
	$(POST_TEST_CMDS)
