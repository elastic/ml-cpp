#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#

include $(CPP_SRC_HOME)/mk/rules.mk

ifndef DEFAULT_TEST_CMDS
DEFAULT_TEST_CMDS=./$(TARGET) $(TEST_ARGS)

ifdef TEST_CASE
DEFAULT_TEST_CMDS+= $(TEST_CASE)
endif

endif # ifndef DEFAULT_TEST_CMDS


ifdef CPP_CROSS_COMPILE
TEST_CMDS=@ echo 'WARNING!!!! CANNOT RUN UNIT TESTS WHEN CROSS COMPILING'
else
#if we haven't defined a specific test just run it
ifndef TEST_CMDS
TEST_CMDS=$(DEFAULT_TEST_CMDS)
endif
endif

ifndef PRE_TEST_CMDS
PRE_TEST_CMDS=
endif

ifndef POST_TEST_CMDS
POST_TEST_CMDS=
endif
