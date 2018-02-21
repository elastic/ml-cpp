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

CPPFLAGS+= $(INCLUDE_PATH)

ifdef NO_TEST_CASES
TEST_CMDS=@ echo 'ERROR!!!! NO UNIT TESTS EXIST FOR $(TARGET)';
TEST_COMPILE_CMDS=@ echo 'ERROR!!!! NO UNIT TESTS EXIST FOR $(TARGET)';
TEST_OBJ_COMPILE_CMDS=@ echo 'ERROR!!!! NO UNIT TESTS EXIST FOR $(TARGET)';
else
	ifndef TEST_DIRS
	TEST_DIRS=unittest
	endif
endif

ifdef TEST_DIRS
CLEAN_CMDS=@ for i in $(TEST_DIRS) ; \
do \
(cd $$i && $(MAKE) clean ); \
done
endif

ifndef TEST_CMDS
TEST_CMDS=\
@ FAILED=0; \
for i in $(TEST_DIRS) ; \
do \
	(cd $$i && $(MAKE) test ); \
	if [ $$? -ne 0 ]; then \
		FAILED=1; \
		echo "`pwd` make test FAILURE!!!"; \
		if [ -z "$$ML_KEEP_GOING" ] ; then \
			exit 1; \
		fi; \
	fi; \
done; \
exit $$FAILED
endif

ifndef TEST_OBJ_COMPILE_CMDS
TEST_OBJ_COMPILE_CMDS:= \
@for i in $(TEST_DIRS) ; \
do \
	(cd $$i && $(MAKE) objcompile ); \
done;
endif

ifndef RELINK_CMDS
RELINK_CMDS=\
for i in $(TEST_DIRS) ; \
do \
	(cd $$i && $(MAKE) relink ); \
done
endif

