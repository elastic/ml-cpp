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
include $(CPP_SRC_HOME)/mk/defines.mk

INSTALL_DIR=$(CPP_PLATFORM_HOME)/$(DYNAMIC_LIB_DIR)
LICENSE_DIR=$(CPP_PLATFORM_HOME)/..

.PHONY: licenses

all: install licenses

install: 3rd_party.sh
	./3rd_party.sh --add $(INSTALL_DIR)

licenses:
	$(MKDIR) $(LICENSE_DIR)
	tar cf - licenses | (cd $(LICENSE_DIR) && tar xvf -)

clean:
	echo Nothing to clean at present

test:
	echo No tests at present

