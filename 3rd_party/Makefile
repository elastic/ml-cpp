#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
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

