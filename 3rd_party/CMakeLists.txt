#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

set(INSTALL_DIR $ENV{CPP_PLATFORM_HOME}/${DYNAMIC_LIB_DIR})
set(LICENSE_DIR $ENV{CPP_PLATFORM_HOME}/../licenses)

# For now we want to co-exist with static Makefiles
# so expect to be built out-of-source, hence the need
# to change the working directory upon execution
add_custom_target(licenses ALL
  COMMAND cmake -E make_directory $ENV{CPP_PLATFORM_HOME}
  COMMAND cmake -E copy_directory licenses ${LICENSE_DIR}
  WORKING_DIRECTORY $ENV{CPP_SRC_HOME}/3rd_party
)

# For now we want to co-exist with static Makefiles
# so expect to be built out-of-source, hence the need
# to change the working directory upon execution
add_custom_target(
    eigen
    DEPENDS 3rd_party.sh pull-eigen.sh
    COMMAND sh -c "./3rd_party.sh --add ${INSTALL_DIR}"
    COMMAND sh -c "./pull-eigen.sh"
    WORKING_DIRECTORY $ENV{CPP_SRC_HOME}/3rd_party
)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES $ENV{CPP_SRC_HOME}/3rd_party/eigen)
