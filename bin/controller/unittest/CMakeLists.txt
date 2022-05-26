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

project("ML controller unit tests")

set (SRCS
    Main.cc
    CBlockingCallCancellingStreamMonitorTest.cc
    CCommandProcessorTest.cc
    CResponseJsonWriterTest.cc
)

set(ML_LINK_LIBRARIES
    ${Boost_LIBRARIES_WITH_UNIT_TEST}
    ${LIBXML2_LIBRARIES}
    MlCore
    MlTest
    MlVer
)

ml_add_test_executable(controller ${SRCS})
