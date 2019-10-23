/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CProcess.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CProcessTest)

BOOST_AUTO_TEST_CASE(testPids) {
    ml::core::CProcess& process = ml::core::CProcess::instance();
    ml::core::CProcess::TPid pid = process.id();
    ml::core::CProcess::TPid ppid = process.parentId();

    LOG_DEBUG(<< "PID = " << pid << " and parent PID = " << ppid);

    BOOST_TEST_REQUIRE(pid != 0);
    BOOST_TEST_REQUIRE(ppid != 0);
    BOOST_TEST_REQUIRE(pid != ppid);
}

BOOST_AUTO_TEST_SUITE_END()
