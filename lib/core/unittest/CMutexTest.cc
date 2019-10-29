/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMutex.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CMutexTest)

BOOST_AUTO_TEST_CASE(testRecursive) {
    ml::core::CMutex mutex;

    mutex.lock();
    mutex.unlock();

    mutex.lock();
    // If the mutex isn't recursive as expected then this test
    // will deadlock rather than fail :-(
    // The assertion here is just to stop Boost.Test complaining
    // that there are no assertions
    BOOST_REQUIRE_NO_THROW(mutex.lock());
    mutex.unlock();
    mutex.unlock();
}

BOOST_AUTO_TEST_SUITE_END()
