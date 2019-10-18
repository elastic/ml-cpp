/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CWindowsError.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CWindowsErrorTest)

BOOST_AUTO_TEST_CASE(testErrors) {
    LOG_INFO(<< "Windows error 1 is : " << ml::core::CWindowsError(1));
    LOG_INFO(<< "Windows error 2 is : " << ml::core::CWindowsError(2));
    LOG_INFO(<< "Windows error 3 is : " << ml::core::CWindowsError(3));
    LOG_INFO(<< "Windows error 4 is : " << ml::core::CWindowsError(4));
    LOG_INFO(<< "Windows error 5 is : " << ml::core::CWindowsError(5));
    LOG_INFO(<< "Windows error 6 is : " << ml::core::CWindowsError(6));
}

BOOST_AUTO_TEST_SUITE_END()
