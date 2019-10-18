/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CSleep.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CSleepTest)


BOOST_AUTO_TEST_CASE(testSleep) {
    ml::core_t::TTime start(ml::core::CTimeUtils::now());

    ml::core::CSleep::sleep(7500);

    ml::core_t::TTime end(ml::core::CTimeUtils::now());

    ml::core_t::TTime diff(end - start);
    LOG_DEBUG(<< "During 7.5 second wait, the clock advanced by " << diff << " seconds");

    // Clock time should be 7 or 8 seconds further ahead
    BOOST_TEST(diff >= 7);
    BOOST_TEST(diff <= 8);
}

BOOST_AUTO_TEST_SUITE_END()
