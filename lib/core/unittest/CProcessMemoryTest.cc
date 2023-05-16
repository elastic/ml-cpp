/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CProcessMemory.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CProcessMemoryTest)

BOOST_AUTO_TEST_CASE(testResidentSetSize) {
    BOOST_REQUIRE_GT(ml::core::CProcessMemory::residentSetSize(), 0);
}

BOOST_AUTO_TEST_CASE(testMaxResidentSetSize) {
    BOOST_REQUIRE_GT(ml::core::CProcessMemory::maxResidentSetSize(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
