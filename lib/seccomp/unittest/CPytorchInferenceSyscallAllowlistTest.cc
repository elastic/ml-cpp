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

#include <seccomp/CPytorchInferenceSyscallAllowlist.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CPytorchInferenceSyscallAllowlistTest)

#ifdef __linux__

BOOST_AUTO_TEST_CASE(testSandbox2AllowsAllLegacySyscalls) {
    BOOST_TEST_REQUIRE(ml::seccomp::pytorch_inference::sandbox2AllowsAllLegacySyscalls());
}

#endif

BOOST_AUTO_TEST_SUITE_END()
