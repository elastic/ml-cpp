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

#include <core/CFunctional.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CFunctionalTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testDereference) {
    double one{1.0};
    double two{2.0};
    double three{3.0};
    std::less<double> less;
    core::CFunctional::SDereference<std::less<double>> derefLess;
    std::vector<const double*> values{&one, &two, &three};
    for (std::size_t i = 0; i < std::size(values); ++i) {
        for (std::size_t j = 0; j < std::size(values); ++j) {
            BOOST_REQUIRE_EQUAL(less(*values[i], *values[j]),
                                derefLess(values[i], values[j]));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
