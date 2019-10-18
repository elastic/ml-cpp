/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CFunctional.h>

#include <boost/test/unit_test.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>

#include <memory>

BOOST_AUTO_TEST_SUITE(CFunctionalTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testIsNull) {
    core::CFunctional::SIsNull isNull;

    {
        double five = 5.0;
        double* null = nullptr;
        const double* notNull = &five;
        BOOST_TEST(isNull(null));
        BOOST_TEST(!isNull(notNull));
    }
    {
        boost::optional<double> null;
        boost::optional<double> notNull(5.0);
        BOOST_TEST(isNull(null));
        BOOST_TEST(!isNull(notNull));
    }
    {
        std::shared_ptr<double> null;
        std::shared_ptr<double> notNull(new double(5.0));
        BOOST_TEST(isNull(null));
        BOOST_TEST(!isNull(notNull));
    }
}

BOOST_AUTO_TEST_CASE(testDereference) {
    double one(1.0);
    double two(2.0);
    double three(3.0);
    const double* null_ = nullptr;

    core::CFunctional::SDereference<core::CFunctional::SIsNull> derefIsNull;
    boost::optional<const double*> null(null_);
    boost::optional<const double*> notNull(&one);
    BOOST_TEST(derefIsNull(null));
    BOOST_TEST(!derefIsNull(notNull));

    std::less<double> less;
    core::CFunctional::SDereference<std::less<double>> derefLess;
    const double* values[] = {&one, &two, &three};
    for (std::size_t i = 0u; i < boost::size(values); ++i) {
        for (std::size_t j = 0u; j < boost::size(values); ++j) {
            BOOST_CHECK_EQUAL(less(*values[i], *values[j]),
                                 derefLess(values[i], values[j]));
        }
    }
}


BOOST_AUTO_TEST_SUITE_END()
