/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CImmutableRadixSet.h>
#include <core/CLogger.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <vector>

using namespace ml;

BOOST_AUTO_TEST_SUITE(CImmutableRadixSetTest)

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

BOOST_AUTO_TEST_CASE(construction) {

    core::CImmutableRadixSet<double> test1{1.0, 3.0, 1.0, 6.0};
    BOOST_REQUIRE_EQUAL(3, test1.size());
    BOOST_REQUIRE_EQUAL(1.0, test1[0]);
    BOOST_REQUIRE_EQUAL(3.0, test1[1]);
    BOOST_REQUIRE_EQUAL(6.0, test1[2]);

    core::CImmutableRadixSet<double> test2{3.4, 1.0, 1.0, 0.1, 1.0};
    BOOST_REQUIRE_EQUAL(3, test2.size());
    BOOST_REQUIRE_EQUAL(0.1, test2[0]);
    BOOST_REQUIRE_EQUAL(1.0, test2[1]);
    BOOST_REQUIRE_EQUAL(3.4, test2[2]);
}

BOOST_AUTO_TEST_CASE(testUpperBound) {

    test::CRandomNumbers rng;

    LOG_DEBUG(<< "Empty");
    {
        core::CImmutableRadixSet<double> empty;
        BOOST_REQUIRE_EQUAL(0, empty.upperBound(0.0));
    }
    {
        core::CImmutableRadixSet<double> unary{1.0};
        BOOST_REQUIRE_EQUAL(0, unary.upperBound(0.0));
        BOOST_REQUIRE_EQUAL(1, unary.upperBound(1.0));
    }

    for (std::size_t test = 0; test < 1000; ++test) {
        TDoubleVec values;
        rng.generateUniformSamples(5.0, 95.0, 50, values);
        for (auto& value : values) {
            value = std::floor(value);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());

        core::CImmutableRadixSet<double> set{values};

        TSizeVec probes;
        rng.generateUniformSamples(0, 100, 500, probes);

        for (auto probe : probes) {
            BOOST_REQUIRE_EQUAL(std::upper_bound(values.begin(), values.end(), probe) - values.begin(),
                                set.upperBound(probe));
        }
    }

    for (std::size_t test = 0; test < 1000; ++test) {
        TDoubleVec values;
        rng.generateUniformSamples(0.0, 10.0, 40, values);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());

        core::CImmutableRadixSet<double> set{values};

        TDoubleVec probes;
        rng.generateNormalSamples(0.0, 20.0, 500, probes);

        for (auto probe : probes) {
            BOOST_REQUIRE_EQUAL(std::upper_bound(values.begin(), values.end(), probe) - values.begin(),
                                set.upperBound(probe));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
