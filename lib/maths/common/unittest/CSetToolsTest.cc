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

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/common/CSetTools.h>

#include <test/CRandomNumbers.h>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

BOOST_AUTO_TEST_SUITE(CSetToolsTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

BOOST_AUTO_TEST_CASE(testSimultaneousRemoveIf) {

    LOG_DEBUG(<< "Edge cases");
    {
        TSizeVec keys{1, 0, 1, 0, 1, 0, 0, 0};
        TSizeVec values{1, 2, 3, 4, 5, 6, 7, 8, 9};
        BOOST_TEST_REQUIRE(maths::common::CSetTools::simultaneousRemoveIf(
                               [](auto key) { return key == 1; }, keys, values) == false);
    }
    {
        TSizeVec keys{1, 1, 1, 1, 1};
        TSizeVec values{1, 2, 3, 4, 5};

        BOOST_TEST_REQUIRE(maths::common::CSetTools::simultaneousRemoveIf(
            [](auto key) { return key == 0; }, keys, values));
        LOG_DEBUG(<< "keys = " << keys << ", values = " << values);
        BOOST_TEST_REQUIRE("[1, 1, 1, 1, 1]", core::CContainerPrinter::print(keys));
        BOOST_REQUIRE_EQUAL("[1, 2, 3, 4, 5]", core::CContainerPrinter::print(values));

        BOOST_TEST_REQUIRE(maths::common::CSetTools::simultaneousRemoveIf(
            [](auto key) { return key == 1; }, keys, values));
        LOG_DEBUG(<< "keys = " << keys << ", values = " << values);
        BOOST_TEST_REQUIRE(keys.empty());
        BOOST_TEST_REQUIRE(values.empty());
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    TSizeVec actualKeys;
    TSizeVec actualValues1;
    TSizeVec actualValues2;
    TSizeVec actualValues3;
    TSizeVec expectedKeys;
    TSizeVec expectedValues1;
    TSizeVec expectedValues2;
    TSizeVec expectedValues3;

    for (std::size_t t = 0; t < 100; ++t) {
        rng.generateUniformSamples(0, 5, 20, actualKeys);
        actualValues1.resize(20);
        actualValues2.resize(20);
        actualValues3.resize(20);
        std::iota(actualValues1.begin(), actualValues1.end(), 0);
        std::iota(actualValues2.begin(), actualValues2.end(), 0);
        std::iota(actualValues3.begin(), actualValues3.end(), 0);

        expectedKeys = actualKeys;
        expectedValues1 = actualValues1;
        expectedValues2 = actualValues2;
        expectedValues3 = actualValues3;

        expectedValues1.erase(
            std::remove_if(expectedValues1.begin(), expectedValues1.end(),
                           [&](auto value) { return expectedKeys[value] == 1; }),
            expectedValues1.end());
        expectedValues2.erase(
            std::remove_if(expectedValues2.begin(), expectedValues2.end(),
                           [&](auto value) { return expectedKeys[value] == 1; }),
            expectedValues2.end());
        expectedValues3.erase(
            std::remove_if(expectedValues3.begin(), expectedValues3.end(),
                           [&](auto value) { return expectedKeys[value] == 1; }),
            expectedValues3.end());
        expectedKeys.erase(std::remove_if(expectedKeys.begin(), expectedKeys.end(),
                                          [&](auto key) { return key == 1; }),
                           expectedKeys.end());

        BOOST_TEST_REQUIRE(maths::common::CSetTools::simultaneousRemoveIf(
            [](auto key) { return key == 1; }, actualKeys, actualValues1,
            actualValues2, actualValues3));

        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedKeys),
                            core::CContainerPrinter::print(actualKeys));
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedValues1),
                            core::CContainerPrinter::print(actualValues1));
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedValues2),
                            core::CContainerPrinter::print(actualValues2));
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedValues3),
                            core::CContainerPrinter::print(actualValues3));

        if (t % 20 == 0) {
            LOG_DEBUG(<< "actual keys   = " << actualKeys);
            LOG_DEBUG(<< "expected keys = " << expectedKeys);
            LOG_DEBUG(<< "actual values1   = " << actualValues1);
            LOG_DEBUG(<< "expected values1 = " << expectedValues1);
            LOG_DEBUG(<< "actual values2   = " << actualValues2);
            LOG_DEBUG(<< "expected values2 = " << expectedValues2);
            LOG_DEBUG(<< "actual values3   = " << actualValues3);
            LOG_DEBUG(<< "expected values3 = " << expectedValues3);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSetSizes) {
    {
        LOG_DEBUG(<< "Edge cases");

        TDoubleVec A{1.0, 1.1, 1.2, 3.4, 7.8};

        for (std::size_t i = 0; i < A.size(); ++i) {
            TDoubleVec left;
            for (std::size_t j = 0; j < i; ++j) {
                left.push_back(A[j]);
            }
            TDoubleVec expected;
            std::set_intersection(A.begin(), A.end(), left.begin(), left.end(),
                                  std::back_inserter(expected));
            std::size_t test = maths::common::CSetTools::setIntersectSize(
                A.begin(), A.end(), left.begin(), left.end());
            LOG_DEBUG(<< "A = " << A << ", B = " << left << ", |A ^ B| = " << test);
            BOOST_REQUIRE_EQUAL(expected.size(), test);

            TDoubleVec right;
            for (std::size_t j = i; j < A.size(); ++j) {
                right.push_back(A[j]);
            }
            expected.clear();
            std::set_intersection(A.begin(), A.end(), right.begin(),
                                  right.end(), std::back_inserter(expected));
            test = maths::common::CSetTools::setIntersectSize(
                A.begin(), A.end(), right.begin(), right.end());
            LOG_DEBUG(<< "A = " << A << ", B = " << right << ", |A ^ B| = " << test);
            BOOST_REQUIRE_EQUAL(expected.size(), test);

            expected.clear();
            std::set_union(left.begin(), left.end(), right.begin(), right.end(),
                           std::back_inserter(expected));
            test = maths::common::CSetTools::setUnionSize(
                left.begin(), left.end(), right.begin(), right.end());
            LOG_DEBUG(<< "A = " << left << ", B = " << right << ", |A U B| = " << test);
            BOOST_REQUIRE_EQUAL(expected.size(), test);
        }
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0; t < 100; ++t) {
        TDoubleVec A;
        rng.generateUniformSamples(0.0, 100.0, t, A);
        std::sort(A.begin(), A.end());

        TDoubleVec B;
        TDoubleVec mask;
        rng.generateUniformSamples(0.0, 1.0, t, mask);
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] < 0.2) {
                B.push_back(A[i]);
            }
        }

        TDoubleVec expected;
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                              std::back_inserter(expected));

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "A = " << A);
            LOG_DEBUG(<< "B = " << B);
        }

        std::size_t test = maths::common::CSetTools::setIntersectSize(
            A.begin(), A.end(), B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "|A ^ B| = " << test);
        }

        BOOST_REQUIRE_EQUAL(expected.size(), test);

        expected.clear();
        std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(expected));

        test = maths::common::CSetTools::setUnionSize(A.begin(), A.end(),
                                                      B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "|A U B| = " << test);
        }

        BOOST_REQUIRE_EQUAL(expected.size(), test);
    }
}

BOOST_AUTO_TEST_CASE(testJaccard) {
    LOG_DEBUG(<< "Edge cases");
    {
        TDoubleVec A{0.0, 1.2, 3.2};
        TDoubleVec B{0.0, 1.2, 3.2, 5.1};
        BOOST_REQUIRE_EQUAL(0.0, maths::common::CSetTools::jaccard(
                                     A.begin(), A.begin(), B.begin(), B.begin()));
        BOOST_REQUIRE_EQUAL(1.0, maths::common::CSetTools::jaccard(
                                     A.begin(), A.end(), B.begin(), B.begin() + 3));
        BOOST_REQUIRE_EQUAL(0.75, maths::common::CSetTools::jaccard(
                                      A.begin(), A.end(), B.begin(), B.end()));
        BOOST_REQUIRE_EQUAL(0.0, maths::common::CSetTools::jaccard(
                                     A.begin(), A.end(), B.begin() + 3, B.end()));
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0; t < 500; ++t) {
        TSizeVec sizes;
        rng.generateUniformSamples(t / 2 + 1, (3 * t) / 2 + 2, 2, sizes);

        TSizeVec A;
        rng.generateUniformSamples(0, (3 * t) / 2 + 1, sizes[0], A);
        std::sort(A.begin(), A.end());
        A.erase(std::unique(A.begin(), A.end()), A.end());

        TSizeVec B;
        rng.generateUniformSamples(0, (3 * t) / 2 + 1, sizes[1], B);
        std::sort(B.begin(), B.end());
        B.erase(std::unique(B.begin(), B.end()), B.end());

        TSizeVec AIntersectB;
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                              std::back_inserter(AIntersectB));

        TSizeVec AUnionB;
        std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(AUnionB));

        double expected = static_cast<double>(AIntersectB.size()) /
                          static_cast<double>(AUnionB.size());
        double actual = maths::common::CSetTools::jaccard(A.begin(), A.end(),
                                                          B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "Jaccard expected = " << expected);
            LOG_DEBUG(<< "Jaccard actual   = " << actual);
        }
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
}

BOOST_AUTO_TEST_CASE(testOverlap) {
    LOG_DEBUG(<< "Edge cases");
    {
        TDoubleVec A{0.0, 1.2, 3.2};
        TDoubleVec B{0.0, 1.2, 3.2, 5.1};
        BOOST_REQUIRE_EQUAL(0.0, maths::common::CSetTools::overlap(
                                     A.begin(), A.begin(), B.begin(), B.begin()));
        BOOST_REQUIRE_EQUAL(1.0, maths::common::CSetTools::overlap(
                                     A.begin(), A.end(), B.begin(), B.begin() + 3));
        BOOST_REQUIRE_EQUAL(1.0, maths::common::CSetTools::overlap(
                                     A.begin(), A.end(), B.begin(), B.end()));
        BOOST_REQUIRE_EQUAL(0.0, maths::common::CSetTools::overlap(
                                     A.begin(), A.end(), B.begin() + 3, B.end()));
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0; t < 500; ++t) {
        TSizeVec sizes;
        rng.generateUniformSamples(t / 2 + 1, (3 * t) / 2 + 2, 2, sizes);

        TSizeVec A;
        rng.generateUniformSamples(0, (3 * t) / 2 + 1, sizes[0], A);
        std::sort(A.begin(), A.end());
        A.erase(std::unique(A.begin(), A.end()), A.end());

        TSizeVec B;
        rng.generateUniformSamples(0, (3 * t) / 2 + 1, sizes[1], B);
        std::sort(B.begin(), B.end());
        B.erase(std::unique(B.begin(), B.end()), B.end());

        TSizeVec AIntersectB;
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                              std::back_inserter(AIntersectB));

        std::size_t min = std::min(A.size(), B.size());

        double expected = static_cast<double>(AIntersectB.size()) /
                          static_cast<double>(min);
        double actual = maths::common::CSetTools::overlap(A.begin(), A.end(),
                                                          B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "Overlap expected = " << expected);
            LOG_DEBUG(<< "Overlap actual   = " << actual);
        }
        BOOST_REQUIRE_EQUAL(expected, actual);
    }
}

BOOST_AUTO_TEST_SUITE_END()
