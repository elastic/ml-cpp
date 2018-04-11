/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include "CSetToolsTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CSetTools.h>

#include <test/CRandomNumbers.h>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

void CSetToolsTest::testInplaceSetDifference() {
    LOG_DEBUG(<< "+-------------------------------------------+");
    LOG_DEBUG(<< "|  CSetToolsTest::testInplaceSetDifference  |");
    LOG_DEBUG(<< "+-------------------------------------------+");

    // Test some edge cases.
    {
        LOG_DEBUG(<< "Edge cases");

        double a[] = {1.0, 1.1, 1.2, 3.4, 7.8};
        TDoubleVec A(boost::begin(a), boost::end(a));

        for (std::size_t i = 0u; i < boost::size(a); ++i) {
            TDoubleVec left;
            for (std::size_t j = 0; j < i; ++j) {
                left.push_back(a[j]);
            }
            TDoubleVec expected;
            std::set_difference(A.begin(), A.end(), left.begin(), left.end(), std::back_inserter(expected));
            TDoubleVec test = A;
            maths::CSetTools::inplace_set_difference(test, left.begin(), left.end());
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(A) << ", B = " << core::CContainerPrinter::print(left)
                      << ", A - B = " << core::CContainerPrinter::print(test));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected), core::CContainerPrinter::print(test));

            TDoubleVec right;
            for (std::size_t j = i; j < boost::size(a); ++j) {
                right.push_back(a[j]);
            }
            expected.clear();
            std::set_difference(A.begin(), A.end(), right.begin(), right.end(), std::back_inserter(expected));
            test = A;
            maths::CSetTools::inplace_set_difference(test, right.begin(), right.end());
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(A) << ", B = " << core::CContainerPrinter::print(right)
                      << ", A - B = " << core::CContainerPrinter::print(test));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected), core::CContainerPrinter::print(test));
        }
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 100; ++t) {
        TDoubleVec A;
        rng.generateUniformSamples(0.0, 100.0, t, A);
        std::sort(A.begin(), A.end());

        TDoubleVec B;
        TDoubleVec mask;
        rng.generateUniformSamples(0.0, 1.0, t, mask);
        for (std::size_t i = 0u; i < mask.size(); ++i) {
            if (mask[i] < 0.2) {
                B.push_back(A[i]);
            }
        }

        TDoubleVec expected;
        std::set_difference(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(expected));

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(A));
            LOG_DEBUG(<< "B = " << core::CContainerPrinter::print(B));
        }

        maths::CSetTools::inplace_set_difference(A, B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "A - B = " << core::CContainerPrinter::print(A));
        }

        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected), core::CContainerPrinter::print(A));
    }
}

void CSetToolsTest::testSetSizes() {
    LOG_DEBUG(<< "+-------------------------------+");
    LOG_DEBUG(<< "|  CSetToolsTest::testSetSizes  |");
    LOG_DEBUG(<< "+-------------------------------+");

    {
        LOG_DEBUG(<< "Edge cases");

        double a[] = {1.0, 1.1, 1.2, 3.4, 7.8};
        TDoubleVec A(boost::begin(a), boost::end(a));

        for (std::size_t i = 0u; i < boost::size(a); ++i) {
            TDoubleVec left;
            for (std::size_t j = 0; j < i; ++j) {
                left.push_back(a[j]);
            }
            TDoubleVec expected;
            std::set_intersection(A.begin(), A.end(), left.begin(), left.end(), std::back_inserter(expected));
            std::size_t test = maths::CSetTools::setIntersectSize(A.begin(), A.end(), left.begin(), left.end());
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(A) << ", B = " << core::CContainerPrinter::print(left)
                      << ", |A ^ B| = " << test);
            CPPUNIT_ASSERT_EQUAL(expected.size(), test);

            TDoubleVec right;
            for (std::size_t j = i; j < boost::size(a); ++j) {
                right.push_back(a[j]);
            }
            expected.clear();
            std::set_intersection(A.begin(), A.end(), right.begin(), right.end(), std::back_inserter(expected));
            test = maths::CSetTools::setIntersectSize(A.begin(), A.end(), right.begin(), right.end());
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(A) << ", B = " << core::CContainerPrinter::print(right)
                      << ", |A ^ B| = " << test);
            CPPUNIT_ASSERT_EQUAL(expected.size(), test);

            expected.clear();
            std::set_union(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(expected));
            test = maths::CSetTools::setUnionSize(left.begin(), left.end(), right.begin(), right.end());
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(left) << ", B = " << core::CContainerPrinter::print(right)
                      << ", |A U B| = " << test);
            CPPUNIT_ASSERT_EQUAL(expected.size(), test);
        }
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 100; ++t) {
        TDoubleVec A;
        rng.generateUniformSamples(0.0, 100.0, t, A);
        std::sort(A.begin(), A.end());

        TDoubleVec B;
        TDoubleVec mask;
        rng.generateUniformSamples(0.0, 1.0, t, mask);
        for (std::size_t i = 0u; i < mask.size(); ++i) {
            if (mask[i] < 0.2) {
                B.push_back(A[i]);
            }
        }

        TDoubleVec expected;
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(expected));

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "A = " << core::CContainerPrinter::print(A));
            LOG_DEBUG(<< "B = " << core::CContainerPrinter::print(B));
        }

        std::size_t test = maths::CSetTools::setIntersectSize(A.begin(), A.end(), B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "|A ^ B| = " << test);
        }

        CPPUNIT_ASSERT_EQUAL(expected.size(), test);

        expected.clear();
        std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(expected));

        test = maths::CSetTools::setUnionSize(A.begin(), A.end(), B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "|A U B| = " << test);
        }

        CPPUNIT_ASSERT_EQUAL(expected.size(), test);
    }
}

void CSetToolsTest::testJaccard() {
    LOG_DEBUG(<< "+------------------------------+");
    LOG_DEBUG(<< "|  CSetToolsTest::testJaccard  |");
    LOG_DEBUG(<< "+------------------------------+");

    {
        LOG_DEBUG(<< "Edge cases");

        double A[] = {0.0, 1.2, 3.2};
        double B[] = {0.0, 1.2, 3.2, 5.1};

        CPPUNIT_ASSERT_EQUAL(0.0, maths::CSetTools::jaccard(A, A, B, B));
        CPPUNIT_ASSERT_EQUAL(1.0, maths::CSetTools::jaccard(A, A + 3, B, B + 3));
        CPPUNIT_ASSERT_EQUAL(0.75, maths::CSetTools::jaccard(A, A + 3, B, B + 4));
        CPPUNIT_ASSERT_EQUAL(0.0, maths::CSetTools::jaccard(A, A + 3, B + 3, B + 4));
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 500; ++t) {
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
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(AIntersectB));

        TSizeVec AUnionB;
        std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(AUnionB));

        double expected = static_cast<double>(AIntersectB.size()) / static_cast<double>(AUnionB.size());
        double actual = maths::CSetTools::jaccard(A.begin(), A.end(), B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "Jaccard expected = " << expected);
            LOG_DEBUG(<< "Jaccard actual   = " << actual);
        }
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
}

void CSetToolsTest::testOverlap() {
    LOG_DEBUG(<< "+------------------------------+");
    LOG_DEBUG(<< "|  CSetToolsTest::testOverlap  |");
    LOG_DEBUG(<< "+------------------------------+");

    {
        LOG_DEBUG(<< "Edge cases");

        double A[] = {0.0, 1.2, 3.2};
        double B[] = {0.0, 1.2, 3.2, 5.1};

        CPPUNIT_ASSERT_EQUAL(0.0, maths::CSetTools::overlap(A, A, B, B));
        CPPUNIT_ASSERT_EQUAL(1.0, maths::CSetTools::overlap(A, A + 3, B, B + 3));
        CPPUNIT_ASSERT_EQUAL(1.0, maths::CSetTools::overlap(A, A + 3, B, B + 4));
        CPPUNIT_ASSERT_EQUAL(0.0, maths::CSetTools::overlap(A, A + 3, B + 3, B + 4));
    }

    LOG_DEBUG(<< "Random");

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 500; ++t) {
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
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(AIntersectB));

        std::size_t min = std::min(A.size(), B.size());

        double expected = static_cast<double>(AIntersectB.size()) / static_cast<double>(min);
        double actual = maths::CSetTools::overlap(A.begin(), A.end(), B.begin(), B.end());

        if ((t + 1) % 10 == 0) {
            LOG_DEBUG(<< "Overlap expected = " << expected);
            LOG_DEBUG(<< "Overlap actual   = " << actual);
        }
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
}

CppUnit::Test* CSetToolsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSetToolsTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CSetToolsTest>("CSetToolsTest::testInplaceSetDifference", &CSetToolsTest::testInplaceSetDifference));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSetToolsTest>("CSetToolsTest::testSetSizes", &CSetToolsTest::testSetSizes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSetToolsTest>("CSetToolsTest::testJaccard", &CSetToolsTest::testJaccard));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSetToolsTest>("CSetToolsTest::testOverlap", &CSetToolsTest::testOverlap));

    return suiteOfTests;
}
