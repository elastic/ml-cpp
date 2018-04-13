/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CKdTreeTest.h"

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/COrderings.h>

#include <test/CRandomNumbers.h>

#include <vector>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TVector2 = maths::CVectorNx1<double, 2>;
using TDoubleVector2Pr = std::pair<double, TVector2>;
using TVector2Vec = std::vector<TVector2>;
using TVector5 = maths::CVectorNx1<double, 5>;
using TDoubleVector5Pr = std::pair<double, TVector5>;
using TVector5Vec = std::vector<TVector5>;

template<typename T>
std::string print(const T &t)
{
    std::ostringstream o;
    o << t;
    return o.str();
}

void CKdTreeTest::testBuild()
{
    LOG_DEBUG("+--------------------------+");
    LOG_DEBUG("|  CKdTreeTest::testBuild  |");
    LOG_DEBUG("+--------------------------+");

    const std::size_t numberTests = 200;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberTests; ++i)
    {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 2 * (i + 1), samples);

        TVector2Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 2)
        {
            points.push_back(TVector2(&samples[j], &samples[j + 2]));
        }

        maths::CKdTree<TVector2> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());
    }

    for (std::size_t i = 0u; i < numberTests; ++i)
    {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 5 * (i + 1), samples);

        TVector5Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 5)
        {
            points.push_back(TVector5(&samples[j], &samples[j + 5]));
        }

        maths::CKdTree<TVector5> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());
    }
}

void CKdTreeTest::testNearestNeighbour()
{
    LOG_DEBUG("+-------------------------------------+");
    LOG_DEBUG("|  CKdTreeTest::testNearestNeighbour  |");
    LOG_DEBUG("+-------------------------------------+");

    const std::size_t numberTests = 200;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberTests; ++i)
    {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 2 * (i + 1), samples);

        TVector2Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 2)
        {
            points.push_back(TVector2(&samples[j], &samples[j + 2]));
        }

        maths::CKdTree<TVector2> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());

        rng.generateUniformSamples(-150.0, 150.0, 2 * 10, samples);

        TVector2Vec tests;
        for (std::size_t j = 0u; j < samples.size(); j += 2)
        {
            tests.push_back(TVector2(&samples[j], &samples[j + 2]));
        }

        if (i % 10 == 0)
        {
            LOG_DEBUG("*** Test " << i << " ***");
        }
        for (std::size_t j = 0u; j < tests.size(); ++j)
        {
            using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<
                                        TDoubleVector2Pr, 1, maths::COrderings::SFirstLess>;

            TMinAccumulator expectedNearest;
            for (std::size_t k = 0u; k < points.size(); ++k)
            {
                expectedNearest.add(TDoubleVector2Pr((tests[j] - points[k]).euclidean(),
                                                     points[k]));
            }

            const TVector2 *nearest = kdTree.nearestNeighbour(tests[j]);
            CPPUNIT_ASSERT(nearest);
            if (i % 10 == 0)
            {
                LOG_DEBUG("Expected nearest = " << expectedNearest[0].second
                          << ", expected distance = " << expectedNearest[0].first);
                LOG_DEBUG("Nearest          = " << *nearest
                          << ", actual distance   = " << (tests[j] - *nearest).euclidean());
            }
            CPPUNIT_ASSERT_EQUAL(print(expectedNearest[0].second),
                                 print(*nearest));
        }
    }
}

CppUnit::Test *CKdTreeTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CKdTreeTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CKdTreeTest>(
                                   "CKdTreeTest::testBuild",
                                   &CKdTreeTest::testBuild) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKdTreeTest>(
                                   "CKdTreeTest::testNearestNeighbour",
                                   &CKdTreeTest::testNearestNeighbour) );

    return suiteOfTests;
}
