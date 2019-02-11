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

namespace {
using TDoubleVec = std::vector<double>;
using TVector2 = maths::CVectorNx1<double, 2>;
using TDoubleVector2Pr = std::pair<double, TVector2>;
using TDoubleVector2PrVec = std::vector<TDoubleVector2Pr>;
using TVector2Vec = std::vector<TVector2>;
using TVector5 = maths::CVectorNx1<double, 5>;
using TDoubleVector5Pr = std::pair<double, TVector5>;
using TDoubleVector5PrVec = std::vector<TDoubleVector5Pr>;
using TVector5Vec = std::vector<TVector5>;
using TVector = maths::CVector<double>;

class CVectorCountMoves : public TVector {
public:
    CVectorCountMoves(const TVector& vector) : TVector(vector), m_Moved(0) {}
    CVectorCountMoves(const CVectorCountMoves& other)
        : TVector(static_cast<const TVector&>(other)), m_Moved(other.m_Moved) {}
    CVectorCountMoves(CVectorCountMoves&& other)
        : TVector(static_cast<TVector&&>(other)), m_Moved(other.m_Moved + 1) {}

    CVectorCountMoves& operator=(const CVectorCountMoves& other) {
        static_cast<TVector&>(*this) = static_cast<const TVector&>(other);
        m_Moved = other.m_Moved;
        return *this;
    }
    CVectorCountMoves& operator=(CVectorCountMoves&& other) {
        static_cast<TVector&>(*this) = static_cast<TVector&&>(other);
        ++m_Moved;
        return *this;
    }

    std::size_t moves() const { return m_Moved; }

private:
    std::size_t m_Moved;
};
using TVectorCountMovesVec = std::vector<CVectorCountMoves>;

template<typename POINT>
void nearestNeightbours(std::size_t k,
                        const std::vector<POINT>& points,
                        const POINT& point,
                        std::vector<std::pair<double, POINT>>& result) {
    using TDoublePointPr = std::pair<double, POINT>;
    using TMinDoublePointPrAccumulator =
        maths::CBasicStatistics::COrderStatisticsHeap<TDoublePointPr>;

    result.clear();
    if (k > 0) {
        TMinDoublePointPrAccumulator result_(k);
        for (const auto& point_ : points) {
            if (&point != &point_) {
                result_.add({maths::las::distance(point, point_), point_});
            }
        }
        result_.sort();
        result.assign(result_.begin(), result_.end());
    }
}

template<typename POINT>
std::string print(const POINT& t) {
    std::ostringstream o;
    o << t;
    return o.str();
}
}

void CKdTreeTest::testBuild() {
    const std::size_t numberTests = 200;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberTests; ++i) {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 2 * (i + 1), samples);

        TVector2Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 2) {
            points.emplace_back(&samples[j], &samples[j + 2]);
        }

        maths::CKdTree<TVector2> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());
    }

    for (std::size_t i = 0u; i < numberTests; ++i) {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 5 * (i + 1), samples);

        TVector5Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 5) {
            points.emplace_back(&samples[j], &samples[j + 5]);
        }

        maths::CKdTree<TVector5> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());
    }
}

void CKdTreeTest::testBuildWithMove() {
    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(-100.0, 100.0, 50, samples);

    TVectorCountMovesVec points;
    for (std::size_t i = 0u; i < samples.size(); /**/) {
        TVector point(5);
        while (i++ < 5) {
            point(i % 5) = samples[i];
        }
        points.emplace_back(point);
    }

    std::size_t moveCounts[]{0u, 0u};
    {
        maths::CKdTree<CVectorCountMoves> kdTree;
        kdTree.build(points);
        for (const auto& point : kdTree) {
            moveCounts[0] += point.moves();
        }
    }
    {
        maths::CKdTree<CVectorCountMoves> kdTree;
        kdTree.build(std::move(points));
        for (const auto& point : kdTree) {
            moveCounts[1] += point.moves();
        }
    }
    LOG_DEBUG(<< "move count = " << moveCounts[1] - moveCounts[0]);
    CPPUNIT_ASSERT(moveCounts[1] > moveCounts[0] + points.size());
}

void CKdTreeTest::testNearestNeighbour() {
    const std::size_t numberTests = 200u;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberTests; ++i) {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 2 * (i + 1), samples);

        TVector2Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 2) {
            points.emplace_back(&samples[j], &samples[j + 2]);
        }

        maths::CKdTree<TVector2> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());

        rng.generateUniformSamples(-150.0, 150.0, 2 * 10, samples);

        TVector2Vec tests;
        for (std::size_t j = 0u; j < samples.size(); j += 2) {
            tests.emplace_back(&samples[j], &samples[j + 2]);
        }

        if (i % 10 == 0) {
            LOG_DEBUG(<< "*** Test " << i << " ***");
        }
        for (std::size_t j = 0u; j < tests.size(); ++j) {
            TDoubleVector2PrVec expectedNearest;
            nearestNeightbours(1, points, tests[j], expectedNearest);

            const TVector2* nearest = kdTree.nearestNeighbour(tests[j]);
            CPPUNIT_ASSERT(nearest);
            if (i % 10 == 0) {
                LOG_DEBUG(<< "Expected nearest = " << expectedNearest[0].second
                          << ", expected distance = " << expectedNearest[0].first);
                LOG_DEBUG(<< "Nearest          = " << *nearest << ", actual distance   = "
                          << (tests[j] - *nearest).euclidean());
            }
            CPPUNIT_ASSERT_EQUAL(print(expectedNearest[0].second), print(*nearest));
        }
    }
}

void CKdTreeTest::testNearestNeighbours() {
    const std::size_t numberTests = 200u;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberTests; ++i) {
        TDoubleVec samples;
        rng.generateUniformSamples(-100.0, 100.0, 5 * 300, samples);

        TVector5Vec points;
        for (std::size_t j = 0u; j < samples.size(); j += 5) {
            points.emplace_back(&samples[j], &samples[j + 5]);
        }

        maths::CKdTree<TVector5> kdTree;
        kdTree.build(points);
        CPPUNIT_ASSERT(kdTree.checkInvariants());

        rng.generateUniformSamples(-100.0, 100.0, 5 * 10, samples);

        TVector5Vec tests;
        for (std::size_t j = 0u; j < samples.size(); j += 5) {
            tests.emplace_back(&samples[j], &samples[j + 5]);
        }

        if (i % 10 == 0) {
            LOG_DEBUG(<< "*** Test " << i << " ***");
        }
        for (std::size_t j = 0u; j < tests.size(); ++j) {
            TDoubleVector5PrVec expectedNeighbours;
            nearestNeightbours(2 * j, points, tests[j], expectedNeighbours);

            TVector5Vec neighbours;
            kdTree.nearestNeighbours(2 * j, tests[j], neighbours);
            CPPUNIT_ASSERT_EQUAL(expectedNeighbours.size(), neighbours.size());
            for (std::size_t k = 0u; k < expectedNeighbours.size(); ++k) {
                if (i % 10 == 0) {
                    LOG_DEBUG(<< "Expected nearest = " << expectedNeighbours[k].second
                              << ", expected distance = "
                              << expectedNeighbours[k].first);
                    LOG_DEBUG(<< "Nearest          = " << neighbours[k] << ", actual distance   = "
                              << (tests[j] - neighbours[k]).euclidean());
                }
                CPPUNIT_ASSERT_EQUAL(print(expectedNeighbours[k].second),
                                     print(neighbours[k]));
            }
        }
    }
}

CppUnit::Test* CKdTreeTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CKdTreeTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CKdTreeTest>(
        "CKdTreeTest::testBuild", &CKdTreeTest::testBuild));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKdTreeTest>(
        "CKdTreeTest::testBuildWithMove", &CKdTreeTest::testBuildWithMove));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKdTreeTest>(
        "CKdTreeTest::testNearestNeighbour", &CKdTreeTest::testNearestNeighbour));
    suiteOfTests->addTest(new CppUnit::TestCaller<CKdTreeTest>(
        "CKdTreeTest::testNearestNeighbours", &CKdTreeTest::testNearestNeighbours));

    return suiteOfTests;
}
