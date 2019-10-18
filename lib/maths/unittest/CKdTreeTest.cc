/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/COrderings.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CKdTreeTest)

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

BOOST_AUTO_TEST_CASE(testBuild) {

    const std::size_t numberTests{200};

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
        BOOST_TEST(kdTree.checkInvariants());
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
        BOOST_TEST(kdTree.checkInvariants());
    }
}

BOOST_AUTO_TEST_CASE(testBuildWithMove) {

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
    BOOST_TEST(moveCounts[1] > moveCounts[0] + points.size());
}

BOOST_AUTO_TEST_CASE(testNearestNeighbour) {

    const std::size_t numberTests{200};

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
        BOOST_TEST(kdTree.checkInvariants());

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
            BOOST_TEST(nearest);
            if (i % 10 == 0) {
                LOG_DEBUG(<< "Expected nearest = " << expectedNearest[0].second
                          << ", expected distance = " << expectedNearest[0].first);
                LOG_DEBUG(<< "Nearest          = " << *nearest << ", actual distance   = "
                          << (tests[j] - *nearest).euclidean());
            }
            BOOST_CHECK_EQUAL(print(expectedNearest[0].second), print(*nearest));
        }
    }
}

BOOST_AUTO_TEST_CASE(testNearestNeighbours) {

    const std::size_t numberTests{200};

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
        BOOST_TEST(kdTree.checkInvariants());

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
            BOOST_CHECK_EQUAL(expectedNeighbours.size(), neighbours.size());
            for (std::size_t k = 0u; k < expectedNeighbours.size(); ++k) {
                if (i % 10 == 0) {
                    LOG_DEBUG(<< "Expected nearest = " << expectedNeighbours[k].second
                              << ", expected distance = "
                              << expectedNeighbours[k].first);
                    LOG_DEBUG(<< "Nearest          = " << neighbours[k] << ", actual distance   = "
                              << (tests[j] - neighbours[k]).euclidean());
                }
                BOOST_CHECK_EQUAL(print(expectedNeighbours[k].second),
                                  print(neighbours[k]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testRequestingEveryPoint) {

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(-100.0, 100.0, 5 * 5, samples);

    TVector5Vec points;
    for (std::size_t j = 0u; j < samples.size(); j += 5) {
        points.emplace_back(&samples[j], &samples[j + 5]);
    }
    std::stable_sort(points.begin(), points.end());
    std::string expected{core::CContainerPrinter::print(points)};

    maths::CKdTree<TVector5> kdTree;
    kdTree.build(points);
    BOOST_TEST(kdTree.checkInvariants());

    TVector5Vec neighbours;
    kdTree.nearestNeighbours(5, TVector5{0.0}, neighbours);
    std::stable_sort(neighbours.begin(), neighbours.end());

    BOOST_CHECK_EQUAL(kdTree.size(), neighbours.size());
    BOOST_CHECK_EQUAL(expected, core::CContainerPrinter::print(neighbours));
}

BOOST_AUTO_TEST_SUITE_END()
