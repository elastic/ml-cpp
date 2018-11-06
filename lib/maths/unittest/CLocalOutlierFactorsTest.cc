/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CLocalOutlierFactorsTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CLocalOutlierFactors.h>
#include <maths/CSetTools.h>

#include <test/CRandomNumbers.h>

#include <numeric>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TDoubleSizePr = std::pair<double, std::size_t>;
using TMaxAccumulator =
    maths::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;
using TVector = maths::CDenseVector<double>;
using TVectorVec = std::vector<TVector>;

class CLocalOutlierFactorsInternals : public maths::CLocalOutlierFactors {
public:
    static void normalize(TDoubleVec& scores) {
        maths::CLocalOutlierFactors::normalize(scores);
    }
};

void nearestNeightbours(std::size_t k, const TVectorVec& points, const TVector& point, TVectorVec& result) {
    using TDoubleVectorPr = std::pair<double, TVector>;
    using TMinDoubleVectorPrAccumulator =
        maths::CBasicStatistics::COrderStatisticsHeap<TDoubleVectorPr>;

    result.clear();
    result.reserve(k);
    TMinDoubleVectorPrAccumulator result_(k);
    for (const auto& point_ : points) {
        if (&point != &point_) {
            result_.add({maths::las::distance(point, point_), point_});
        }
    }
    result_.sort();
    for (const auto& neighbour : result_) {
        result.push_back(neighbour.second);
    }
}

void gaussianWithUniformNoise(test::CRandomNumbers& rng, TVectorVec& points) {
    TDoubleVec mean{1.0, 10.0, 4.0, 8.0, 3.0, 5.0};
    TDoubleVecVec covariance{
        {1.0, 0.1, -0.1, 0.3, 0.2, -0.1}, {0.1, 1.3, -0.3, 0.1, 0.1, 0.1},
        {-0.1, -0.3, 2.1, 0.1, 0.2, 0.1}, {0.3, 0.1, 0.1, 0.8, 0.2, -0.2},
        {0.2, 0.1, 0.2, 0.2, 2.2, -0.1},  {-0.1, 0.1, 0.1, -0.2, -0.1, 3.1}};

    TDoubleVecVec inliers;
    rng.generateMultivariateNormalSamples(mean, covariance, 100, inliers);

    TDoubleVec outliers;
    rng.generateUniformSamples(0.0, 10.0, 20 * 6, outliers);

    points.assign(120, TVector(6));
    for (std::size_t i = 0u; i < inliers.size(); ++i) {
        for (std::size_t j = 0u; j < 6; ++j) {
            points[i](j) = inliers[i][j];
        }
    }
    for (std::size_t i = 100u, j = 0u; j < outliers.size(); ++i) {
        for (std::size_t end = j + 6; j < end; ++j) {
            points[i](j % 6) = outliers[j];
        }
    }
}
}

void CLocalOutlierFactorsTest::testLof() {
    // Test vanilla verses sklearn.

    test::CRandomNumbers rng;
    TVectorVec points;
    gaussianWithUniformNoise(rng, points);

    std::string expected[]{"[1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, "
                           "-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,"
                           " 1, -1, -1, -1, -1]",
                           "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, "
                           "-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,"
                           " 1, -1, -1, -1, -1]",
                           "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,"
                           " 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, "
                           "-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, "
                           "-1, -1, -1, -1, -1]"};
    for (auto k : {5, 10, 15}) {
        TDoubleVec scores;
        maths::CLocalOutlierFactors::normalizedLof(k, false, points, scores);

        TMaxAccumulator outliers_(20);
        for (std::size_t i = 0u; i < scores.size(); ++i) {
            outliers_.add({scores[i], i});
        }
        TSizeVec outliers(20);
        std::transform(outliers_.begin(), outliers_.end(), outliers.begin(),
                       [](const TDoubleSizePr& value) { return value.second; });
        std::sort(outliers.begin(), outliers.end());
        LOG_DEBUG(<< "outliers = " << core::CContainerPrinter::print(outliers));
        TDoubleVec indicator(120, 1);
        for (auto outlier : outliers) {
            indicator[outlier] = -1;
        }
        CPPUNIT_ASSERT_EQUAL(expected[k / 5 - 1], core::CContainerPrinter::print(indicator));
    }
}

void CLocalOutlierFactorsTest::testDlof() {
    test::CRandomNumbers rng;
    TVectorVec points;
    gaussianWithUniformNoise(rng, points);

    TDoubleVec scores;
    maths::CLocalOutlierFactors::normalizedLdof(20, false, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TDoubleVec ldof;
    TVectorVec neighbours;
    for (const auto& point : points) {
        nearestNeightbours(20, points, point, neighbours);
        TMeanAccumulator d, D;
        for (std::size_t i = 0u; i < neighbours.size(); ++i) {
            d.add(maths::las::distance(point, neighbours[i]));
            for (std::size_t j = 0u; j < i; ++j) {
                D.add(maths::las::distance(neighbours[i], neighbours[j]));
            }
        }
        ldof.push_back(maths::CBasicStatistics::mean(d) / maths::CBasicStatistics::mean(D));
    }
    CLocalOutlierFactorsInternals::normalize(ldof);
    LOG_DEBUG(<< "normalized ldof = " << core::CContainerPrinter::print(ldof));

    for (std::size_t i = 0u; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(ldof[i], scores[i], 1e-6);
    }
}

void CLocalOutlierFactorsTest::testDistancekNN() {
    // Gaussian with uniform noise.

    test::CRandomNumbers rng;
    TVectorVec points;
    gaussianWithUniformNoise(rng, points);

    TDoubleVec scores;
    maths::CLocalOutlierFactors::normalizedDistancekNN(3, false, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TDoubleVec distances;
    for (const auto& point : points) {
        TVectorVec neighbours;
        nearestNeightbours(3, points, point, neighbours);
        distances.push_back(maths::las::distance(point, neighbours.back()));
    }
    CLocalOutlierFactorsInternals::normalize(distances);
    LOG_DEBUG(<< "normalized distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0u; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(distances[i], scores[i], 1e-6);
    }

    // Test projection.
}

void CLocalOutlierFactorsTest::testTotalDistancekNN() {
    // Gaussian with uniform noise.

    test::CRandomNumbers rng;
    TVectorVec points;
    gaussianWithUniformNoise(rng, points);

    TDoubleVec scores;
    maths::CLocalOutlierFactors::normalizedTotalDistancekNN(3, false, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TDoubleVec distances;
    for (const auto& point : points) {
        TVectorVec neighbours;
        nearestNeightbours(3, points, point, neighbours);
        distances.push_back(
            std::accumulate(neighbours.begin(), neighbours.end(), 0.0,
                            [&point](double total, const TVector& neighbour) {
                                return total + maths::las::distance(point, neighbour);
                            }));
    }
    CLocalOutlierFactorsInternals::normalize(distances);
    LOG_DEBUG(<< "normalized distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0u; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(distances[i], scores[i], 1e-6);
    }
}

void CLocalOutlierFactorsTest::testEnsemble() {
    test::CRandomNumbers rng;

    // Check error stats for scores, 0.05, 0.1 and 0.5. We should see precision increase
    // for higher scores but recall decrease.
    //
    // In practice, the samples are randomly generated so it isn't necessarily the case
    // that those generated from the different process are the outliers, they simply have
    // a much higher probability of this being the case.

    double TP[]{0.0, 0.0, 0.0};
    double TN[]{0.0, 0.0, 0.0};
    double FP[]{0.0, 0.0, 0.0};
    double FN[]{0.0, 0.0, 0.0};

    TSizeVec trueOutliers{100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119};

    for (std::size_t t = 0u; t < 100; ++t) {
        TVectorVec points;
        gaussianWithUniformNoise(rng, points);

        TDoubleVec scores;
        maths::CLocalOutlierFactors::ensemble(points, scores);

        TMaxAccumulator top20Outliers(20);
        TSizeVec outliers[3];
        for (std::size_t i = 0u; i < scores.size(); ++i) {
            top20Outliers.add({scores[i], i});
            if (scores[i] >= 0.01) {
                outliers[0].push_back(i);
            }
            if (scores[i] >= 0.1) {
                outliers[1].push_back(i);
            }
            if (scores[i] >= 0.5) {
                outliers[2].push_back(i);
            }
        }

        if (t % 10 == 0) {
            LOG_DEBUG(<< "outliers at 0.01 = "
                      << core::CContainerPrinter::print(outliers[0]));
            LOG_DEBUG(<< "outliers at 0.1  = "
                      << core::CContainerPrinter::print(outliers[1]));
            LOG_DEBUG(<< "outliers at 0.5  = "
                      << core::CContainerPrinter::print(outliers[2]));
        }

        for (std::size_t i = 0; i < 3; ++i) {
            double correct{static_cast<double>(maths::CSetTools::setIntersectSize(
                trueOutliers.begin(), trueOutliers.end(), outliers[i].begin(),
                outliers[i].end()))};
            double total{static_cast<double>(outliers[i].size())};
            TP[i] += correct;
            TN[i] += 100.0 - total + correct;
            FP[i] += total - correct;
            FN[i] += 20.0 - correct;
        }
    }

    LOG_DEBUG(<< "At 0.01: TP = " << TP[0] << " TN = " << TN[0]
              << " FP = " << FP[0] << " FN = " << FN[0]);
    LOG_DEBUG(<< "At 0.1:  TP = " << TP[1] << " TN = " << TN[1]
              << " FP = " << FP[1] << " FN = " << FN[1]);
    LOG_DEBUG(<< "At 0.5:  TP = " << TP[2] << " TN = " << TN[2]
              << " FP = " << FP[2] << " FN = " << FN[2]);

    double precisionLowerBounds[]{0.95, 0.98, 1.0};
    double recallLowerBounds[]{0.75, 0.38, 0.11};
    for (std::size_t i = 0; i < 3; ++i) {
        double precision{TP[i] / (TP[i] + FP[i])};
        double recall{TP[i] / (TP[i] + FN[i])};
        LOG_DEBUG(<< "precision = " << precision);
        LOG_DEBUG(<< "recall = " << recall);
        CPPUNIT_ASSERT(precision >= precisionLowerBounds[i]);
        CPPUNIT_ASSERT(recall >= recallLowerBounds[i]);
    }
}

CppUnit::Test* CLocalOutlierFactorsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CLocalOutlierFactorsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CLocalOutlierFactorsTest>(
        "CLocalOutlierFactorsTest::testLof", &CLocalOutlierFactorsTest::testLof));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLocalOutlierFactorsTest>(
        "CLocalOutlierFactorsTest::testDlof", &CLocalOutlierFactorsTest::testDlof));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLocalOutlierFactorsTest>(
        "CLocalOutlierFactorsTest::testDistancekNN", &CLocalOutlierFactorsTest::testDistancekNN));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLocalOutlierFactorsTest>(
        "CLocalOutlierFactorsTest::testTotalDistancekNN",
        &CLocalOutlierFactorsTest::testTotalDistancekNN));
    suiteOfTests->addTest(new CppUnit::TestCaller<CLocalOutlierFactorsTest>(
        "CLocalOutlierFactorsTest::testEnsemble", &CLocalOutlierFactorsTest::testEnsemble));

    return suiteOfTests;
}
