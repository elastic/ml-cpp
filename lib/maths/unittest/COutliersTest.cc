/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "COutliersTest.h"

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/COutliers.h>
#include <maths/CSetTools.h>

#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>

#include <atomic>
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
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>(const TVectorVec&)>;

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

void gaussianWithUniformNoise(test::CRandomNumbers& rng,
                              std::size_t numberInliers,
                              std::size_t numberOutliers,
                              TVectorVec& points) {
    points.assign(numberInliers + numberOutliers, TVector(6));

    TDoubleVec mean{1.0, 10.0, 4.0, 8.0, 3.0, 5.0};
    TDoubleVecVec covariance{
        {1.0, 0.1, -0.1, 0.3, 0.2, -0.1}, {0.1, 1.3, -0.3, 0.1, 0.1, 0.1},
        {-0.1, -0.3, 2.1, 0.1, 0.2, 0.1}, {0.3, 0.1, 0.1, 0.8, 0.2, -0.2},
        {0.2, 0.1, 0.2, 0.2, 2.2, -0.1},  {-0.1, 0.1, 0.1, -0.2, -0.1, 3.1}};

    TDoubleVecVec inliers;
    rng.generateMultivariateNormalSamples(mean, covariance, numberInliers, inliers);

    TDoubleVec outliers;
    rng.generateUniformSamples(0.0, 10.0, numberOutliers * 6, outliers);

    for (std::size_t i = 0; i < inliers.size(); ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            points[i](j) = inliers[i][j];
        }
    }
    for (std::size_t i = inliers.size(), j = 0; j < outliers.size(); ++i) {
        for (std::size_t end = j + 6; j < end; ++j) {
            points[i](j % 6) = outliers[j];
        }
    }
}

template<typename FACTORY>
void outlierErrorStatisticsForEnsemble(FACTORY pointsToDataFrame,
                                       std::size_t numberInliers,
                                       std::size_t numberOutliers,
                                       TDoubleVec& TP,
                                       TDoubleVec& TN,
                                       TDoubleVec& FP,
                                       TDoubleVec& FN) {
    test::CRandomNumbers rng;

    TP.assign(3, 0.0);
    TN.assign(3, 0.0);
    FP.assign(3, 0.0);
    FN.assign(3, 0.0);
    TSizeVec trueOutliers(numberOutliers, 0);
    std::iota(trueOutliers.begin(), trueOutliers.end(), numberInliers);

    TVectorVec points;
    TDoubleVec scores(numberInliers + numberOutliers);

    for (std::size_t t = 0; t < 100; ++t) {
        gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

        auto dataFrame = pointsToDataFrame(points);

        maths::COutliers::compute(1, *dataFrame);

        dataFrame->readRows(1, [&scores](core::CDataFrame::TRowItr beginRows,
                                         core::CDataFrame::TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                scores[row->index()] = (*row)[6];
            }
        });

        TSizeVec outliers[3];
        for (std::size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] >= 0.1) {
                outliers[0].push_back(i);
            }
            if (scores[i] >= 0.5) {
                outliers[1].push_back(i);
            }
            if (scores[i] >= 0.9) {
                outliers[2].push_back(i);
            }
        }

        if (t % 10 == 0) {
            LOG_DEBUG(<< "outliers at 0.1 = "
                      << core::CContainerPrinter::print(outliers[0]));
            LOG_DEBUG(<< "outliers at 0.5 = "
                      << core::CContainerPrinter::print(outliers[1]));
            LOG_DEBUG(<< "outliers at 0.9 = "
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

    LOG_DEBUG(<< "At 0.1: TP = " << TP[0] << " TN = " << TN[0]
              << " FP = " << FP[0] << " FN = " << FN[0]);
    LOG_DEBUG(<< "At 0.5: TP = " << TP[1] << " TN = " << TN[1]
              << " FP = " << FP[1] << " FN = " << FN[1]);
    LOG_DEBUG(<< "At 0.9: TP = " << TP[2] << " TN = " << TN[2]
              << " FP = " << FP[2] << " FN = " << FN[2]);
}
}

void COutliersTest::testLof() {
    // Test vanilla verses sklearn.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

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
        maths::COutliers::lof(k, points, scores);

        TMaxAccumulator outliers_(numberOutliers);
        for (std::size_t i = 0u; i < scores.size(); ++i) {
            outliers_.add({scores[i], i});
        }
        TSizeVec outliers(numberOutliers);
        std::transform(outliers_.begin(), outliers_.end(), outliers.begin(),
                       [](const TDoubleSizePr& value) { return value.second; });
        std::sort(outliers.begin(), outliers.end());
        LOG_DEBUG(<< "outliers = " << core::CContainerPrinter::print(outliers));
        TDoubleVec indicator(numberInliers + numberOutliers, 1);
        for (auto outlier : outliers) {
            indicator[outlier] = -1;
        }
        CPPUNIT_ASSERT_EQUAL(expected[k / 5 - 1], core::CContainerPrinter::print(indicator));
    }
}

void COutliersTest::testDlof() {
    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    TDoubleVec scores;
    std::size_t k{10};
    maths::COutliers::ldof(k, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TMaxAccumulator outlierScoresWithoutProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresWithoutProjecting.add({scores[i], i});
    }

    TDoubleVec ldof;
    TVectorVec neighbours;
    for (const auto& point : points) {
        nearestNeightbours(k, points, point, neighbours);
        TMeanAccumulator d, D;
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            d.add(maths::las::distance(point, neighbours[i]));
            for (std::size_t j = 0; j < i; ++j) {
                D.add(maths::las::distance(neighbours[i], neighbours[j]));
            }
        }
        ldof.push_back(maths::CBasicStatistics::mean(d) / maths::CBasicStatistics::mean(D));
    }
    LOG_DEBUG(<< "ldof = " << core::CContainerPrinter::print(ldof));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(ldof[i], scores[i], 1e-5);
    }
}

void COutliersTest::testDistancekNN() {
    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    TDoubleVec scores;
    std::size_t k{10};
    maths::COutliers::distancekNN(k, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TMaxAccumulator outlierScoresWithoutProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresWithoutProjecting.add({scores[i], i});
    }

    TDoubleVec distances;
    for (const auto& point : points) {
        TVectorVec neighbours;
        nearestNeightbours(k, points, point, neighbours);
        distances.push_back(maths::las::distance(point, neighbours.back()));
    }
    LOG_DEBUG(<< "distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(distances[i], scores[i], 1e-5);
    }
}

void COutliersTest::testTotalDistancekNN() {
    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    TDoubleVec scores;
    std::size_t k{10};
    maths::COutliers::totalDistancekNN(k, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TMaxAccumulator outlierScoresWithoutProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresWithoutProjecting.add({scores[i], i});
    }

    TDoubleVec distances;
    for (const auto& point : points) {
        TVectorVec neighbours;
        nearestNeightbours(k, points, point, neighbours);
        distances.push_back(
            std::accumulate(neighbours.begin(), neighbours.end(), 0.0,
                            [&point](double total, const TVector& neighbour) {
                                return total + maths::las::distance(point, neighbour);
                            }) /
            static_cast<double>(k));
    }
    LOG_DEBUG(<< "distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(distances[i], scores[i], 1e-5);
    }
}

void COutliersTest::testEnsemble() {
    // Check error stats for scores, 0.1, 0.5 and 0.9. We should see precision increase
    // for higher scores but recall decrease.
    //
    // In practice, the samples are randomly generated so it isn't necessarily the case
    // that those generated from the different process are the outliers, they simply have
    // a much higher probability of this being the case.

    // TODO test outlier detection with and without partitioning are equivalent.

    std::size_t numberInliers{400};
    std::size_t numberOutliers{20};

    TDoubleVec TP;
    TDoubleVec TN;
    TDoubleVec FP;
    TDoubleVec FN;
    double precisionLowerBounds[]{0.23, 0.85, 0.98};
    double recallLowerBounds[]{0.98, 0.92, 0.62};

    // Test sequential then parallel.

    std::string tags[]{"sequential", "parallel"};

    core::stopDefaultAsyncExecutor();

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t]);

        outlierErrorStatisticsForEnsemble(
            test::CDataFrameTestUtils::SToMainMemoryDataFrame(), numberInliers,
            numberOutliers, TP, TN, FP, FN);

        for (std::size_t i = 0; i < 3; ++i) {
            double precision{TP[i] / (TP[i] + FP[i])};
            double recall{TP[i] / (TP[i] + FN[i])};
            LOG_DEBUG(<< "precision = " << precision);
            LOG_DEBUG(<< "recall = " << recall);
            CPPUNIT_ASSERT(precision >= precisionLowerBounds[i]);
            CPPUNIT_ASSERT(recall >= recallLowerBounds[i]);
        }

        core::startDefaultAsyncExecutor();
    }

    core::stopDefaultAsyncExecutor();
}

void COutliersTest::testProgressMonitoring() {

    // Test progress monitoring invariants.

    // TODO test outlier detection with and without partitioning are equivalent.

    std::size_t numberInliers{10000};
    std::size_t numberOutliers{500};

    test::CRandomNumbers rng;

    TVectorVec points(numberInliers + numberOutliers, TVector(6));
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    auto dataFrame = test::CDataFrameTestUtils::toMainMemoryDataFrame(points);

    std::atomic_int totalFractionalProgress{0};

    auto reportProgress = [&totalFractionalProgress](double fractionalProgress) {
        totalFractionalProgress.fetch_add(static_cast<int>(65536.0 * fractionalProgress + 0.5));
    };

    std::atomic_bool finished{false};

    std::thread worker{[&]() {
        maths::COutliers::compute(1, *dataFrame, reportProgress);
        finished.store(true);
    }};

    int lastTotalFractionalProgress{0};
    int lastProgressReport{0};

    bool monotonic{true};
    std::size_t percentage{0};
    while (finished.load() == false) {
        if (totalFractionalProgress.load() > lastProgressReport) {
            LOG_DEBUG(<< percentage << "% complete");
            percentage += 10;
            lastProgressReport += 6554;
        }
        monotonic &= (totalFractionalProgress.load() >= lastTotalFractionalProgress);
        lastTotalFractionalProgress = totalFractionalProgress.load();
    }
    worker.join();

    CPPUNIT_ASSERT(monotonic);
    CPPUNIT_ASSERT(std::fabs(65536 - totalFractionalProgress.load()) < 100);
}

CppUnit::Test* COutliersTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("COutliersTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<COutliersTest>(
        "COutliersTest::testLof", &COutliersTest::testLof));
    suiteOfTests->addTest(new CppUnit::TestCaller<COutliersTest>(
        "COutliersTest::testDlof", &COutliersTest::testDlof));
    suiteOfTests->addTest(new CppUnit::TestCaller<COutliersTest>(
        "COutliersTest::testDistancekNN", &COutliersTest::testDistancekNN));
    suiteOfTests->addTest(new CppUnit::TestCaller<COutliersTest>(
        "COutliersTest::testTotalDistancekNN", &COutliersTest::testTotalDistancekNN));
    suiteOfTests->addTest(new CppUnit::TestCaller<COutliersTest>(
        "COutliersTest::testEnsemble", &COutliersTest::testEnsemble));
    suiteOfTests->addTest(new CppUnit::TestCaller<COutliersTest>(
        "COutliersTest::testProgressMonitoring", &COutliersTest::testProgressMonitoring));

    return suiteOfTests;
}
