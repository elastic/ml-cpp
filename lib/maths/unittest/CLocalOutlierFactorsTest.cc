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

template<typename VECTOR>
void gaussianWithUniformNoiseForPresizedPoints(test::CRandomNumbers& rng,
                                               std::size_t numberInliers,
                                               std::size_t numberOutliers,
                                               std::vector<VECTOR>& points) {

    CPPUNIT_ASSERT_EQUAL(numberInliers + numberOutliers, points.size());

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
    for (std::size_t i = 100, j = 0; j < outliers.size(); ++i) {
        for (std::size_t end = j + 6; j < end; ++j) {
            points[i](j % 6) = outliers[j];
        }
    }
}

void gaussianWithUniformNoise(test::CRandomNumbers& rng,
                              std::size_t numberInliers,
                              std::size_t numberOutliers,
                              TVectorVec& points) {
    points.assign(120, TVector(6));
    gaussianWithUniformNoiseForPresizedPoints(rng, numberInliers, numberOutliers, points);
}

template<typename VECTOR>
void outlierErrorStatisticsForEnsemble(test::CRandomNumbers& rng,
                                       std::size_t numberInliers,
                                       std::size_t numberOutliers,
                                       std::vector<VECTOR>& points,
                                       TDoubleVec& TP,
                                       TDoubleVec& TN,
                                       TDoubleVec& FP,
                                       TDoubleVec& FN) {

    TP.assign(3, 0.0);
    TN.assign(3, 0.0);
    FP.assign(3, 0.0);
    FN.assign(3, 0.0);

    TSizeVec trueOutliers{100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119};

    for (std::size_t t = 0; t < 100; ++t) {
        gaussianWithUniformNoiseForPresizedPoints(rng, numberInliers, numberOutliers, points);

        maths::CLocalOutlierFactors lofs;

        TDoubleVec scores;
        lofs.ensemble(points, scores);

        TSizeVec outliers[3];
        for (std::size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] >= 0.1) {
                outliers[0].push_back(i);
            }
            if (scores[i] >= 1.0) {
                outliers[1].push_back(i);
            }
            if (scores[i] >= 10.0) {
                outliers[2].push_back(i);
            }
        }

        if (t % 10 == 0) {
            LOG_DEBUG(<< "outliers at 0.1  = "
                      << core::CContainerPrinter::print(outliers[0]));
            LOG_DEBUG(<< "outliers at 1.0  = "
                      << core::CContainerPrinter::print(outliers[1]));
            LOG_DEBUG(<< "outliers at 10.0 = "
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

    LOG_DEBUG(<< "At 0.1:  TP = " << TP[0] << " TN = " << TN[0]
              << " FP = " << FP[0] << " FN = " << FN[0]);
    LOG_DEBUG(<< "At 1.0:  TP = " << TP[1] << " TN = " << TN[1]
              << " FP = " << FP[1] << " FN = " << FN[1]);
    LOG_DEBUG(<< "At 10.0: TP = " << TP[2] << " TN = " << TN[2]
              << " FP = " << FP[2] << " FN = " << FN[2]);
}
}

void CLocalOutlierFactorsTest::testLof() {
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
        maths::CLocalOutlierFactors lofs;

        TDoubleVec scores;
        lofs.normalizedLof(k, false, points, scores);

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

void CLocalOutlierFactorsTest::testDlof() {
    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    maths::CLocalOutlierFactors lofs;

    TDoubleVec scores;
    std::size_t k{10};
    lofs.normalizedLdof(k, false, points, scores);
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
    CLocalOutlierFactorsInternals::normalize(ldof);
    LOG_DEBUG(<< "normalized ldof = " << core::CContainerPrinter::print(ldof));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(ldof[i], scores[i], 1e-6);
    }

    // Compare outliers when projecting (should be similar).

    lofs.normalizedLdof(k, true, points, scores);

    TMaxAccumulator outlierScoresProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresProjecting.add({scores[i], i});
    }

    TSizeVec outliersWithoutProjecting;
    TSizeVec outliersProjecting;
    for (std::size_t i = 0; i < numberOutliers; ++i) {
        outliersWithoutProjecting.push_back(outlierScoresWithoutProjecting[i].second);
        outliersProjecting.push_back(outlierScoresProjecting[i].second);
    }

    std::sort(outliersWithoutProjecting.begin(), outliersWithoutProjecting.end());
    std::sort(outliersProjecting.begin(), outliersProjecting.end());
    LOG_DEBUG(<< "without projecting = "
              << core::CContainerPrinter::print(outliersWithoutProjecting));
    LOG_DEBUG(<< "projecting         = "
              << core::CContainerPrinter::print(outliersProjecting));

    double similarity{maths::CSetTools::jaccard(
        outliersWithoutProjecting.begin(), outliersWithoutProjecting.end(),
        outliersProjecting.begin(), outliersProjecting.end())};

    CPPUNIT_ASSERT(similarity > 0.4);
}

void CLocalOutlierFactorsTest::testDistancekNN() {
    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    maths::CLocalOutlierFactors lofs;

    TDoubleVec scores;
    std::size_t k{10};
    lofs.normalizedDistancekNN(k, false, points, scores);
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
    CLocalOutlierFactorsInternals::normalize(distances);
    LOG_DEBUG(<< "normalized distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(distances[i], scores[i], 1e-6);
    }

    // Compare outliers when projecting (should be similar).

    lofs.normalizedDistancekNN(k, true, points, scores);

    TMaxAccumulator outlierScoresProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresProjecting.add({scores[i], i});
    }

    TSizeVec outliersWithoutProjecting;
    TSizeVec outliersProjecting;
    for (std::size_t i = 0; i < numberOutliers; ++i) {
        outliersWithoutProjecting.push_back(outlierScoresWithoutProjecting[i].second);
        outliersProjecting.push_back(outlierScoresProjecting[i].second);
    }

    std::sort(outliersWithoutProjecting.begin(), outliersWithoutProjecting.end());
    std::sort(outliersProjecting.begin(), outliersProjecting.end());
    LOG_DEBUG(<< "without projecting = "
              << core::CContainerPrinter::print(outliersWithoutProjecting));
    LOG_DEBUG(<< "projecting         = "
              << core::CContainerPrinter::print(outliersProjecting));

    double similarity{maths::CSetTools::jaccard(
        outliersWithoutProjecting.begin(), outliersWithoutProjecting.end(),
        outliersProjecting.begin(), outliersProjecting.end())};
    CPPUNIT_ASSERT(similarity > 0.9);
}

void CLocalOutlierFactorsTest::testTotalDistancekNN() {
    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TVectorVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    maths::CLocalOutlierFactors lofs;

    TDoubleVec scores;
    std::size_t k{10};
    lofs.normalizedTotalDistancekNN(k, false, points, scores);
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
                            }));
    }
    CLocalOutlierFactorsInternals::normalize(distances);
    LOG_DEBUG(<< "normalized distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(distances[i], scores[i], 1e-6);
    }

    // Compare outliers when projecting (should be similar).

    lofs.normalizedTotalDistancekNN(k, true, points, scores);

    TMaxAccumulator outlierScoresProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresProjecting.add({scores[i], i});
    }

    TSizeVec outliersWithoutProjecting;
    TSizeVec outliersProjecting;
    for (std::size_t i = 0; i < numberOutliers; ++i) {
        outliersWithoutProjecting.push_back(outlierScoresWithoutProjecting[i].second);
        outliersProjecting.push_back(outlierScoresProjecting[i].second);
    }

    std::sort(outliersWithoutProjecting.begin(), outliersWithoutProjecting.end());
    std::sort(outliersProjecting.begin(), outliersProjecting.end());
    LOG_DEBUG(<< "without projecting = "
              << core::CContainerPrinter::print(outliersWithoutProjecting));
    LOG_DEBUG(<< "projecting         = "
              << core::CContainerPrinter::print(outliersProjecting));

    double similarity{maths::CSetTools::jaccard(
        outliersWithoutProjecting.begin(), outliersWithoutProjecting.end(),
        outliersProjecting.begin(), outliersProjecting.end())};
    CPPUNIT_ASSERT(similarity > 0.9);
}

void CLocalOutlierFactorsTest::testEnsemble() {
    // Check error stats for scores, 0.1, 1.0 and 10.0. We should see precision increase
    // for higher scores but recall decrease.
    //
    // In practice, the samples are randomly generated so it isn't necessarily the case
    // that those generated from the different process are the outliers, they simply have
    // a much higher probability of this being the case.

    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};

    TDoubleVec TP;
    TDoubleVec TN;
    TDoubleVec FP;
    TDoubleVec FN;
    double precisionLowerBounds[]{0.8, 0.85, 0.95};
    double recallLowerBounds[]{0.86, 0.7, 0.21};

    // Test sequential then parallel.

    core::stopDefaultAsyncExecutor();

    std::string tags[]{"sequential", "parallel"};

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t]);

        test::CRandomNumbers rng;

        TVectorVec points(numberInliers + numberOutliers, TVector(6));

        outlierErrorStatisticsForEnsemble(rng, numberInliers, numberOutliers,
                                          points, TP, TN, FP, FN);

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

    for (std::size_t t = 0; t < 2; ++t) {
        LOG_DEBUG(<< "Testing " << tags[t]);

        using TMemoryMappedVector = maths::CMemoryMappedDenseVector<double>;
        using TMemoryMappedVectorVec = std::vector<TMemoryMappedVector>;

        test::CRandomNumbers rng;

        TDoubleVec storage((numberInliers + numberOutliers) * 6);
        TMemoryMappedVectorVec points;
        for (std::size_t i = 0; i < numberInliers + numberOutliers; ++i) {
            points.emplace_back(&storage[6 * i], 6);
        }

        outlierErrorStatisticsForEnsemble(rng, numberInliers, numberOutliers,
                                          points, TP, TN, FP, FN);
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

void CLocalOutlierFactorsTest::testProgressMonitoring() {

    // Test progress monitoring invariants.

    std::size_t numberInliers{100000};
    std::size_t numberOutliers{500};

    test::CRandomNumbers rng;

    TVectorVec points(numberInliers + numberOutliers, TVector(6));

    std::atomic_int totalFractionalProgress{0};

    auto reportProgress = [&totalFractionalProgress](double fractionalProgress) {
        totalFractionalProgress.fetch_add(static_cast<int>(1024.0 * fractionalProgress + 0.5));
    };

    std::atomic_bool finished{false};

    std::thread worker{[&](TVectorVec points_) {
                           maths::CLocalOutlierFactors lofs{reportProgress};
                           TDoubleVec scores;
                           lofs.ensemble(points_, scores);
                           finished.store(true);
                       },
                       std::move(points)};

    int lastTotalFractionalProgress{0};
    int lastProgressReport{0};

    bool monotonic{true};
    while (finished.load() == false) {
        if (totalFractionalProgress.load() > lastProgressReport) {
            LOG_DEBUG(<< (static_cast<double>(lastProgressReport) / 10) << "% complete");
            lastProgressReport += 100;
        }
        monotonic &= (totalFractionalProgress.load() >= lastTotalFractionalProgress);
        lastTotalFractionalProgress = totalFractionalProgress.load();
    }
    worker.join();

    CPPUNIT_ASSERT(monotonic);
    CPPUNIT_ASSERT_EQUAL(1024, totalFractionalProgress.load());
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
    suiteOfTests->addTest(new CppUnit::TestCaller<CLocalOutlierFactorsTest>(
        "CLocalOutlierFactorsTest::testProgressMonitoring",
        &CLocalOutlierFactorsTest::testProgressMonitoring));

    return suiteOfTests;
}
