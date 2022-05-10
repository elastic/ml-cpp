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

#include <core/CAlignment.h>
#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>

#include <maths/analytics/CDataFrameAnalysisInstrumentationInterface.h>
#include <maths/analytics/COutliers.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CSetTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameTestUtils.h>
#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/test/unit_test.hpp>

#include <atomic>
#include <numeric>

BOOST_AUTO_TEST_SUITE(COutliersTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TDoubleSizePr = std::pair<double, std::size_t>;
using TMaxAccumulator =
    maths::common::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;
using TPoint = maths::common::CDenseVector<double>;
using TPointVec = std::vector<TPoint>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>(const TPointVec&)>;

class CTestInstrumentation final : public maths::analytics::CDataFrameOutliersInstrumentationStub {
public:
    using TProgressCallbackOpt = boost::optional<TProgressCallback>;
    using TMemoryUsageCallbackOpt = boost::optional<TMemoryUsageCallback>;

public:
    void updateMemoryUsage(std::int64_t delta) override {
        if (m_MemoryUsageCallback) {
            m_MemoryUsageCallback.get()(delta);
        }
    }

    void updateProgress(double d) override {
        if (m_ProgressCallback) {
            m_ProgressCallback.get()(d);
        }
    }

    void progressCallback(const TProgressCallback& progressCallback) {
        m_ProgressCallback = progressCallback;
    }

    void memoryUsageCallback(const TMemoryUsageCallback& memoryUsageCallback) {
        m_MemoryUsageCallback = memoryUsageCallback;
    }

    void flush(const std::string& /*tag*/) override {}

private:
    TProgressCallbackOpt m_ProgressCallback;
    TMemoryUsageCallbackOpt m_MemoryUsageCallback;
};

void nearestNeightbours(std::size_t k, const TPointVec& points, const TPoint& point, TPointVec& result) {
    using TDoubleVectorPr = std::pair<double, TPoint>;
    using TMinDoubleVectorPrAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsHeap<TDoubleVectorPr>;

    result.clear();
    result.reserve(k);
    TMinDoubleVectorPrAccumulator result_(k);
    for (const auto& point_ : points) {
        if (&point != &point_) {
            result_.add({maths::common::las::distance(point, point_), point_});
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
                              TPointVec& points) {
    points.assign(numberInliers + numberOutliers, TPoint(6));

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
void outlierErrorStatisticsForEnsemble(std::size_t numberThreads,
                                       std::size_t numberPartitions,
                                       FACTORY pointsToDataFrame,
                                       std::size_t numberInliers,
                                       std::size_t numberOutliers,
                                       TDoubleVec& TP,
                                       TDoubleVec& TN,
                                       TDoubleVec& FP,
                                       TDoubleVec& FN) {

    LOG_DEBUG(<< "# partitions = " << numberPartitions << " # threads = " << numberThreads);

    test::CRandomNumbers rng;

    TP.assign(3, 0.0);
    TN.assign(3, 0.0);
    FP.assign(3, 0.0);
    FN.assign(3, 0.0);
    TSizeVec trueOutliers(numberOutliers, 0);
    std::iota(trueOutliers.begin(), trueOutliers.end(), numberInliers);

    TPointVec points;
    TDoubleVec scores(numberInliers + numberOutliers);

    CTestInstrumentation instrumentation;

    for (std::size_t t = 0; t < 100; ++t) {
        gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

        auto frame = pointsToDataFrame(points);

        maths::analytics::COutliers::SComputeParameters params{
            numberThreads,
            numberPartitions,
            true, // Standardize columns
            maths::analytics::COutliers::E_Ensemble,
            0,     // Compute number neighbours
            false, // Compute feature influences
            0.05}; // Outlier fraction
        maths::analytics::COutliers::compute(params, *frame, instrumentation);

        frame->readRows(1, [&scores](const core::CDataFrame::TRowItr& beginRows,
                                     const core::CDataFrame::TRowItr& endRows) {
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

        if (t % 20 == 0) {
            LOG_DEBUG(<< "outliers at 0.1 = "
                      << core::CContainerPrinter::print(outliers[0]));
            LOG_DEBUG(<< "outliers at 0.5 = "
                      << core::CContainerPrinter::print(outliers[1]));
            LOG_DEBUG(<< "outliers at 0.9 = "
                      << core::CContainerPrinter::print(outliers[2]));
        }

        for (std::size_t i = 0; i < 3; ++i) {
            double correct{static_cast<double>(maths::common::CSetTools::setIntersectSize(
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

BOOST_AUTO_TEST_CASE(testLof) {
    // Test vanilla verses sklearn.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TPointVec points;
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
        maths::analytics::COutliers::lof(k, points, scores);

        TMaxAccumulator outliers_(numberOutliers);
        for (std::size_t i = 0; i < scores.size(); ++i) {
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
        BOOST_REQUIRE_EQUAL(expected[k / 5 - 1], core::CContainerPrinter::print(indicator));
    }
}

BOOST_AUTO_TEST_CASE(testDlof) {

    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TPointVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    TDoubleVec scores;
    std::size_t k{10};
    maths::analytics::COutliers::ldof(k, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TMaxAccumulator outlierScoresWithoutProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresWithoutProjecting.add({scores[i], i});
    }

    TDoubleVec ldof;
    TPointVec neighbours;
    for (const auto& point : points) {
        nearestNeightbours(k, points, point, neighbours);
        TMeanAccumulator d, D;
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            d.add(maths::common::las::distance(point, neighbours[i]));
            for (std::size_t j = 0; j < i; ++j) {
                D.add(maths::common::las::distance(neighbours[i], neighbours[j]));
            }
        }
        ldof.push_back(maths::common::CBasicStatistics::mean(d) /
                       maths::common::CBasicStatistics::mean(D));
    }
    LOG_DEBUG(<< "ldof = " << core::CContainerPrinter::print(ldof));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(ldof[i], scores[i], 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testDistancekNN) {

    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TPointVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    TDoubleVec scores;
    std::size_t k{10};
    maths::analytics::COutliers::distancekNN(k, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TMaxAccumulator outlierScoresWithoutProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresWithoutProjecting.add({scores[i], i});
    }

    TDoubleVec distances;
    for (const auto& point : points) {
        TPointVec neighbours;
        nearestNeightbours(k, points, point, neighbours);
        distances.push_back(maths::common::las::distance(point, neighbours.back()));
    }
    LOG_DEBUG(<< "distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(distances[i], scores[i], 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testTotalDistancekNN) {

    // Test against definition without projecting.

    test::CRandomNumbers rng;
    std::size_t numberInliers{100};
    std::size_t numberOutliers{20};
    TPointVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    TDoubleVec scores;
    std::size_t k{10};
    maths::analytics::COutliers::totalDistancekNN(k, points, scores);
    LOG_DEBUG(<< "scores = " << core::CContainerPrinter::print(scores));

    TMaxAccumulator outlierScoresWithoutProjecting(numberOutliers);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        outlierScoresWithoutProjecting.add({scores[i], i});
    }

    TDoubleVec distances;
    for (const auto& point : points) {
        TPointVec neighbours;
        nearestNeightbours(k, points, point, neighbours);
        distances.push_back(
            std::accumulate(neighbours.begin(), neighbours.end(), 0.0,
                            [&point](double total, const TPoint& neighbour) {
                                return total + maths::common::las::distance(point, neighbour);
                            }) /
            static_cast<double>(k));
    }
    LOG_DEBUG(<< "distances = " << core::CContainerPrinter::print(distances));

    for (std::size_t i = 0; i < scores.size(); ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(distances[i], scores[i], 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testEnsemble) {

    // Check error stats for scores, 0.1, 0.5 and 0.9. We should see precision increase
    // for higher scores but recall decrease.
    //
    // In practice, the samples are randomly generated so it isn't necessarily the case
    // that those generated from the different process are the outliers, they simply have
    // a much higher probability of this being the case.

    TFactoryFunc toMainMemoryDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toMainMemoryDataFrame(points);
    }};
    TFactoryFunc toOnDiskDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toOnDiskDataFrame(
            test::CTestTmpDir::tmpDir(), points);
    }};
    TFactoryFunc factories[]{toMainMemoryDataFrame, toOnDiskDataFrame};
    std::size_t numberPartitions[]{1, 3};
    std::size_t numberThreads[]{1, 4};

    std::size_t numberInliers{400};
    std::size_t numberOutliers{20};

    TDoubleVec TP;
    TDoubleVec TN;
    TDoubleVec FP;
    TDoubleVec FN;
    double precisionLowerBounds[]{0.27, 0.83, 0.98};
    double recallLowerBounds[]{0.98, 0.93, 0.65};

    core::stopDefaultAsyncExecutor();

    std::string tags[]{"sequential", "parallel"};

    // Test in/out of core.
    for (std::size_t i = 0; i < 2; ++i) {

        // Test sequential then parallel.
        for (std::size_t j = 0; j < 2; ++j) {
            LOG_DEBUG(<< "Testing " << tags[j]);

            outlierErrorStatisticsForEnsemble(numberThreads[j], numberPartitions[i],
                                              factories[i], numberInliers,
                                              numberOutliers, TP, TN, FP, FN);

            for (std::size_t k = 0; k < 3; ++k) {
                double precision{TP[k] / (TP[k] + FP[k])};
                double recall{TP[k] / (TP[k] + FN[k])};
                LOG_DEBUG(<< "precision = " << precision);
                LOG_DEBUG(<< "recall = " << recall);
                BOOST_TEST_REQUIRE(precision >= precisionLowerBounds[k]);
                BOOST_TEST_REQUIRE(recall >= recallLowerBounds[k]);
            }

            core::startDefaultAsyncExecutor();
        }

        core::stopDefaultAsyncExecutor();
    }
}

BOOST_AUTO_TEST_CASE(testFeatureInfluences) {

    // Test calculation of outlier significant features.

    // We have the following basic geometry:
    //   1) Two clusters of points (x, y)
    //   2) Three outliers
    //         i) One displaced in the x-direction
    //        ii) One displaced in the y-direction
    //       iii) One displaced in both directions

    TFactoryFunc toMainMemoryDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toMainMemoryDataFrame(points);
    }};
    TFactoryFunc toOnDiskDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toOnDiskDataFrame(
            test::CTestTmpDir::tmpDir(), points);
    }};
    TFactoryFunc factories[]{toMainMemoryDataFrame, toOnDiskDataFrame};
    TSizeVec numberPartitions{1, 3};
    TSizeVec numberThreads{1, 4};
    TSizeVec dimensions{2, 7};
    TDoubleVecVec thresholds{{0.95, 0.75, 0.32}, {0.4, 0.4, 0.2}};

    test::CRandomNumbers rng;

    for (std::size_t t = 0; t < dimensions.size(); ++t) {
        std::size_t dimension{dimensions[t]};
        LOG_DEBUG(<< "dimension = " << dimension);

        TDoubleVec means[]{{0.0, 0.0}, {100.0, 100.0}};
        TDoubleVecVec covariances[]{{{7.0, 1.0}, {1.0, 8.0}}, {{5.0, 2.0}, {2.0, 12.0}}};
        TPointVec points;
        for (std::size_t i = 0; i < 2; ++i) {
            TDoubleVecVec inliers;
            rng.generateMultivariateNormalSamples(means[i], covariances[i], 100, inliers);
            points.resize(points.size() + inliers.size(), TPoint{dimension});
            for (std::size_t j = inliers.size(); j > 0; --j) {
                for (std::size_t k = 0; k < inliers[j - 1].size(); ++k) {
                    points[points.size() - j](k) = inliers[j - 1][k];
                }
            }
        }
        points.emplace_back(dimension);
        points.back()(0) = 100.0;
        points.back()(1) = 250.0;
        points.emplace_back(dimension);
        points.back()(0) = -150.0;
        points.back()(1) = 0.0;
        points.emplace_back(dimension);
        points.back()(0) = -100.0;
        points.back()(1) = -100.0;
        if (dimension > 2) {
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, 1.0, (dimension - 2) * points.size(), noise);
            for (std::size_t i = 0, k = 0; i < points.size(); ++i) {
                for (std::size_t j = 2; j < dimension; ++j) {
                    points[i](j) = noise[k++];
                }
            }
        }

        TSizeVec outlierIndexes{points.size() - 3, points.size() - 2, points.size() - 1};

        core::stopDefaultAsyncExecutor();
        std::string tags[]{"sequential", "parallel"};

        CTestInstrumentation instrumentation;

        // Test in/out of core.
        for (std::size_t i = 0; i < 2; ++i) {

            // Test sequential then parallel.
            for (std::size_t j = 0; j < 2; ++j) {
                LOG_DEBUG(<< "Testing " << tags[j]);

                auto frame = factories[i](points);
                maths::analytics::COutliers::SComputeParameters params{
                    numberThreads[j],
                    numberPartitions[i],
                    true, // Standardize columns
                    maths::analytics::COutliers::E_Ensemble,
                    0,     // Compute number neighbours
                    true,  // Compute feature influences
                    0.05}; // Outlier fraction
                maths::analytics::COutliers::compute(params, *frame, instrumentation);

                bool passed{true};
                TMeanAccumulatorVec averageInfluences(dimension);

                frame->readRows(1, [&](const core::CDataFrame::TRowItr& beginRows,
                                       const core::CDataFrame::TRowItr& endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        double sum{0.0};
                        TDoubleVec influences;
                        for (std::size_t k = 0; k < dimension; ++k) {
                            passed &= ((*row)[dimension + 1 + k] >= 0.0);
                            sum += (*row)[dimension + 1 + k];
                            averageInfluences[k].add((*row)[dimension + 1 + k]);
                            influences.push_back((*row)[dimension + 1 + k]);
                        }
                        passed &= (std::fabs(sum - 1.0) < 1e-6);
                        if (row->index() == outlierIndexes[0]) {
                            LOG_DEBUG(<< "influences = "
                                      << core::CContainerPrinter::print(influences));
                            passed &= (*row)[dimension + 2] > thresholds[t][0];
                        }
                        if (row->index() == outlierIndexes[1]) {
                            LOG_DEBUG(<< "influences = "
                                      << core::CContainerPrinter::print(influences));
                            passed &= (*row)[dimension + 1] > thresholds[t][1];
                        }
                        if (row->index() == outlierIndexes[2]) {
                            LOG_DEBUG(<< "influences = "
                                      << core::CContainerPrinter::print(influences));
                            passed &= ((*row)[dimension + 2] > thresholds[t][2]) &&
                                      ((*row)[dimension + 1] > thresholds[t][2]);
                        }
                    }
                });
                BOOST_TEST_REQUIRE(passed);

                BOOST_TEST_REQUIRE(
                    std::fabs(maths::common::CBasicStatistics::mean(averageInfluences[0]) -
                              maths::common::CBasicStatistics::mean(averageInfluences[1])) < 0.08);
                core::startDefaultAsyncExecutor();
            }

            core::stopDefaultAsyncExecutor();
        }
    }
}

BOOST_AUTO_TEST_CASE(testEstimateMemoryUsedByCompute) {

    // Test that the memory estimated for compute is close to what it uses.

    TFactoryFunc toMainMemoryDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toMainMemoryDataFrame(points);
    }};
    TFactoryFunc toOnDiskDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toOnDiskDataFrame(
            test::CTestTmpDir::tmpDir(), points);
    }};
    TFactoryFunc factories[]{toMainMemoryDataFrame, toOnDiskDataFrame};

    std::size_t numberPartitions[]{1, 3};
    maths::analytics::COutliers::EMethod methods[]{
        maths::analytics::COutliers::E_Ensemble, maths::analytics::COutliers::E_Lof};
    std::size_t numberNeighbours[]{0, 5};
    bool computeFeatureInfluences[]{true, false};

    std::size_t numberInliers{40000};
    std::size_t numberOutliers{500};
    std::size_t numberPoints{numberInliers + numberOutliers};

    test::CRandomNumbers rng;

    TPointVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    core::startDefaultAsyncExecutor(3);

    for (std::size_t i = 0; i < 2; ++i) {

        LOG_DEBUG(<< "# partitions = " << numberPartitions[i]);

        auto frame = factories[i](points);

        maths::analytics::COutliers::SComputeParameters params{
            2, // Number threads
            numberPartitions[i],
            true, // Standardize columns
            methods[i],
            numberNeighbours[i],
            computeFeatureInfluences[i],
            0.05}; // Outlier fraction

        std::int64_t estimatedMemoryUsage(
            core::CDataFrame::estimateMemoryUsage(i == 0, 40500, 6, core::CAlignment::E_Aligned16) +
            maths::analytics::COutliers::estimateMemoryUsedByCompute(
                params, numberPoints,
                (numberPoints + numberPartitions[i] - 1) / numberPartitions[i],
                6 /*dimension*/));

        std::atomic<std::int64_t> memoryUsage{0};
        std::atomic<std::int64_t> maxMemoryUsage{0};

        CTestInstrumentation instrumentation;

        auto memoryUsageCallback = [&](std::int64_t delta) {
            std::int64_t memoryUsage_{memoryUsage.fetch_add(delta)};

            std::int64_t prevMaxMemoryUsage{maxMemoryUsage};
            while (prevMaxMemoryUsage < memoryUsage_ &&
                   maxMemoryUsage.compare_exchange_weak(prevMaxMemoryUsage,
                                                        memoryUsage_) == false) {
            }
            LOG_TRACE(<< "current memory = " << memoryUsage_
                      << ", high water mark = " << maxMemoryUsage.load());
        };
        instrumentation.memoryUsageCallback(memoryUsageCallback);

        maths::analytics::COutliers::compute(params, *frame, instrumentation);

        LOG_DEBUG(<< "estimated peak memory = " << estimatedMemoryUsage);
        LOG_DEBUG(<< "high water mark = " << maxMemoryUsage);
        BOOST_TEST_REQUIRE(std::abs(maxMemoryUsage - estimatedMemoryUsage) <
                           std::max(maxMemoryUsage.load(), estimatedMemoryUsage) / 3);
    }
}

BOOST_AUTO_TEST_CASE(testProgressMonitoring) {

    // Test progress monitoring invariants with and without partitioning.

    TFactoryFunc toMainMemoryDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toMainMemoryDataFrame(points);
    }};
    TFactoryFunc toOnDiskDataFrame{[](const TPointVec& points) {
        return test::CDataFrameTestUtils::toOnDiskDataFrame(
            test::CTestTmpDir::tmpDir(), points);
    }};
    TFactoryFunc factories[]{toMainMemoryDataFrame, toOnDiskDataFrame};
    std::size_t numberPartitions[]{1, 3};

    std::size_t numberInliers{10000};
    std::size_t numberOutliers{500};

    test::CRandomNumbers rng;

    TPointVec points;
    gaussianWithUniformNoise(rng, numberInliers, numberOutliers, points);

    core::startDefaultAsyncExecutor(2);

    for (std::size_t i = 0; i < 2; ++i) {

        LOG_DEBUG(<< "# partitions = " << numberPartitions[i]);

        auto frame = factories[i](points);

        std::atomic_int totalFractionalProgress{0};

        CTestInstrumentation instrumentation;
        auto reportProgress = [&totalFractionalProgress](double fractionalProgress) {
            totalFractionalProgress.fetch_add(
                static_cast<int>(65536.0 * fractionalProgress + 0.5));
        };
        instrumentation.progressCallback(std::move(reportProgress));

        std::atomic_bool finished{false};

        std::thread worker{[&]() {
            maths::analytics::COutliers::SComputeParameters params{
                2, // Number threads
                numberPartitions[i],
                true, // Standardize columns
                maths::analytics::COutliers::E_Ensemble,
                0,     // Compute number neighbours
                false, // Compute feature influences
                0.05}; // Outlier fraction
            maths::analytics::COutliers::compute(params, *frame, instrumentation);
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

        BOOST_TEST_REQUIRE(monotonic);

        LOG_DEBUG(<< "total fractional progress = " << totalFractionalProgress.load());
        BOOST_TEST_REQUIRE(std::fabs(65536 - totalFractionalProgress.load()) < 300);
    }

    core::startDefaultAsyncExecutor();
}

BOOST_AUTO_TEST_CASE(testMostlyDuplicate) {
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;

    TPointVec points;

    TSizeDoublePrVec outliers{{68, 0.4},   {5252, 0.2},  {5822, 0.2},
                              {7929, 0.2}, {12692, 0.4}, {16792, 0.4}};
    for (std::size_t i = 0; i < 16793; ++i) {
        auto outlier = std::find_if(
            outliers.begin(), outliers.end(),
            [i](const TSizeDoublePr& outlier_) { return i == outlier_.first; });
        TPoint point(1);
        point(0) = outlier != outliers.end() ? outlier->second : 0.0;
        points.push_back(std::move(point));
    }

    CTestInstrumentation instrumentation;

    for (std::size_t numberPartitions : {1, 3}) {
        auto frame = test::CDataFrameTestUtils::toMainMemoryDataFrame(points);

        maths::analytics::COutliers::SComputeParameters params{
            1, // Number threads
            numberPartitions,
            true, // Standardize columns
            maths::analytics::COutliers::E_Ensemble,
            0,     // Compute number neighbours
            false, // Compute feature influences
            0.05}; // Outlier fraction
        maths::analytics::COutliers::compute(params, *frame, instrumentation);

        TDoubleVec outlierScores(outliers.size());
        frame->readRows(1, [&](const core::CDataFrame::TRowItr& beginRows,
                               const core::CDataFrame::TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                auto outlier = std::find_if(
                    outliers.begin(),
                    outliers.end(), [i = row->index()](const TSizeDoublePr& outlier_) {
                        return i == outlier_.first;
                    });
                if (outlier != outliers.end()) {
                    outlierScores[outlier - outliers.begin()] = (*row)[1];
                }
            }
        });

        LOG_DEBUG(<< "outlier scores = " << core::CContainerPrinter::print(outlierScores));
        for (auto score : outlierScores) {
            BOOST_REQUIRE_CLOSE_ABSOLUTE(0.98, score, 0.02);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFewPoints) {

    // Check there are no failures when there only a few points.

    std::size_t rows{101};
    test::CRandomNumbers rng;

    CTestInstrumentation instrumentation;

    for (std::size_t numberPoints : {1, 2, 5}) {

        LOG_DEBUG(<< "# points = " << numberPoints);

        TPointVec points;
        TDoubleVec components;
        for (std::size_t i = 0; i < numberPoints; ++i) {
            TPoint point(rows);
            rng.generateUniformSamples(0.0, 10.0, rows, components);
            for (std::size_t j = 0; j < components.size(); ++j) {
                point(j) = components[j];
            }
            points.push_back(std::move(point));
        }

        auto frame = test::CDataFrameTestUtils::toMainMemoryDataFrame(points);

        maths::analytics::COutliers::SComputeParameters params{
            1,    // Number threads
            1,    // Number partitions,
            true, // Standardize columns
            maths::analytics::COutliers::E_Ensemble,
            0,     // Compute number neighbours
            true,  // Compute feature influences
            0.05}; // Outlier fraction
        maths::analytics::COutliers::compute(params, *frame, instrumentation);

        bool passed{true};

        frame->readRows(1, [&](const core::CDataFrame::TRowItr& beginRows,
                               const core::CDataFrame::TRowItr& endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                // Check score is in range 0 to 1.
                LOG_DEBUG(<< "outlier score = " << (*row)[rows]);
                passed &= (*row)[rows] >= 0.0 && (*row)[rows] <= 1.0;
            }
        });

        BOOST_TEST_REQUIRE(passed);
    }
}

BOOST_AUTO_TEST_SUITE_END()
