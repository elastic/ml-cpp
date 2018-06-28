/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CClusterEvaluationTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CClusterEvaluation.h>
#include <maths/CLinearAlgebra.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TVector3 = maths::CVectorNx1<double, 3>;
using TVector3Vec = std::vector<TVector3>;
using TVector3VecVec = std::vector<TVector3Vec>;
using TMatrix3 = maths::CSymmetricMatrixNxN<double, 3>;
using TMatrix3Vec = std::vector<TMatrix3>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

void CClusterEvaluationTest::testSilhouetteExact(void) {
    LOG_DEBUG("+-----------------------------------------------+");
    LOG_DEBUG("|  CClusterEvaluationTest::testSilhouetteExact  |");
    LOG_DEBUG("+-----------------------------------------------+");

    // Test verses sklearn metrics implementation.
    // Test some corner cases, i.e. empty clusters and empty cluster.
    // Check invariants, i.e. must be in the range [-1,1] and scores
    // should be maximal for the "best" clustering.

    {
        double coordinates[][3]{{1.1, 1.5, 1.6},   {2.1, 1.3, 1.7},
                                {0.1, 1.6, 0.9},   {1.8, 1.1, 1.4},
                                {10.1, 9.3, 12.7}, {8.1, 12.3, 8.7},
                                {8.1, 12.3, 8.7}};
        TVector3VecVec clusters{{TVector3(coordinates[0]), TVector3(coordinates[1]),
                                 TVector3(coordinates[2]), TVector3(coordinates[3])},
                                {TVector3(coordinates[4]), TVector3(coordinates[5]),
                                 TVector3(coordinates[6])}};
        TDoubleVecVec expected{{0.9325687, 0.91753216, 0.89131175, 0.93142234},
                               {0.67153939, 0.81837762, 0.81837762}};

        TDoubleVecVec statistics;
        maths::CClusterEvaluation::silhouetteExact(clusters, statistics);
        LOG_DEBUG("statistics = " << core::CContainerPrinter::print(statistics));

        for (std::size_t i = 0u; i < statistics.size(); ++i) {
            CPPUNIT_ASSERT_EQUAL(expected[i].size(), statistics[i].size());
            for (std::size_t j = 0u; j < statistics[i].size(); ++j) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[i][j], statistics[i][j], 5e-8);
            }
        }
    }
    {
        double coordinates[][3]{{1.1, 1.5, 1.6}, {2.1, 1.3, 1.7}};

        TVector3VecVec clusters;
        TDoubleVecVec statistics;
        maths::CClusterEvaluation::silhouetteExact(clusters, statistics);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), statistics.size());

        clusters.push_back({TVector3(coordinates[0]), TVector3(coordinates[1])});

        maths::CClusterEvaluation::silhouetteExact(clusters, statistics);
        LOG_DEBUG("statistics = " << core::CContainerPrinter::print(statistics));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), statistics.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), statistics[0].size());
        for (auto statistic : statistics[0]) {
            CPPUNIT_ASSERT_EQUAL(0.0, statistic);
        }

        clusters.push_back(TVector3Vec());
        maths::CClusterEvaluation::silhouetteExact(clusters, statistics);
        LOG_DEBUG("statistics = " << core::CContainerPrinter::print(statistics));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), statistics.size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), statistics[0].size());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), statistics[1].size());
        for (auto statistic : statistics[0]) {
            CPPUNIT_ASSERT_EQUAL(0.0, statistic);
        }
    }
    {
        test::CRandomNumbers rng;

        TDoubleVec coordinates[2];
        rng.generateUniformSamples(0.0, 5.0, 60, coordinates[0]);
        rng.generateUniformSamples(20.0, 25.0, 90, coordinates[1]);
        TVector3Vec points;
        for (std::size_t i = 0u; i < 2; ++i) {
            for (std::size_t j = 0u; j < coordinates[i].size(); j += 3) {
                points.emplace_back(&coordinates[i][j], &coordinates[i][j + 3]);
            }
        }
        TSizeVec indicator(20, 0);
        indicator.resize(50, 1);

        double score = -1.0;
        std::size_t best = 100;
        for (std::size_t t = 0u; t < 100; ++t) {
            TVector3VecVec clusters(2);
            for (std::size_t i = 0u; i < points.size(); ++i) {
                clusters[indicator[i]].push_back(points[i]);
            }

            TDoubleVecVec statistics;
            maths::CClusterEvaluation::silhouetteExact(clusters, statistics);

            TMeanAccumulator meanScore_;
            for (const auto& clusterStatistics : statistics) {
                for (auto statistic : clusterStatistics) {
                    CPPUNIT_ASSERT(statistic >= -1.0 && statistic <= 1.0);
                    meanScore_.add(statistic);
                }
            }

            double meanScore = maths::CBasicStatistics::mean(meanScore_);
            boost::tie(score, best) = meanScore > score ? boost::tie(meanScore, t)
                                                        : boost::tie(score, best);
            if (t % 10 == 0) {
                LOG_DEBUG("score = " << meanScore);
            }
            rng.random_shuffle(indicator.begin(), indicator.end());
        }

        LOG_DEBUG("best = " << best);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), best);
    }
}

void CClusterEvaluationTest::testSilhouetteApprox(void) {
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CClusterEvaluationTest::testSilhouetteApprox  |");
    LOG_DEBUG("+------------------------------------------------+");

    test::CRandomNumbers rng;

    TMeanAccumulator meanErrorEstimate;
    for (std::size_t t = 0u; t < 5; ++t) {
        TSizeVec sizes{5500, 10000, 2500};
        TVector3Vec means;
        TMatrix3Vec covariances;
        TVector3VecVec clusters;
        rng.generateRandomMultivariateNormals(sizes, means, covariances, clusters);

        TDoubleVecVec exact;
        maths::CClusterEvaluation::silhouetteExact(clusters, exact);

        TDoubleVecVec approx;
        TDoubleVecVec errors;
        maths::CClusterEvaluation::silhouetteApprox(clusters, approx, errors);

        CPPUNIT_ASSERT_EQUAL(exact.size(), approx.size());
        CPPUNIT_ASSERT_EQUAL(approx.size(), errors.size());

        TMeanAccumulator meanError_;
        TMeanAccumulator errorEstimate_;
        for (std::size_t i = 0u; i < exact.size(); ++i) {
            CPPUNIT_ASSERT_EQUAL(exact[i].size(), approx[i].size());
            CPPUNIT_ASSERT_EQUAL(approx[i].size(), errors[i].size());
            for (std::size_t j = 0u; j < exact[i].size(); ++j) {
                meanError_.add(std::pow(exact[i][j] - approx[i][j], 2.0));
                errorEstimate_.add(std::sqrt(errors[i][j]));
            }
        }
        double meanError{std::sqrt(maths::CBasicStatistics::mean(meanError_))};
        double errorEstimate{maths::CBasicStatistics::mean(errorEstimate_)};

        LOG_DEBUG("error = " << meanError);
        LOG_DEBUG("error estimate = " << errorEstimate);
        CPPUNIT_ASSERT(meanError < 0.005);
        CPPUNIT_ASSERT(std::fabs(meanError / errorEstimate - 1.0) < 0.6);
        meanErrorEstimate.add(std::fabs(meanError / errorEstimate - 1.0));
    }

    LOG_DEBUG("mean error estimate = " << maths::CBasicStatistics::mean(meanErrorEstimate));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanErrorEstimate) < 0.35);

    TSizeVec sizes{55000, 140000, 65000, 70000};
    TVector3Vec means;
    TMatrix3Vec covariances;
    TVector3VecVec clusters;
    rng.generateRandomMultivariateNormals(sizes, means, covariances, clusters);

    TDoubleVecVec approx;
    TDoubleVecVec errors;
    maths::CClusterEvaluation::silhouetteApprox(clusters, approx, errors);

    for (std::size_t p = 0; p < 50000; p += 5000) {
        for (std::size_t i = 0u; i < clusters.size(); ++i) {
            maths::CClusterEvaluation::CClusterDissimilarity<TVector3> dissimilarity(
                clusters[i][p]);
            double a{dissimilarity(clusters[i])};
            double b{std::numeric_limits<double>::max()};
            for (std::size_t j = 0u; j < clusters.size(); ++j) {
                if (i != j) {
                    b = std::min(b, dissimilarity(clusters[j]));
                }
            }
            double exact{(b - a) / std::max(a, b)};
            double error{std::sqrt(errors[i][p])};
            LOG_DEBUG("exact = " << exact << ", approx = " << approx[i][p]
                                 << ", error = " << error);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(exact, approx[i][p],
                                         3.0 * std::sqrt(errors[i][p]));
        }
    }
}

CppUnit::Test* CClusterEvaluationTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CClusterEvaluationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CClusterEvaluationTest>(
        "CClusterEvaluationTest::testSilhouetteExact",
        &CClusterEvaluationTest::testSilhouetteExact));
    suiteOfTests->addTest(new CppUnit::TestCaller<CClusterEvaluationTest>(
        "CClusterEvaluationTest::testSilhouetteApprox",
        &CClusterEvaluationTest::testSilhouetteApprox));

    return suiteOfTests;
}
