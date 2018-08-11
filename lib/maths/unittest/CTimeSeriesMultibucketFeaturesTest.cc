/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesMultibucketFeaturesTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CTimeSeriesMultibucketFeatures.h>

#include <tuple>

using namespace ml;

using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;

void CTimeSeriesMultibucketFeaturesTest::testUnivariateMean() {
    // Test we get the values and weights we expect.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TTimeMeanAccumulatorPr = std::pair<core_t::TTime, TMeanAccumulator>;
    using TTimeMeanAccumulatorPrVec = std::vector<TTimeMeanAccumulatorPr>;

    TTimeMeanAccumulatorPrVec buf;

    TDouble1Vec mean;
    maths_t::TDoubleWeightsAry1Vec weight;
    std::tie(mean, weight) = maths::CTimeSeriesMultibucketFeatures::mean<double>(
        buf.begin(), buf.end());
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());

    buf.emplace_back(10, maths::CBasicStatistics::accumulator(1.1, 5.0));
    buf.emplace_back(20, maths::CBasicStatistics::accumulator(0.3, 7.0));
    buf.emplace_back(30, maths::CBasicStatistics::accumulator(0.6, 3.0));

    std::tie(mean, weight) = maths::CTimeSeriesMultibucketFeatures::mean<double>(
        buf.begin(), buf.begin() + 1);
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(5.0, mean[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.1, maths_t::countForUpdate(weight[0]));

    std::tie(mean, weight) = maths::CTimeSeriesMultibucketFeatures::mean<double>(
        buf.begin(), buf.end());
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.6252, mean[0], 5e-5);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.807, maths_t::countForUpdate(weight[0]), 5e-5);
}

void CTimeSeriesMultibucketFeaturesTest::testMultivariateMean() {
    // Test we get the values and weights we expect.

    using TVector = maths::CVector<double>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<TVector>::TAccumulator;
    using TTimeMeanAccumulatorPr = std::pair<core_t::TTime, TMeanAccumulator>;
    using TTimeMeanAccumulatorPrVec = std::vector<TTimeMeanAccumulatorPr>;

    TTimeMeanAccumulatorPrVec buf;

    TDouble10Vec1Vec mean;
    maths_t::TDouble10VecWeightsAry1Vec weight;
    std::tie(mean, weight) = maths::CTimeSeriesMultibucketFeatures::mean<TDouble10Vec>(
        buf.begin(), buf.end());
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());

    buf.emplace_back(1, maths::CBasicStatistics::accumulator(
                            1.1, TVector(TDouble1Vec{5.0, 4.0})));
    buf.emplace_back(2, maths::CBasicStatistics::accumulator(
                            0.3, TVector(TDouble1Vec{7.0, 6.0})));
    buf.emplace_back(3, maths::CBasicStatistics::accumulator(
                            0.6, TVector(TDouble1Vec{3.0, 2.0})));

    std::tie(mean, weight) = maths::CTimeSeriesMultibucketFeatures::mean<TDouble10Vec>(
        buf.begin(), buf.begin() + 1);
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(std::string("[5, 4]"),
                         core::CContainerPrinter::print(mean[0]));
    CPPUNIT_ASSERT_EQUAL(std::string("[1, 1]"),
                         core::CContainerPrinter::print(
                             maths_t::seasonalVarianceScale(weight[0])));
    CPPUNIT_ASSERT_EQUAL(
        std::string("[1, 1]"),
        core::CContainerPrinter::print(maths_t::countVarianceScale(weight[0])));
    CPPUNIT_ASSERT_EQUAL(
        std::string("[1.1, 1.1]"),
        core::CContainerPrinter::print(maths_t::countForUpdate(weight[0])));

    std::tie(mean, weight) = maths::CTimeSeriesMultibucketFeatures::mean<TDouble10Vec>(
        buf.begin(), buf.end());
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), mean[0].size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), maths_t::countForUpdate(weight[0]).size());
    double expectedMean[]{4.6252, 3.6252};
    for (std::size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean[i], mean[0][i], 5e-5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.807, maths_t::countForUpdate(weight[0])[i], 5e-5);
    }
}

CppUnit::Test* CTimeSeriesMultibucketFeaturesTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CTimeSeriesMultibucketFeaturesTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesMultibucketFeaturesTest>(
        "CTimeSeriesMultibucketFeaturesTest::testUnivariateMean",
        &CTimeSeriesMultibucketFeaturesTest::testUnivariateMean));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesMultibucketFeaturesTest>(
        "CTimeSeriesMultibucketFeaturesTest::testMultivariateMean",
        &CTimeSeriesMultibucketFeaturesTest::testMultivariateMean));

    return suiteOfTests;
}
