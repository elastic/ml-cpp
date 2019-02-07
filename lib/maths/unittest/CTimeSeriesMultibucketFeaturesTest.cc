/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesMultibucketFeaturesTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsCovariances.h>
#include <maths/CTimeSeriesMultibucketFeatures.h>

#include <test/CRandomNumbers.h>

#include <tuple>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;

void CTimeSeriesMultibucketFeaturesTest::testUnivariateMean() {

    // Test we get the values and weights we expect.

    using TMultibucketMean = maths::CTimeSeriesMultibucketMean<double>;

    TMultibucketMean feature{3};

    TDouble1Vec mean;
    maths_t::TDoubleWeightsAry1Vec weight;
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());
    CPPUNIT_ASSERT_EQUAL(0.0, feature.correlationWithBucketValue());

    feature.add(0, 100, {5.0}, {maths_t::countWeight(1.1)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(0), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), weight.size());
    CPPUNIT_ASSERT_EQUAL(1.0, feature.correlationWithBucketValue());

    feature.add(1, 100, {7.0}, {maths_t::countWeight(0.3)});
    feature.add(2, 100, {3.0}, {maths_t::countWeight(0.6)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.6252, mean[0], 5e-5);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6498, maths_t::countForUpdate(weight[0]), 5e-5);

    // Test removing old values.

    feature.add(3, 100, {3.0}, {maths_t::countWeight(0.6)});
    feature.add(4, 100, {3.0}, {maths_t::countWeight(0.6)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mean[0], 5e-5);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6, maths_t::countForUpdate(weight[0]), 5e-5);

    // Clone.

    auto clone = feature.clone();
    CPPUNIT_ASSERT_EQUAL(feature.checksum(), clone->checksum());

    // Memory accounting through base pointer.

    maths::CTimeSeriesMultibucketFeature<double>* base = &feature;
    LOG_DEBUG(<< "size = " << core::CMemory::dynamicSize(&feature));

    CPPUNIT_ASSERT_EQUAL(core::CMemory::dynamicSize(base),
                         core::CMemory::dynamicSize(&feature));

    // Check clearing the feature.

    feature.clear();
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());
    CPPUNIT_ASSERT_EQUAL(0.0, feature.correlationWithBucketValue());

    // Check updating with non-unit variance scale.

    feature.add(0, 100, {5.0}, {maths_t::seasonalVarianceScaleWeight(4.0)});
    feature.add(1, 100, {5.0}, {maths_t::seasonalVarianceScaleWeight(4.0)});
    feature.add(2, 100, {5.0}, {maths_t::seasonalVarianceScaleWeight(4.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(2.5, mean[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0]));

    feature.add(0, 100, {5.0}, {maths_t::countVarianceScaleWeight(4.0)});
    feature.add(1, 100, {5.0}, {maths_t::countVarianceScaleWeight(4.0)});
    feature.add(2, 100, {5.0}, {maths_t::countVarianceScaleWeight(4.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(2.5, mean[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0]));

    // Multiple values.

    feature.add(3, 100, {5.0, 4.0, 8.0},
                {maths_t::countWeight(2.0), maths_t::countWeight(2.0),
                 maths_t::countWeight(1.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.5119, mean[0], 5e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.476, maths_t::countForUpdate(weight[0]), 5e-5);

    // Test unequal time intervals.

    feature.add(0, 100, {5.0}, {maths_t::countWeight(1.0)});
    feature.add(2, 100, {2.0}, {maths_t::countWeight(1.0)});
    feature.add(8, 100, {9.0}, {maths_t::countWeight(1.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.54, mean[0], 5e-5);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0]));

    feature.add(10, 100, {1.0}, {maths_t::countWeight(5.0)});
    feature.add(12, 100, {1.0}, {maths_t::countWeight(2.0)});
    feature.add(18, 100, {1.0}, {maths_t::countWeight(9.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(1.0, mean[0]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.54, maths_t::countForUpdate(weight[0]), 5e-5);

    // Test correlation between the feature and the last bucket
    // value matches the expected value.

    maths::CBasicStatistics::SSampleMean<double>::TAccumulator meanCorrelation;

    test::CRandomNumbers rng;
    for (std::size_t t = 0; t < 10; ++t) {
        TDoubleVec samples;
        rng.generateNormalSamples(0.0, 1.0, 300, samples);

        maths::CBasicStatistics::SSampleCovariances<maths::CVectorNx1<double, 2>> covariance(2);
        for (std::size_t i = 0; i < samples.size(); i += 3) {
            for (std::size_t j = 0; j < 3; ++j) {
                feature.add(static_cast<core_t::TTime>(10 * (i + j)), 10,
                            {samples[i + j]}, maths_t::CUnitWeights::SINGLE_UNIT);
            }
            TDouble1Vec value;
            std::tie(mean, std::ignore) = feature.value();
            covariance.add(maths::CVectorNx1<double, 2>{{samples[i + 2], mean[0]}});
        }

        auto c = maths::CBasicStatistics::covariances(covariance);
        double correlation{c(0, 1) / std::sqrt(c(0, 0) * c(1, 1))};
        LOG_DEBUG(<< "correlation = " << correlation
                  << " expected = " << feature.correlationWithBucketValue());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(feature.correlationWithBucketValue(), correlation,
                                     0.15 * feature.correlationWithBucketValue());
        meanCorrelation.add(correlation);
    }

    LOG_DEBUG(<< "mean correlation = " << maths::CBasicStatistics::mean(meanCorrelation));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(feature.correlationWithBucketValue(),
                                 maths::CBasicStatistics::mean(meanCorrelation),
                                 0.015 * feature.correlationWithBucketValue());

    // Check persist and restore is idempotent.
    for (auto i : {0, 1}) {
        std::string xml;
        ml::core::CRapidXmlStatePersistInserter inserter{"root"};
        feature.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< i << ") model XML representation:\n" << xml);
        LOG_DEBUG(<< i << ") model XML size: " << xml.size());

        // Restore the XML into a new feature and assert checksums.

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        TMultibucketMean restored{3};
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            boost::bind(&TMultibucketMean::acceptRestoreTraverser, &restored, _1)));
        CPPUNIT_ASSERT_EQUAL(feature.checksum(), restored.checksum());
        feature.clear();
    }
}

void CTimeSeriesMultibucketFeaturesTest::testMultivariateMean() {

    // Test we get the values and weights we expect.

    using TMultibucketMean = maths::CTimeSeriesMultibucketMean<TDouble10Vec>;

    TMultibucketMean feature{3};

    TDouble10Vec1Vec mean;
    maths_t::TDouble10VecWeightsAry1Vec weight;
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());
    CPPUNIT_ASSERT_EQUAL(0.0, feature.correlationWithBucketValue());

    feature.add(0, 100, {{5.0, 4.0}}, {maths_t::countWeight(1.1, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());
    CPPUNIT_ASSERT_EQUAL(1.0, feature.correlationWithBucketValue());

    feature.add(1, 100, {{7.0, 6.0}}, {maths_t::countWeight(0.3, 2)});
    feature.add(2, 100, {{3.0, 2.0}}, {maths_t::countWeight(0.6, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), mean[0].size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), maths_t::countForUpdate(weight[0]).size());
    double expectedMean[]{4.6252, 3.6252};
    for (std::size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean[i], mean[0][i], 5e-5);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0])[i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0])[i]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6498, maths_t::countForUpdate(weight[0])[i], 5e-5);
    }

    // Clone.

    auto clone = feature.clone();
    CPPUNIT_ASSERT_EQUAL(feature.checksum(), clone->checksum());

    // Memory accounting through base pointer.

    maths::CTimeSeriesMultibucketFeature<TDouble10Vec>* base = &feature;
    LOG_DEBUG(<< "size = " << core::CMemory::dynamicSize(&feature));

    CPPUNIT_ASSERT_EQUAL(core::CMemory::dynamicSize(base),
                         core::CMemory::dynamicSize(&feature));

    // Check clearing the feature.

    feature.clear();
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT(mean.empty());
    CPPUNIT_ASSERT(weight.empty());
    CPPUNIT_ASSERT_EQUAL(0.0, feature.correlationWithBucketValue());

    // Check updating with non-unit variance scale.

    feature.add(0, 100, {{5.0, 10.0}}, {maths_t::seasonalVarianceScaleWeight(4.0, 2)});
    feature.add(1, 100, {{5.0, 10.0}}, {maths_t::seasonalVarianceScaleWeight(4.0, 2)});
    feature.add(2, 100, {{5.0, 10.0}}, {maths_t::seasonalVarianceScaleWeight(4.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    expectedMean[0] = 2.5;
    expectedMean[1] = 5.0;
    for (std::size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT_EQUAL(expectedMean[i], mean[0][i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0])[i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0])[i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0])[i]);
    }

    feature.add(0, 100, {{5.0, 10.0}}, {maths_t::countVarianceScaleWeight(4.0, 2)});
    feature.add(1, 100, {{5.0, 10.0}}, {maths_t::countVarianceScaleWeight(4.0, 2)});
    feature.add(2, 100, {{5.0, 10.0}}, {maths_t::countVarianceScaleWeight(4.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    for (std::size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT_EQUAL(expectedMean[i], mean[0][i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0])[i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countVarianceScale(weight[0])[i]);
        CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0])[i]);
    }

    // Multiple values.

    feature.add(3, 100, {{5.0, 4.0}, {4.0, 3.0}, {8.0, 7.0}},
                {maths_t::countWeight(2.0, 2), maths_t::countWeight(2.0, 2),
                 maths_t::countWeight(1.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.5119, mean[0][0], 5e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.4039, mean[0][1], 5e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.476, maths_t::countForUpdate(weight[0])[0], 5e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.476, maths_t::countForUpdate(weight[0])[1], 5e-5);

    // Test unequal time intervals.

    feature.add(0, 100, {{5.0, 4.0}}, {maths_t::countWeight(1.0, 2)});
    feature.add(2, 100, {{2.0, 1.0}}, {maths_t::countWeight(1.0, 2)});
    feature.add(8, 100, {{9.0, 8.0}}, {maths_t::countWeight(1.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.54, mean[0][0], 5e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.54, mean[0][1], 5e-5);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0])[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, maths_t::countForUpdate(weight[0])[1]);

    feature.add(10, 100, {{1.0, 1.0}}, {maths_t::countWeight(5.0, 2)});
    feature.add(12, 100, {{1.0, 1.0}}, {maths_t::countWeight(2.0, 2)});
    feature.add(18, 100, {{1.0, 1.0}}, {maths_t::countWeight(9.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), mean.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), weight.size());
    CPPUNIT_ASSERT_EQUAL(1.0, mean[0][0]);
    CPPUNIT_ASSERT_EQUAL(1.0, mean[0][1]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.54, maths_t::countForUpdate(weight[0])[0], 5e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.54, maths_t::countForUpdate(weight[0])[1], 5e-5);

    // Check persist and restore is idempotent.
    for (auto i : {0, 1}) {
        std::string xml;
        ml::core::CRapidXmlStatePersistInserter inserter{"root"};
        feature.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< i << ") model XML representation:\n" << xml);
        LOG_DEBUG(<< i << ") model XML size: " << xml.size());

        // Restore the XML into a new feature and assert checksums.

        TMultibucketMean restored{3};
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));

        core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            boost::bind(&TMultibucketMean::acceptRestoreTraverser, &restored, _1)));
        CPPUNIT_ASSERT_EQUAL(feature.checksum(), restored.checksum());
        feature.clear();
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
