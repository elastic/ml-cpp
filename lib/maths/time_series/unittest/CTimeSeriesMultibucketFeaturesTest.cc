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

#include <core/CLogger.h>
#include <core/CMemoryDef.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CSmallVector.h>
#include <core/Constants.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsCovariances.h>

#include <maths/time_series/CTimeSeriesMultibucketFeatures.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <tuple>

BOOST_AUTO_TEST_SUITE(CTimeSeriesMultibucketFeaturesTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
using TVector = maths::common::CVector<double>;
using TVector2x1 = maths::common::CVectorNx1<double, 2>;
using TCovariance2x2 = maths::common::CBasicStatistics::SSampleCovariances<TVector2x1>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;

BOOST_AUTO_TEST_CASE(testUnivariateMean) {

    // Test we get the values and weights we expect.

    maths::time_series::CTimeSeriesMultibucketScalarMean feature{3};

    TDouble1Vec mean;
    maths_t::TDoubleWeightsAry1Vec weight;
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_TEST_REQUIRE(mean.empty());
    BOOST_TEST_REQUIRE(weight.empty());
    BOOST_REQUIRE_EQUAL(0.0, feature.correlationWithBucketValue());

    feature.add(0, 100, {5.0}, {maths_t::countWeight(1.1)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(0, mean.size());
    BOOST_REQUIRE_EQUAL(0, weight.size());
    BOOST_REQUIRE_EQUAL(1.0, feature.correlationWithBucketValue());

    feature.add(1, 100, {7.0}, {maths_t::countWeight(0.3)});
    feature.add(2, 100, {3.0}, {maths_t::countWeight(0.6)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(4.6252, mean[0], 5e-5);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6498, maths_t::countForUpdate(weight[0]), 5e-5);

    // Test removing old values.

    feature.add(3, 100, {3.0}, {maths_t::countWeight(0.6)});
    feature.add(4, 100, {3.0}, {maths_t::countWeight(0.6)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(3.0, mean[0], 5e-5);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6, maths_t::countForUpdate(weight[0]), 5e-5);

    // Clone.

    auto clone = feature.clone();
    BOOST_REQUIRE_EQUAL(feature.checksum(), clone->checksum());

    // Memory accounting through base pointer.

    maths::time_series::CTimeSeriesMultibucketScalarMean* base = &feature;
    LOG_DEBUG(<< "size = " << core::CMemory::dynamicSize(&feature));

    BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(base),
                        core::CMemory::dynamicSize(&feature));

    // Check clearing the feature.

    feature.clear();
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_TEST_REQUIRE(mean.empty());
    BOOST_TEST_REQUIRE(weight.empty());
    BOOST_REQUIRE_EQUAL(0.0, feature.correlationWithBucketValue());

    // Check updating with non-unit variance scale.

    feature.add(0, 100, {5.0}, {maths_t::seasonalVarianceScaleWeight(4.0)});
    feature.add(1, 100, {5.0}, {maths_t::seasonalVarianceScaleWeight(4.0)});
    feature.add(2, 100, {5.0}, {maths_t::seasonalVarianceScaleWeight(4.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_EQUAL(2.5, mean[0]);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0]));

    feature.add(0, 100, {5.0}, {maths_t::countVarianceScaleWeight(4.0)});
    feature.add(1, 100, {5.0}, {maths_t::countVarianceScaleWeight(4.0)});
    feature.add(2, 100, {5.0}, {maths_t::countVarianceScaleWeight(4.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_EQUAL(2.5, mean[0]);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0]));
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0]));
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0]));

    // Multiple values.

    feature.add(3, 100, {5.0, 4.0, 8.0},
                {maths_t::countWeight(2.0), maths_t::countWeight(2.0),
                 maths_t::countWeight(1.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(4.5119, mean[0], 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.476, maths_t::countForUpdate(weight[0]), 5e-5);

    // Test unequal time intervals.

    feature.add(0, 100, {5.0}, {maths_t::countWeight(1.0)});
    feature.add(2, 100, {2.0}, {maths_t::countWeight(1.0)});
    feature.add(8, 100, {9.0}, {maths_t::countWeight(1.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(5.54, mean[0], 5e-5);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0]));

    feature.add(10, 100, {1.0}, {maths_t::countWeight(5.0)});
    feature.add(12, 100, {1.0}, {maths_t::countWeight(2.0)});
    feature.add(18, 100, {1.0}, {maths_t::countWeight(9.0)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_EQUAL(1.0, mean[0]);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(5.54, maths_t::countForUpdate(weight[0]), 5e-5);

    // Test correlation between the feature and the last bucket value matches
    // the expected value.

    TMeanAccumulator meanCorrelation;

    test::CRandomNumbers rng;
    for (std::size_t t = 0; t < 10; ++t) {
        TDoubleVec samples;
        rng.generateNormalSamples(0.0, 1.0, 300, samples);

        TCovariance2x2 covarianceAccumulator{2};
        for (std::size_t i = 0; i < samples.size(); i += 3) {
            for (std::size_t j = 0; j < 3; ++j) {
                feature.add(static_cast<core_t::TTime>(10 * (i + j)), 10,
                            {samples[i + j]}, maths_t::CUnitWeights::SINGLE_UNIT);
            }
            TDouble1Vec value;
            std::tie(mean, std::ignore) = feature.value();
            covarianceAccumulator.add(TVector2x1{{samples[i + 2], mean[0]}});
        }

        auto covariance = maths::common::CBasicStatistics::covariances(covarianceAccumulator);
        double correlation{covariance(0, 1) /
                           std::sqrt(covariance(0, 0) * covariance(1, 1))};
        LOG_DEBUG(<< "correlation = " << correlation
                  << " expected = " << feature.correlationWithBucketValue());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(feature.correlationWithBucketValue(), correlation,
                                     0.15 * feature.correlationWithBucketValue());
        meanCorrelation.add(correlation);
    }

    LOG_DEBUG(<< "mean correlation = "
              << maths::common::CBasicStatistics::mean(meanCorrelation));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(feature.correlationWithBucketValue(),
                                 maths::common::CBasicStatistics::mean(meanCorrelation),
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
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser{parser};

        maths::time_series::CTimeSeriesMultibucketScalarMean restored{3};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](core::CStateRestoreTraverser& traverser_) {
            return restored.acceptRestoreTraverser(traverser_);
        }));
        BOOST_REQUIRE_EQUAL(feature.checksum(), restored.checksum());
        feature.clear();
    }
}

BOOST_AUTO_TEST_CASE(testMultivariateMean) {

    // Test we get the values and weights we expect.

    maths::time_series::CTimeSeriesMultibucketVectorMean feature{3};

    TDouble10Vec1Vec mean;
    maths_t::TDouble10VecWeightsAry1Vec weight;
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);
    BOOST_TEST_REQUIRE(mean.empty());
    BOOST_TEST_REQUIRE(weight.empty());
    BOOST_REQUIRE_EQUAL(0.0, feature.correlationWithBucketValue());

    feature.add(0, 100, {{5.0, 4.0}}, {maths_t::countWeight(1.1, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_TEST_REQUIRE(mean.empty());
    BOOST_TEST_REQUIRE(weight.empty());
    BOOST_REQUIRE_EQUAL(1.0, feature.correlationWithBucketValue());

    feature.add(1, 100, {{7.0, 6.0}}, {maths_t::countWeight(0.3, 2)});
    feature.add(2, 100, {{3.0, 2.0}}, {maths_t::countWeight(0.6, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_EQUAL(2, mean[0].size());
    BOOST_REQUIRE_EQUAL(2, maths_t::countForUpdate(weight[0]).size());
    double expectedMean[]{4.6252, 3.6252};
    for (std::size_t i = 0; i < 2; ++i) {
        BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedMean[i], mean[0][i], 5e-5);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0])[i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0])[i]);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(0.6498, maths_t::countForUpdate(weight[0])[i], 5e-5);
    }

    // Clone.

    auto clone = feature.clone();
    BOOST_REQUIRE_EQUAL(feature.checksum(), clone->checksum());

    // Memory accounting through base pointer.

    maths::time_series::CTimeSeriesMultibucketVectorMean* base = &feature;
    LOG_DEBUG(<< "size = " << core::CMemory::dynamicSize(&feature));

    BOOST_REQUIRE_EQUAL(core::CMemory::dynamicSize(base),
                        core::CMemory::dynamicSize(&feature));

    // Check clearing the feature.

    feature.clear();
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_TEST_REQUIRE(mean.empty());
    BOOST_TEST_REQUIRE(weight.empty());
    BOOST_REQUIRE_EQUAL(0.0, feature.correlationWithBucketValue());

    // Check updating with non-unit variance scale.

    feature.add(0, 100, {{5.0, 10.0}}, {maths_t::seasonalVarianceScaleWeight(4.0, 2)});
    feature.add(1, 100, {{5.0, 10.0}}, {maths_t::seasonalVarianceScaleWeight(4.0, 2)});
    feature.add(2, 100, {{5.0, 10.0}}, {maths_t::seasonalVarianceScaleWeight(4.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    expectedMean[0] = 2.5;
    expectedMean[1] = 5.0;
    for (std::size_t i = 0; i < 2; ++i) {
        BOOST_REQUIRE_EQUAL(expectedMean[i], mean[0][i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0])[i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0])[i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0])[i]);
    }

    feature.add(0, 100, {{5.0, 10.0}}, {maths_t::countVarianceScaleWeight(4.0, 2)});
    feature.add(1, 100, {{5.0, 10.0}}, {maths_t::countVarianceScaleWeight(4.0, 2)});
    feature.add(2, 100, {{5.0, 10.0}}, {maths_t::countVarianceScaleWeight(4.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    for (std::size_t i = 0; i < 2; ++i) {
        BOOST_REQUIRE_EQUAL(expectedMean[i], mean[0][i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::seasonalVarianceScale(weight[0])[i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::countVarianceScale(weight[0])[i]);
        BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0])[i]);
    }

    // Multiple values.

    feature.add(3, 100, {{5.0, 4.0}, {4.0, 3.0}, {8.0, 7.0}},
                {maths_t::countWeight(2.0, 2), maths_t::countWeight(2.0, 2),
                 maths_t::countWeight(1.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(4.5119, mean[0][0], 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(4.4039, mean[0][1], 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.476, maths_t::countForUpdate(weight[0])[0], 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.476, maths_t::countForUpdate(weight[0])[1], 5e-5);

    // Test unequal time intervals.

    feature.add(0, 100, {{5.0, 4.0}}, {maths_t::countWeight(1.0, 2)});
    feature.add(2, 100, {{2.0, 1.0}}, {maths_t::countWeight(1.0, 2)});
    feature.add(8, 100, {{9.0, 8.0}}, {maths_t::countWeight(1.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(5.54, mean[0][0], 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(4.54, mean[0][1], 5e-5);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0])[0]);
    BOOST_REQUIRE_EQUAL(1.0, maths_t::countForUpdate(weight[0])[1]);

    feature.add(10, 100, {{1.0, 1.0}}, {maths_t::countWeight(5.0, 2)});
    feature.add(12, 100, {{1.0, 1.0}}, {maths_t::countWeight(2.0, 2)});
    feature.add(18, 100, {{1.0, 1.0}}, {maths_t::countWeight(9.0, 2)});
    std::tie(mean, weight) = feature.value();
    LOG_DEBUG(<< "mean = " << mean << " weight = " << weight);

    BOOST_REQUIRE_EQUAL(1, mean.size());
    BOOST_REQUIRE_EQUAL(1, weight.size());
    BOOST_REQUIRE_EQUAL(1.0, mean[0][0]);
    BOOST_REQUIRE_EQUAL(1.0, mean[0][1]);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(5.54, maths_t::countForUpdate(weight[0])[0], 5e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(5.54, maths_t::countForUpdate(weight[0])[1], 5e-5);

    // Check persist and restore is idempotent.
    for (auto i : {0, 1}) {
        std::string xml;
        ml::core::CRapidXmlStatePersistInserter inserter{"root"};
        feature.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< i << ") model XML representation:\n" << xml);
        LOG_DEBUG(<< i << ") model XML size: " << xml.size());

        // Restore the XML into a new feature and assert checksums.

        maths::time_series::CTimeSeriesMultibucketVectorMean restored{3};
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));

        core::CRapidXmlStateRestoreTraverser traverser{parser};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](core::CStateRestoreTraverser& traverser_) {
            return restored.acceptRestoreTraverser(traverser_);
        }));
        BOOST_REQUIRE_EQUAL(feature.checksum(), restored.checksum());
        feature.clear();
    }
}

BOOST_AUTO_TEST_SUITE_END()
