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

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>
#include <core/UnwrapRef.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CGathererTools.h>
#include <model/CMetricBucketGatherer.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>
#include <model/SModelParams.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <algorithm>
#include <boost/test/unit_test.hpp>

#include <optional>
#include <ranges>

#include "ModelTestHelpers.h"

BOOST_AUTO_TEST_SUITE(CMetricDataGathererTest)

using namespace ml;
using namespace model;

namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TFeatureVec = std::vector<model_t::EFeature>;
using TSizeUInt64Pr = std::pair<std::size_t, std::uint64_t>;
using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
using TStrVec = std::vector<std::string>;
using TSizeFeatureDataPr = std::pair<std::size_t, SMetricFeatureData>;
using TSizeFeatureDataPrVec = std::vector<TSizeFeatureDataPr>;
using TFeatureSizeFeatureDataPrVecPr = std::pair<model_t::EFeature, TSizeFeatureDataPrVec>;
using TFeatureSizeFeatureDataPrVecPrVec = std::vector<TFeatureSizeFeatureDataPrVecPr>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TTimeDoublePrVecVec = std::vector<TTimeDoublePrVec>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;

std::size_t addPerson(const std::string& p,
                      CDataGatherer& gatherer,
                      CResourceMonitor& resourceMonitor,
                      std::size_t numInfluencers = 0) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    std::string const i("i");
    for (std::size_t j = 0; j < numInfluencers; ++j) {
        person.push_back(&i);
    }
    person.resize(gatherer.fieldsOfInterest().size(), nullptr);
    CEventData result;
    gatherer.processFields(person, result, resourceMonitor);
    return *result.personId();
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                double value) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    const std::string valueAsString(core::CStringUtils::typeToStringPrecise(
        value, core::CIEEE754::E_DoublePrecision));
    fieldValues.push_back(&valueAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                double lat,
                double lng,
                std::string_view delimiter) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    std::string latlngAsString;
    latlngAsString += core::CStringUtils::typeToStringPrecise(lat, core::CIEEE754::E_DoublePrecision);
    latlngAsString += delimiter;
    latlngAsString += core::CStringUtils::typeToStringPrecise(lng, core::CIEEE754::E_DoublePrecision);
    fieldValues.push_back(&latlngAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                double value,
                const std::string& influencer1,
                const std::string& influencer2) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    fieldValues.push_back(influencer1.empty() ? nullptr : &influencer1);
    fieldValues.push_back(influencer2.empty() ? nullptr : &influencer2);
    std::string const valueAsString(core::CStringUtils::typeToString(value));
    fieldValues.push_back(&valueAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

double doubleToStringToDouble(double value) {
    std::string const valueAsString(core::CStringUtils::typeToStringPrecise(
        value, core::CIEEE754::E_DoublePrecision));
    double result(0.0);
    core::CStringUtils::stringToType(valueAsString, result);
    return result;
}

void addArrivals(CDataGatherer& gatherer,
                 CResourceMonitor& resourceMonitor,
                 core_t::TTime time,
                 core_t::TTime increment,
                 const std::string& person,
                 const TDoubleVec& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        addArrival(gatherer, resourceMonitor, time + (i * increment), person, values[i]);
    }
}

double variance(const TDoubleVec& values, double& mean) {
    double total = 0.0;
    for (double const value : values) {
        total += value;
    }
    mean = total / static_cast<double>(values.size());

    total = 0.0;
    for (double const value : values) {
        double const x = value - mean;
        total += (x * x);
    }
    // Population variance, not sample variance (which is / n - 1)
    return total / static_cast<double>(values.size());
}

const CSearchKey KEY;
}

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testSingleSeries, CTestFixture) {
    // Test that the various statistics come back as we suspect.

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;

    std::array<TTimeDoublePr, 6> bucket1 = {
        {{1, 1.0}, {15, 2.1}, {180, 0.9}, {190, 1.5}, {400, 1.5}, {550, 2.0}}};
    std::array<TTimeDoublePr, 3> bucket2 = {{{600, 2.0}, {799, 2.2}, {1199, 1.8}}};
    std::array<TTimeDoublePr, 2> bucket3 = {{{1200, 2.1}, {1250, 2.5}}};
    std::array<TTimeDoublePr, 1> bucket4 = {{{1900, 3.5}}};
    std::array<TTimeDoublePr, 3> bucket5 = {{{2420, 3.5}, {2480, 3.2}, {2490, 3.8}}};
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        SModelParams const params(bucketLength);
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .sampleCountOverride(2U)
                                     .build();
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson("p", gatherer, m_ResourceMonitor));

        BOOST_REQUIRE_EQUAL(4, gatherer.numberFeatures());
        for (std::size_t i = 0; i < 4; ++i) {
            BOOST_REQUIRE_EQUAL(features[i], gatherer.feature(i));
        }

        BOOST_REQUIRE_EQUAL(1, gatherer.numberActivePeople());
        BOOST_REQUIRE_EQUAL(1, gatherer.numberByFieldValues());
        BOOST_REQUIRE_EQUAL(std::string("p"), gatherer.personName(0));
        BOOST_REQUIRE_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        BOOST_TEST_REQUIRE(gatherer.personId("p", pid));
        BOOST_REQUIRE_EQUAL(0, pid);
        BOOST_TEST_REQUIRE(!gatherer.personId("a.n.other p", pid));

        {
            addArrival(gatherer, m_ResourceMonitor, bucket1[0].first, "p",
                       bucket1[0].second);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[1].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[2].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
        }

        for (std::size_t i = 1; i < std::size(bucket1); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[i].first, "p",
                       bucket1[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData((startTime + bucketLength - 1), bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                0.9, featureData[1].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.1, featureData[2].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                9.0, featureData[3].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(
                std::string("[(8 [1.55] 1 2), (185 [1.2] 1 2), (475 [1.75] 1 2)]"),
                core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(
                std::string("[(8 [1] 1 2), (185 [0.9] 1 2), (475 [1.5] 1 2)]"),
                core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(
                std::string("[(8 [2.1] 1 2), (185 [1.5] 1 2), (475 [2] 1 2)]"),
                core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(0 [9] 1 6)]"),
                                core::CContainerPrinter::print(
                                    featureData[3].second[0].second.s_Samples));
            testPersistence(params, gatherer, model_t::E_Metric);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (const auto& value : bucket2) {
            addArrival(gatherer, m_ResourceMonitor, value.first, "p", value.second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + bucketLength);
            gatherer.featureData(startTime + bucketLength, bucketLength, featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.0, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.8, featureData[1].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.2, featureData[2].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                6.0, featureData[3].second[0].second.s_BucketValue->value()[0], 1e-14);
            BOOST_REQUIRE_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(std::string("[(700 [2.1] 1 2)]"),
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(700 [2] 1 2)]"),
                                core::CContainerPrinter::print(
                                    featureData[1].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(700 [2.2] 1 2)]"),
                                core::CContainerPrinter::print(
                                    featureData[2].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(600 [6] 1 3)]"),
                                core::CContainerPrinter::print(
                                    featureData[3].second[0].second.s_Samples));
            testPersistence(params, gatherer, model_t::E_Metric);
        }
    }

    // Test capture of sample measurement count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        SModelParams const params(bucketLength);
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .build();
        BOOST_REQUIRE_EQUAL(0, addPerson("p", gatherer, m_ResourceMonitor));

        TTimeDoublePrVecVec buckets;
        buckets.emplace_back(std::begin(bucket1), std::end(bucket1));
        buckets.emplace_back(std::begin(bucket2), std::end(bucket2));
        buckets.emplace_back(std::begin(bucket3), std::end(bucket3));
        buckets.emplace_back(std::begin(bucket4), std::end(bucket4));
        buckets.emplace_back(std::begin(bucket5), std::end(bucket5));

        for (std::size_t i = 0; i < buckets.size(); ++i) {
            LOG_DEBUG(<< "Processing bucket " << i);
            gatherer.timeNow(startTime + (i * bucketLength));
            const TTimeDoublePrVec& bucket = buckets[i];
            for (const auto& j : bucket) {
                addArrival(gatherer, m_ResourceMonitor, j.first, "p", j.second);
            }
        }

        BOOST_REQUIRE_EQUAL(4.0, gatherer.effectiveSampleCount(0));
        TFeatureSizeFeatureDataPrVecPrVec featureData;
        constexpr auto featureBucketStart = (startTime + (4 * bucketLength));
        gatherer.sampleNow(featureBucketStart);
        gatherer.featureData(featureBucketStart, bucketLength, featureData);
        BOOST_TEST_REQUIRE(!featureData.empty());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            3.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            3.2, featureData[1].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            3.8, featureData[2].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            10.5, featureData[3].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

        BOOST_REQUIRE_EQUAL(
            std::string("[(2323 [3.5] 1 4)]"),
            core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            std::string("[(2323 [3.2] 1 4)]"),
            core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            std::string("[(2323 [3.8] 1 4)]"),
            core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            std::string("[(2400 [10.5] 1 3)]"),
            core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
    }
}

BOOST_FIXTURE_TEST_CASE(testMultipleSeries, CTestFixture) {
    // Test that the various statistics come back as we suspect
    // for multiple people.

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    SModelParams const params(bucketLength);
    CDataGatherer gatherer =
        CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
            .build();
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(1, addPerson("p2", gatherer, m_ResourceMonitor));

    std::array bucket11 = {TTimeDoublePr{1, 1.0},   TTimeDoublePr{15, 2.1},
                           TTimeDoublePr{180, 0.9}, TTimeDoublePr{190, 1.5},
                           TTimeDoublePr{400, 1.5}, TTimeDoublePr{550, 2.0}};
    std::array bucket12 = {TTimeDoublePr{600, 2.0}, TTimeDoublePr{799, 2.2},
                           TTimeDoublePr{1199, 1.8}};
    std::array bucket13 = {TTimeDoublePr{1200, 2.1}, TTimeDoublePr{1250, 2.5}};
    std::array bucket14 = {TTimeDoublePr{1900, 3.5}};
    std::array bucket15 = {TTimeDoublePr{2420, 3.5}, TTimeDoublePr{2480, 3.2},
                           TTimeDoublePr{2490, 3.8}};
    TTimeDoublePrVecVec buckets1;
    buckets1.emplace_back(std::begin(bucket11), std::end(bucket11));
    buckets1.emplace_back(std::begin(bucket12), std::end(bucket12));
    buckets1.emplace_back(std::begin(bucket13), std::end(bucket13));
    buckets1.emplace_back(std::begin(bucket14), std::end(bucket14));
    buckets1.emplace_back(std::begin(bucket15), std::end(bucket15));

    std::array bucket21 = {TTimeDoublePr{1, 1.0},   TTimeDoublePr{5, 1.0},
                           TTimeDoublePr{15, 2.1},  TTimeDoublePr{25, 2.0},
                           TTimeDoublePr{180, 0.9}, TTimeDoublePr{190, 1.5},
                           TTimeDoublePr{400, 1.5}, TTimeDoublePr{550, 2.0}};
    std::array bucket22 = {TTimeDoublePr{600, 2.0}, TTimeDoublePr{605, 2.0},
                           TTimeDoublePr{609, 2.0}, TTimeDoublePr{799, 2.2},
                           TTimeDoublePr{1199, 1.8}};
    std::array bucket23 = {TTimeDoublePr{1200, 2.1}, TTimeDoublePr{1250, 2.5},
                           TTimeDoublePr{1255, 2.2}, TTimeDoublePr{1256, 2.4},
                           TTimeDoublePr{1300, 2.2}, TTimeDoublePr{1400, 2.5}};
    std::array bucket24 = {TTimeDoublePr{1900, 3.5}, TTimeDoublePr{1950, 3.5}};
    std::array bucket25 = {TTimeDoublePr{2420, 3.5}, TTimeDoublePr{2480, 2.9},
                           TTimeDoublePr{2490, 3.9}, TTimeDoublePr{2500, 3.4},
                           TTimeDoublePr{2550, 4.1}, TTimeDoublePr{2600, 3.8}};
    TTimeDoublePrVecVec buckets2;
    buckets2.emplace_back(std::begin(bucket21), std::end(bucket21));
    buckets2.emplace_back(std::begin(bucket22), std::end(bucket22));
    buckets2.emplace_back(std::begin(bucket23), std::end(bucket23));
    buckets2.emplace_back(std::begin(bucket24), std::end(bucket24));
    buckets2.emplace_back(std::begin(bucket25), std::end(bucket25));

    for (std::size_t i = 0; i < 5; ++i) {
        LOG_DEBUG(<< "Processing bucket " << i);
        gatherer.timeNow(startTime + (i * bucketLength));

        for (const TTimeDoublePrVec& bucket1 = buckets1[i]; const auto& j : bucket1) {
            addArrival(gatherer, m_ResourceMonitor, j.first, "p1", j.second);
        }

        const TTimeDoublePrVec& bucket2 = buckets2[i];
        TMeanAccumulator a;
        for (const auto& j : bucket2) {
            addArrival(gatherer, m_ResourceMonitor, j.first, "p2", j.second);
            a.add(j.second);
        }
    }

    BOOST_REQUIRE_CLOSE_ABSOLUTE(4.0, gatherer.effectiveSampleCount(0), 1e-10);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(6.0, gatherer.effectiveSampleCount(1), 1e-10);

    TSizeUInt64PrVec nonZeroCounts;
    gatherer.personNonZeroCounts(startTime + (4 * bucketLength), nonZeroCounts);
    BOOST_REQUIRE_EQUAL(std::string("[(0, 3), (1, 6)]"),
                        core::CContainerPrinter::print(nonZeroCounts));

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    constexpr auto featureBucketStart = (startTime + (4 * bucketLength));
    gatherer.sampleNow(featureBucketStart);
    gatherer.featureData(featureBucketStart, bucketLength, featureData);

    BOOST_TEST_REQUIRE(!featureData.empty());
    BOOST_REQUIRE_EQUAL(2, featureData[0].second.size());
    BOOST_REQUIRE_EQUAL(2, featureData[1].second.size());
    BOOST_REQUIRE_EQUAL(2, featureData[2].second.size());
    BOOST_REQUIRE_EQUAL(2, featureData[3].second.size());

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        3.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
    BOOST_REQUIRE_EQUAL(3.2, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.8, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(10.5, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

    BOOST_REQUIRE_EQUAL(
        std::string("[(2323 [3.5] 1 4)]"),
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2323 [3.2] 1 4)]"),
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2323 [3.8] 1 4)]"),
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2400 [10.5] 1 3)]"),
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        3.6, featureData[0].second[1].second.s_BucketValue->value()[0], 1e-10);
    BOOST_REQUIRE_EQUAL(2.9, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(4.1, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(21.6, featureData[3].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(false, featureData[0].second[1].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[1].second[1].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[2].second[1].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[3].second[1].second.s_IsInteger);

    BOOST_REQUIRE_EQUAL(
        std::string("[(2290 [3.45] 1 6)]"),
        core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2290 [2.9] 1 6)]"),
        core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2290 [3.9] 1 6)]"),
        core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2400 [21.6] 1 6)]"),
        core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));
    testPersistence(params, gatherer, model_t::E_Metric);

    // Remove person p1.
    TSizeVec peopleToRemove;
    peopleToRemove.push_back(0);
    gatherer.recyclePeople(peopleToRemove);

    BOOST_REQUIRE_EQUAL(1, gatherer.numberActivePeople());
    BOOST_REQUIRE_EQUAL(1, gatherer.numberByFieldValues());
    BOOST_REQUIRE_EQUAL(std::string("-"), gatherer.personName(0));
    BOOST_REQUIRE_EQUAL(std::string("p2"), gatherer.personName(1));
    std::size_t pid;
    BOOST_TEST_REQUIRE(gatherer.personId("p2", pid));
    BOOST_REQUIRE_EQUAL(1, pid);
    BOOST_TEST_REQUIRE(!gatherer.personId("p1", pid));

    BOOST_REQUIRE_EQUAL(0, gatherer.numberActiveAttributes());
    BOOST_REQUIRE_EQUAL(0, gatherer.numberOverFieldValues());

    gatherer.personNonZeroCounts(startTime + (4 * bucketLength), nonZeroCounts);
    BOOST_REQUIRE_EQUAL(std::string("[(1, 6)]"),
                        core::CContainerPrinter::print(nonZeroCounts));

    BOOST_REQUIRE_CLOSE_ABSOLUTE(6.0, gatherer.effectiveSampleCount(1), 1e-10);

    gatherer.featureData((startTime + (4 * bucketLength)), bucketLength, featureData);

    BOOST_TEST_REQUIRE(!featureData.empty());
    BOOST_REQUIRE_EQUAL(1, featureData[0].second.size());
    BOOST_REQUIRE_EQUAL(1, featureData[1].second.size());
    BOOST_REQUIRE_EQUAL(1, featureData[2].second.size());

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        3.6, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
    BOOST_REQUIRE_EQUAL(2.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(4.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(21.6, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

    BOOST_REQUIRE_EQUAL(
        std::string("[(2290 [3.45] 1 6)]"),
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2290 [2.9] 1 6)]"),
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2290 [3.9] 1 6)]"),
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(2400 [21.6] 1 6)]"),
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
}

BOOST_FIXTURE_TEST_CASE(testSampleCount, CTestFixture) {
    // Test that we set sensible sample counts for each person.

    // Person 1 has constant update rate of 4 values per bucket.
    // Person 2 has variable rate with mean of 2 values per bucket.

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;
    constexpr std::size_t numberBuckets = 3;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    SModelParams const params(bucketLength);
    CDataGatherer gatherer =
        CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
            .build();
    std::size_t const pid1 = addPerson("p1", gatherer, m_ResourceMonitor);
    std::size_t const pid2 = addPerson("p2", gatherer, m_ResourceMonitor);

    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < numberBuckets; ++i) {
        LOG_DEBUG(<< "Processing bucket " << i);
        gatherer.timeNow(startTime + (i * bucketLength));

        {
            LOG_DEBUG(<< "count p1 = 6");
            addArrival(gatherer, m_ResourceMonitor,
                       startTime + (i * bucketLength) + 20, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor,
                       startTime + (i * bucketLength) + 40, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor,
                       startTime + (i * bucketLength) + 60, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor,
                       startTime + (i * bucketLength) + 80, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor,
                       startTime + (i * bucketLength) + 100, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor,
                       startTime + (i * bucketLength) + 120, "p1", 1.0);
        }
        {
            TDoubleVec count;
            rng.generateUniformSamples(1.0, 5.0, 1, count);
            LOG_DEBUG(<< "count p2 = " << std::floor(count[0]));
            for (std::size_t j = 0; j < static_cast<std::size_t>(count[0]); ++j) {
                addArrival(gatherer, m_ResourceMonitor,
                           startTime + (i * bucketLength) + (100 * (j + 1)), "p2", 1.0);
            }
        }
    }
    gatherer.timeNow(startTime + (numberBuckets * bucketLength));

    LOG_DEBUG(<< "p1 sample count = " << gatherer.effectiveSampleCount(pid1));
    LOG_DEBUG(<< "p2 sample count = " << gatherer.effectiveSampleCount(pid2));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(6.0, gatherer.effectiveSampleCount(pid1), 1e-5);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.0, gatherer.effectiveSampleCount(pid2), 1.0 + 1e-5);

    for (std::size_t i = numberBuckets; i < 100; ++i) {
        gatherer.timeNow(startTime + (i * bucketLength));
        addArrival(gatherer, m_ResourceMonitor,
                   startTime + (i * bucketLength) + 10, "p1", 1.0);
    }
    LOG_DEBUG(<< "p1 sample count = " << gatherer.effectiveSampleCount(pid1));
    BOOST_REQUIRE_CLOSE_ABSOLUTE(2.0, gatherer.effectiveSampleCount(pid1), 0.5);
}

BOOST_FIXTURE_TEST_CASE(testRemovePeople, CTestFixture) {
    // Test various combinations of removed people.

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 3600;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    SModelParams const params(bucketLength);
    CDataGatherer gatherer =
        CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
            .build();
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(1, addPerson("p2", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(2, addPerson("p3", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(3, addPerson("p4", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(4, addPerson("p5", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(5, addPerson("p6", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(6, addPerson("p7", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(7, addPerson("p8", gatherer, m_ResourceMonitor));

    std::array<std::array<core_t::TTime, 8>, 8> times = {{
        {0, 0, 0, 0, 0, 0, 0, 0},
        {10, 20, 100, 0, 0, 0, 0, 0},
        {110, 120, 150, 170, 200, 0, 0, 0},
        {210, 220, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {400, 410, 480, 510, 530, 0, 0, 0},
        {1040, 1100, 1080, 1200, 1300, 1311, 2100, 0},
        {2200, 2500, 2600, 2610, 2702, 2731, 2710, 2862},
    }};
    std::array<std::array<double, 8>, 8> values = {{
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.0, 5.0, 6.0, 1.0, 0.2, 0.0, 0.0, 0.0},
        {2.1, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {4.0, 1.0, 8.0, 1.0, 0.3, 0.0, 0.0, 0.0},
        {4.0, 1.0, 8.0, 1.0, 0.3, 1.1, 10.3, 0.0},
        {2.0, 5.0, 6.0, 1.0, 0.2, 3.1, 7.1, 6.2},
    }};
    for (std::size_t i = 0; i < std::size(values); ++i) {
        for (std::size_t j = 0; j < std::size(values[i]); ++j) {
            if (values[i][j] > 0.0) {
                addArrival(gatherer, m_ResourceMonitor, startTime + times[i][j],
                           gatherer.personName(i), values[i][j]);
            }
        }
    }

    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(0);
        peopleToRemove.push_back(1);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer =
            CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
                .build();
        BOOST_REQUIRE_EQUAL(0, addPerson("p3", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(1, addPerson("p4", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(2, addPerson("p5", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(3, addPerson("p6", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(4, addPerson("p7", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(5, addPerson("p8", expectedGatherer, m_ResourceMonitor));

        const std::array people = {2, 3, 4, 5, 6, 7};
        for (std::size_t i = 0; i < std::size(people); ++i) {
            for (std::size_t j = 0; j < std::size(values[people[i]]); ++j) {
                if (values[people[i]][j] > 0.0) {
                    addArrival(expectedGatherer, m_ResourceMonitor,
                               startTime + times[people[i]][j],
                               expectedGatherer.personName(i), values[people[i]][j]);
                }
            }
        }

        LOG_DEBUG(<< "checksum          = " << gatherer.checksum());
        LOG_DEBUG(<< "expected checksum = " << expectedGatherer.checksum());
        BOOST_REQUIRE_EQUAL(expectedGatherer.checksum(), gatherer.checksum());
    }
    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(3);
        peopleToRemove.push_back(4);
        peopleToRemove.push_back(7);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer =
            CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
                .build();

        BOOST_REQUIRE_EQUAL(0, addPerson("p3", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(1, addPerson("p6", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(2, addPerson("p7", expectedGatherer, m_ResourceMonitor));

        constexpr std::array people = {2, 5, 6};
        for (std::size_t i = 0; i < std::size(people); ++i) {
            for (std::size_t j = 0; j < std::size(values[people[i]]); ++j) {
                if (values[people[i]][j] > 0.0) {
                    addArrival(expectedGatherer, m_ResourceMonitor,
                               startTime + times[people[i]][j],
                               expectedGatherer.personName(i), values[people[i]][j]);
                }
            }
        }

        LOG_DEBUG(<< "checksum          = " << gatherer.checksum());
        LOG_DEBUG(<< "expected checksum = " << expectedGatherer.checksum());
        BOOST_REQUIRE_EQUAL(expectedGatherer.checksum(), gatherer.checksum());
    }
    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(2);
        peopleToRemove.push_back(5);
        peopleToRemove.push_back(6);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer const expectedGatherer =
            CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
                .build();

        LOG_DEBUG(<< "checksum          = " << gatherer.checksum());
        LOG_DEBUG(<< "expected checksum = " << expectedGatherer.checksum());
        BOOST_REQUIRE_EQUAL(expectedGatherer.checksum(), gatherer.checksum());
    }

    TSizeVec expectedRecycled;
    expectedRecycled.push_back(addPerson("p1", gatherer, m_ResourceMonitor));
    expectedRecycled.push_back(addPerson("p7", gatherer, m_ResourceMonitor));

    LOG_DEBUG(<< "recycled          = " << gatherer.recycledPersonIds());
    LOG_DEBUG(<< "expected recycled = " << expectedRecycled);
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedRecycled),
                        core::CContainerPrinter::print(gatherer.recycledPersonIds()));
}

BOOST_FIXTURE_TEST_CASE(testSum, CTestFixture) {
    // Test sum and non-zero sum work as expected.

    constexpr core_t::TTime bucketLength = 600;
    const std::array bucketCounts = {2, 5, 2, 1, 0, 0, 4, 8, 0, 1};
    constexpr core_t::TTime startTime = 0;

    test::CRandomNumbers rng;

    TFeatureVec sumFeatures;
    sumFeatures.push_back(model_t::E_IndividualSumByBucketAndPerson);
    SModelParams const params(bucketLength);
    CDataGatherer sum = CDataGathererBuilder(model_t::E_Metric, sumFeatures,
                                             params, KEY, startTime)
                            .build();
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", sum, m_ResourceMonitor));

    TFeatureVec nonZeroSumFeatures;
    nonZeroSumFeatures.push_back(model_t::E_IndividualNonNullSumByBucketAndPerson);

    CDataGatherer nonZeroSum = CDataGathererBuilder(model_t::E_Metric, nonZeroSumFeatures,
                                                    params, KEY, startTime)
                                   .build();
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", nonZeroSum, m_ResourceMonitor));

    core_t::TTime bucketStart = startTime;
    for (const auto count : bucketCounts) {
        TDoubleVec times;
        rng.generateUniformSamples(0.0, bucketLength - 0.1, count, times);
        std::sort(times.begin(), times.end());

        TDoubleVec values;
        rng.generateNormalSamples(5.0, 4.0, count, values);

        double expected = 0.0;
        for (std::size_t j = 0; j < times.size(); ++j) {
            addArrival(sum, m_ResourceMonitor,
                       bucketStart + static_cast<core_t::TTime>(times[j]), "p1",
                       values[j]);
            addArrival(nonZeroSum, m_ResourceMonitor,
                       bucketStart + static_cast<core_t::TTime>(times[j]), "p1",
                       values[j]);
            expected += doubleToStringToDouble(values[j]);
        }

        LOG_DEBUG(<< "bucket: count = " << count << ", sum = " << expected);
        {
            TFeatureSizeFeatureDataPrVecPrVec data;
            sum.featureData(bucketStart, bucketLength, data);
            BOOST_REQUIRE_EQUAL(1, data.size());
            for (std::size_t j = 0; j < data.size(); ++j) {
                const TSizeFeatureDataPrVec& featureData = data[j].second;
                BOOST_REQUIRE_EQUAL(1, featureData.size());
                BOOST_REQUIRE_EQUAL(
                    expected, featureData[j].second.s_BucketValue->value()[0]);
                BOOST_REQUIRE_EQUAL(
                    static_cast<std::size_t>(1),
                    core::unwrap_ref(featureData[j].second.s_Samples).size());
                BOOST_REQUIRE_EQUAL(
                    expected,
                    core::unwrap_ref(featureData[j].second.s_Samples)[0].value()[0]);
            }
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec data;
            nonZeroSum.featureData(bucketStart, bucketLength, data);
            BOOST_REQUIRE_EQUAL(1, data.size());
            for (std::size_t j = 0; j < data.size(); ++j) {
                const TSizeFeatureDataPrVec& featureData = data[j].second;
                if (count == 0) {
                    BOOST_REQUIRE_EQUAL(0, featureData.size());
                } else {
                    BOOST_REQUIRE_EQUAL(1, featureData.size());
                    BOOST_REQUIRE_EQUAL(
                        expected, featureData[j].second.s_BucketValue->value()[0]);
                    BOOST_REQUIRE_EQUAL(
                        static_cast<std::size_t>(1),
                        core::unwrap_ref(featureData[j].second.s_Samples).size());
                    BOOST_REQUIRE_EQUAL(
                        expected,
                        core::unwrap_ref(featureData[j].second.s_Samples)[0].value()[0]);
                }
            }
        }

        bucketStart += bucketLength;
        sum.timeNow(bucketStart);
        nonZeroSum.timeNow(bucketStart);
    }
}

BOOST_FIXTURE_TEST_CASE(testSingleSeriesOutOfOrder, CTestFixture) {
    // Test that the various statistics come back as we suspect.

    constexpr core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 1;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    {
        constexpr std::array bucket1 = {
            TTimeDoublePr(1, 1.0), TTimeDoublePr(15, 2.1), TTimeDoublePr(180, 0.9),
            TTimeDoublePr(400, 1.5), TTimeDoublePr(550, 2.0)};
        constexpr core_t::TTime startTime = 0;
        constexpr std::array bucket2 = {TTimeDoublePr(600, 2.0),
                                        TTimeDoublePr(190, 1.5), TTimeDoublePr(799, 2.2),
                                        TTimeDoublePr(1199, 1.8)};
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .sampleCountOverride(2U)
                                     .build();
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson("p", gatherer, m_ResourceMonitor));

        BOOST_REQUIRE_EQUAL(4, gatherer.numberFeatures());
        for (std::size_t i = 0; i < 4; ++i) {
            BOOST_REQUIRE_EQUAL(features[i], gatherer.feature(i));
        }

        BOOST_REQUIRE_EQUAL(1, gatherer.numberActivePeople());
        BOOST_REQUIRE_EQUAL(1, gatherer.numberByFieldValues());
        BOOST_REQUIRE_EQUAL(std::string("p"), gatherer.personName(0));
        BOOST_REQUIRE_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        BOOST_TEST_REQUIRE(gatherer.personId("p", pid));
        BOOST_REQUIRE_EQUAL(0, pid);
        BOOST_TEST_REQUIRE(!gatherer.personId("a.n.other p", pid));

        {
            addArrival(gatherer, m_ResourceMonitor, bucket1[0].first, "p",
                       bucket1[0].second);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[1].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[2].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
        }

        for (std::size_t i = 1; i < std::size(bucket1); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[i].first, "p",
                       bucket1[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData((startTime + bucketLength - 1), bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_EQUAL(
                1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                0.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                2.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                7.5, featureData[3].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(std::string("[]"),
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[]"),
                                core::CContainerPrinter::print(
                                    featureData[1].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[]"),
                                core::CContainerPrinter::print(
                                    featureData[2].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(0 [7.5] 1 5)]"),
                                core::CContainerPrinter::print(
                                    featureData[3].second[0].second.s_Samples));
            testPersistence(params, gatherer, model_t::E_Metric);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (const auto& value : bucket2) {
            addArrival(gatherer, m_ResourceMonitor, value.first, "p", value.second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData(startTime, bucketLength, featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_EQUAL(
                1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                0.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                2.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                9.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(
                std::string("[(8 [1.55] 1 2), (257 [1.3] 0.666667 3)]"),
                core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(8 [1] 1 2), (257 [0.9] 1 3)]"),
                                core::CContainerPrinter::print(
                                    featureData[1].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(8 [2.1] 1 2), (257 [1.5] 1 3)]"),
                                core::CContainerPrinter::print(
                                    featureData[2].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL(std::string("[(0 [9] 1 6)]"),
                                core::CContainerPrinter::print(
                                    featureData[3].second[0].second.s_Samples));
            testPersistence(params, gatherer, model_t::E_Metric);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testResetBucketGivenSingleSeries, CTestFixture) {
    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    constexpr std::array data = {
        TTimeDoublePr(1, 1.0),    TTimeDoublePr(550, 2.0),
        TTimeDoublePr(600, 3.0),  TTimeDoublePr(700, 4.0),
        TTimeDoublePr(1000, 5.0), TTimeDoublePr(1200, 6.0)};

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    CDataGatherer gatherer =
        CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
            .sampleCountOverride(2U)
            .build();
    addPerson("p", gatherer, m_ResourceMonitor);

    for (const auto& value : data) {
        addArrival(gatherer, m_ResourceMonitor, value.first, "p", value.second);
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    constexpr TSizeSizePr pidCidPr(0, 0);

    gatherer.featureData(0, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(600, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(4.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(12.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(3),
                        gatherer.bucketCounts(600).find(pidCidPr)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr)->second);

    gatherer.resetBucket(600);
    addArrival(gatherer, m_ResourceMonitor, 610, "p", 2.0);
    addArrival(gatherer, m_ResourceMonitor, 620, "p", 3.0);

    gatherer.featureData(0, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(600, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(2.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr)->second);

    gatherer.sampleNow(0);
    gatherer.featureData(0, bucketLength, featureData);

    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(0 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));

    gatherer.sampleNow(600);
    gatherer.featureData(600, bucketLength, featureData);

    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(600 [5] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
}

BOOST_FIXTURE_TEST_CASE(testResetBucketGivenMultipleSeries, CTestFixture) {
    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    constexpr std::array data = {
        TTimeDoublePr(1, 1.0),    TTimeDoublePr(550, 2.0),
        TTimeDoublePr(600, 3.0),  TTimeDoublePr(700, 4.0),
        TTimeDoublePr(1000, 5.0), TTimeDoublePr(1200, 6.0)};

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    CDataGatherer gatherer =
        CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
            .sampleCountOverride(2U)
            .build();
    addPerson("p1", gatherer, m_ResourceMonitor);
    addPerson("p2", gatherer, m_ResourceMonitor);
    addPerson("p3", gatherer, m_ResourceMonitor);

    for (const auto& value : data) {
        for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
            addArrival(gatherer, m_ResourceMonitor, value.first,
                       gatherer.personName(pid), value.second);
        }
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    constexpr TSizeSizePr pidCidPr0(0, 0);
    constexpr TSizeSizePr pidCidPr1(1, 0);
    constexpr TSizeSizePr pidCidPr2(2, 0);

    gatherer.featureData(0, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr2)->second);

    gatherer.featureData(600, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(4.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(4.0, featureData[0].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(4.0, featureData[0].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(12.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(12.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(12.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(3),
                        gatherer.bucketCounts(600).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(3),
                        gatherer.bucketCounts(600).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(3),
                        gatherer.bucketCounts(600).find(pidCidPr2)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr2)->second);

    gatherer.resetBucket(600);
    for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
        addArrival(gatherer, m_ResourceMonitor, 610, gatherer.personName(pid), 2.0);
        addArrival(gatherer, m_ResourceMonitor, 620, gatherer.personName(pid), 3.0);
    }

    gatherer.featureData(0, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(0).find(pidCidPr2)->second);

    gatherer.featureData(600, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(2.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.5, featureData[0].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.5, featureData[0].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(600).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(600).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(2),
                        gatherer.bucketCounts(600).find(pidCidPr2)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(static_cast<std::uint64_t>(1),
                        gatherer.bucketCounts(1200).find(pidCidPr2)->second);

    gatherer.sampleNow(0);
    gatherer.featureData(0, bucketLength, featureData);

    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [1] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(276 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(0 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(0 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(0 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[2].second.s_Samples));

    gatherer.sampleNow(600);
    gatherer.featureData(600, bucketLength, featureData);

    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2.5] 1 2)]"),
        core::CContainerPrinter::print(featureData[0].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [2] 1 2)]"),
        core::CContainerPrinter::print(featureData[1].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(615 [3] 1 2)]"),
        core::CContainerPrinter::print(featureData[2].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(600 [5] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(600 [5] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        std::string("[(600 [5] 1 2)]"),
        core::CContainerPrinter::print(featureData[3].second[2].second.s_Samples));
}

BOOST_FIXTURE_TEST_CASE(testInfluenceStatistics, CTestFixture) {
    using TTimeDoubleStrStrTuple =
        boost::tuple<core_t::TTime, double, std::string, std::string>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TStrDoubleDoublePrPr = std::pair<std::string, TDoubleDoublePr>;
    using TStrDoubleDoublePrPrVec = std::vector<TStrDoubleDoublePrPr>;

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    constexpr std::array influencerNames_ = {"i1", "i2"};
    std::array<std::array<std::string, 3>, 2> influencerValues = {
        {{"i11", "i12", "i13"}, {"i21", "i22", "i23"}}};

    std::array data = {
        TTimeDoubleStrStrTuple(1, 1.0, influencerValues[0][0], influencerValues[1][0]), // Bucket 1
        TTimeDoubleStrStrTuple(150, 5.0, influencerValues[0][1], influencerValues[1][1]),
        TTimeDoubleStrStrTuple(150, 3.0, influencerValues[0][2], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(550, 2.0, influencerValues[0][0], influencerValues[1][0]),
        TTimeDoubleStrStrTuple(551, 2.1, influencerValues[0][1], influencerValues[1][1]),
        TTimeDoubleStrStrTuple(552, 4.0, influencerValues[0][2], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(554, 2.3, influencerValues[0][2], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(600, 3.0, influencerValues[0][1], influencerValues[1][0]), // Bucket 2
        TTimeDoubleStrStrTuple(660, 3.0, influencerValues[0][0], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(690, 7.1, influencerValues[0][1], ""),
        TTimeDoubleStrStrTuple(700, 4.0, influencerValues[0][0], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(800, 2.1, influencerValues[0][2], influencerValues[1][0]),
        TTimeDoubleStrStrTuple(900, 2.5, influencerValues[0][1], influencerValues[1][0]),
        TTimeDoubleStrStrTuple(1000, 5.0, influencerValues[0][1], influencerValues[1][0]),
        TTimeDoubleStrStrTuple(1200, 6.4, "", influencerValues[1][2]), // Bucket 3
        TTimeDoubleStrStrTuple(1210, 6.0, "", influencerValues[1][2]),
        TTimeDoubleStrStrTuple(1240, 7.0, "", influencerValues[1][1]),
        TTimeDoubleStrStrTuple(1600, 11.0, "", influencerValues[1][0]),
        TTimeDoubleStrStrTuple(1800, 11.0, "", "") // Sentinel
    };

    std::array expectedStatistics = {
        "[(i11, (1.5, 2)), (i12, (3.55, 2)), (i13, (3.1, 3)), (i21, (1.5, 2)), (i22, (3.55, 2)), (i23, (3.1, 3))]",
        "[(i11, (1.5, 2)), (i12, (3.55, 2)), (i13, (3.1, 3)), (i21, (1.5, 2)), (i22, (3.55, 2)), (i23, (3.1, 3))]",
        "[(i11, (1, 1)), (i12, (2.1, 1)), (i13, (2.3, 1)), (i21, (1, 1)), (i22, (2.1, 1)), (i23, (2.3, 1))]",
        "[(i11, (1, 1)), (i12, (2.1, 1)), (i13, (2.3, 1)), (i21, (1, 1)), (i22, (2.1, 1)), (i23, (2.3, 1))]",
        "[(i11, (2, 1)), (i12, (5, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (5, 1)), (i23, (4, 1))]",
        "[(i11, (2, 1)), (i12, (5, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (5, 1)), (i23, (4, 1))]",
        "[(i11, (3, 1)), (i12, (7.1, 1)), (i13, (9.3, 1)), (i21, (3, 1)), (i22, (7.1, 1)), (i23, (9.3, 1))]",
        "[(i11, (3, 1)), (i12, (7.1, 1)), (i13, (9.3, 1)), (i21, (3, 1)), (i22, (7.1, 1)), (i23, (9.3, 1))]",
        "[(i11, (3.5, 2)), (i12, (4.4, 4)), (i13, (2.1, 1)), (i21, (3.15, 4)), (i23, (3.5, 2))]",
        "[(i11, (3.5, 2)), (i12, (4.4, 4)), (i13, (2.1, 1)), (i21, (3.15, 4)), (i23, (3.5, 2))]",
        "[(i11, (3, 1)), (i12, (2.5, 1)), (i13, (2.1, 1)), (i21, (2.1, 1)), (i23, (3, 1))]",
        "[(i11, (3, 1)), (i12, (2.5, 1)), (i13, (2.1, 1)), (i21, (2.1, 1)), (i23, (3, 1))]",
        "[(i11, (4, 1)), (i12, (7.1, 1)), (i13, (2.1, 1)), (i21, (5, 1)), (i23, (4, 1))]",
        "[(i11, (4, 1)), (i12, (7.1, 1)), (i13, (2.1, 1)), (i21, (5, 1)), (i23, (4, 1))]",
        "[(i11, (7, 1)), (i12, (17.6, 1)), (i13, (2.1, 1)), (i21, (12.6, 1)), (i23, (7, 1))]",
        "[(i11, (7, 1)), (i12, (17.6, 1)), (i13, (2.1, 1)), (i21, (12.6, 1)), (i23, (7, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.2, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.2, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 1))]"};
    const char** expected = expectedStatistics.data();

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    TStrVec const influencerNames(std::begin(influencerNames_), std::end(influencerNames_));
    CDataGatherer gatherer =
        CDataGathererBuilder(model_t::E_Metric, features, params, KEY, startTime)
            .influenceFieldNames(influencerNames)
            .sampleCountOverride(2U)
            .build();
    addPerson("p1", gatherer, m_ResourceMonitor, influencerNames.size());
    addPerson("p2", gatherer, m_ResourceMonitor, influencerNames.size());

    core_t::TTime bucketStart = startTime;
    for (auto& i : data) {
        if (i.get<0>() >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "*** processing bucket ***");
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(bucketStart, bucketLength, featureData);
            for (auto const & [ feature, data_ ] : featureData) {
                LOG_DEBUG(<< "feature = " << model_t::print(feature));

                for (const auto & [ _, val ] : data_) {
                    TStrDoubleDoublePrPrVec statistics;
                    for (const auto& influenceValue : val.s_InfluenceValues) {
                        for (const auto & [ fst, snd ] : influenceValue) {
                            statistics.emplace_back(
                                fst, TDoubleDoublePr(snd.first[0], snd.second));
                        }
                    }
                    std::sort(statistics.begin(), statistics.end(),
                              maths::common::COrderings::SFirstLess());

                    LOG_DEBUG(<< "statistics = " << statistics);
                    LOG_DEBUG(<< "expected   = " << *expected);
                    BOOST_REQUIRE_EQUAL((*expected++),
                                        core::CContainerPrinter::print(statistics));
                }
            }

            bucketStart += bucketLength;
        }
        for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
            addArrival(gatherer, m_ResourceMonitor, i.get<0>(),
                       gatherer.personName(pid), i.get<1>(), i.get<2>(), i.get<3>());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testMultivariate, CTestFixture) {
    using TTimeDoubleDoubleTuple = boost::tuple<core_t::TTime, double, double>;
    using TTimeDoubleDoubleTupleVec = std::vector<TTimeDoubleDoubleTuple>;
    using TTimeDoubleDoubleTupleVecVec = std::vector<TTimeDoubleDoubleTupleVec>;

    static const std::string DELIMITER("__");

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;

    SModelParams params(bucketLength);
    params.s_MultivariateComponentDelimiter = DELIMITER;

    std::array bucket1 = {TTimeDoubleDoubleTuple(1, 1.0, 1.0),
                          TTimeDoubleDoubleTuple(15, 2.1, 2.0),
                          TTimeDoubleDoubleTuple(180, 0.9, 0.8),
                          TTimeDoubleDoubleTuple(190, 1.5, 1.4),
                          TTimeDoubleDoubleTuple(400, 1.5, 1.4),
                          TTimeDoubleDoubleTuple(550, 2.0, 1.8)};
    std::array bucket2 = {TTimeDoubleDoubleTuple(600, 2.0, 1.8),
                          TTimeDoubleDoubleTuple(799, 2.2, 2.0),
                          TTimeDoubleDoubleTuple(1199, 1.8, 1.6)};
    std::array bucket3 = {TTimeDoubleDoubleTuple(1200, 2.1, 2.0),
                          TTimeDoubleDoubleTuple(1250, 2.5, 2.4)};
    std::array bucket4 = {TTimeDoubleDoubleTuple(1900, 3.5, 3.2)};
    std::array bucket5 = {TTimeDoubleDoubleTuple(2420, 3.5, 3.2),
                          TTimeDoubleDoubleTuple(2480, 3.2, 3.0),
                          TTimeDoubleDoubleTuple(2490, 3.8, 3.8)};

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        const TStrVec influencerNames;
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .influenceFieldNames(influencerNames)
                                     .sampleCountOverride(2U)
                                     .build();
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson("p", gatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(1, gatherer.numberFeatures());
        BOOST_REQUIRE_EQUAL(features[0], gatherer.feature(0));

        BOOST_REQUIRE_EQUAL(1, gatherer.numberActivePeople());
        BOOST_REQUIRE_EQUAL(1, gatherer.numberByFieldValues());
        BOOST_REQUIRE_EQUAL(std::string("p"), gatherer.personName(0));
        BOOST_REQUIRE_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        BOOST_TEST_REQUIRE(gatherer.personId("p", pid));
        BOOST_REQUIRE_EQUAL(0, pid);
        BOOST_TEST_REQUIRE(!gatherer.personId("a.n.other p", pid));

        {
            addArrival(gatherer, m_ResourceMonitor, bucket1[0].get<0>(), "p",
                       bucket1[0].get<1>(), bucket1[0].get<2>(), DELIMITER);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[0].second[0].second.s_BucketValue->value()[1]);
            BOOST_REQUIRE_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
        }

        for (std::size_t i = 1; i < std::size(bucket1); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[i].get<0>(), "p",
                       bucket1[i].get<1>(), bucket1[i].get<2>(), DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData((startTime + bucketLength - 1), bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.4, featureData[0].second[0].second.s_BucketValue->value()[1], 1e-10);
            BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL(
                std::string("[(8 [1.55, 1.5] 1 2), (185 [1.2, 1.1] 1 2), (475 [1.75, 1.6] 1 2)]"),
                core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
            testPersistence(params, gatherer, model_t::E_Metric);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (const auto& value : bucket2) {
            addArrival(gatherer, m_ResourceMonitor, value.get<0>(), "p",
                       value.get<1>(), value.get<2>(), DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + bucketLength);
            gatherer.featureData(startTime + bucketLength, bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.0, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.8, featureData[0].second[0].second.s_BucketValue->value()[1], 1e-10);
            BOOST_REQUIRE_EQUAL(std::string("[(700 [2.1, 1.9] 1 2)]"),
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            testPersistence(params, gatherer, model_t::E_Metric);
        }

        gatherer.timeNow(startTime + (2 * bucketLength));
        for (const auto& value : bucket3) {
            addArrival(gatherer, m_ResourceMonitor, value.get<0>(), "p",
                       value.get<1>(), value.get<2>(), DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + (2 * bucketLength));
            gatherer.featureData(startTime + (2 * bucketLength), bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.3, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.2, featureData[0].second[0].second.s_BucketValue->value()[1], 1e-10);
            BOOST_REQUIRE_EQUAL(std::string("[(1200 [1.95, 1.8] 1 2)]"),
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
        }
    }

    // Test capture of sample measurement count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .build();
        BOOST_REQUIRE_EQUAL(0, addPerson("p", gatherer, m_ResourceMonitor));

        TTimeDoubleDoubleTupleVecVec buckets;
        buckets.emplace_back(std::begin(bucket1), std::end(bucket1));
        buckets.emplace_back(std::begin(bucket2), std::end(bucket2));
        buckets.emplace_back(std::begin(bucket3), std::end(bucket3));
        buckets.emplace_back(std::begin(bucket4), std::end(bucket4));
        buckets.emplace_back(std::begin(bucket5), std::end(bucket5));

        for (std::size_t i = 0; i < buckets.size(); ++i) {
            LOG_DEBUG(<< "Processing bucket " << i);
            gatherer.timeNow(startTime + (i * bucketLength));
            const TTimeDoubleDoubleTupleVec& bucket = buckets[i];
            for (const auto& j : bucket) {
                addArrival(gatherer, m_ResourceMonitor, j.get<0>(), "p",
                           j.get<1>(), j.get<2>(), DELIMITER);
            }
        }

        BOOST_REQUIRE_EQUAL(4.0, gatherer.effectiveSampleCount(0));
        TFeatureSizeFeatureDataPrVecPrVec featureData;
        constexpr auto featureBucketStart = (startTime + (4 * bucketLength));
        gatherer.sampleNow(featureBucketStart);
        gatherer.featureData(featureBucketStart, bucketLength, featureData);
        BOOST_TEST_REQUIRE(!featureData.empty());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            3.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
        BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
        LOG_DEBUG(<< "featureData = " << featureData);
        BOOST_REQUIRE_EQUAL(
            std::string("[(2323 [3.5, 3.3] 1 4)]"),
            core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    }
}

BOOST_FIXTURE_TEST_CASE(testStatisticsPersist, CTestFixture) {
    CGathererTools::TMeanGatherer::TMetricPartialStatistic stat(1);
    stat.add(TDoubleVec(1, 44.4), 1299196740, 1);
    stat.add(TDoubleVec(1, 5.5), 1299196741, 1);
    stat.add(TDoubleVec(1, 0.6), 1299196742, 1);

    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, [&stat](core::CJsonStatePersistInserter& inserter) {
            stat.persist(inserter);
        });

    core_t::TTime const origTime = stat.time();
    std::ostringstream restoredJson;
    core_t::TTime restoredTime;
    {
        std::istringstream origJsonStrm{"{\"topLevel\":" + origJson.str() + "}"};
        core::CJsonStateRestoreTraverser traverser(origJsonStrm);
        CGathererTools::TMeanGatherer::TMetricPartialStatistic restored(1);
        traverser.traverseSubLevel([&restored](auto&& PH1) mutable {
            return restored.restore(std::forward<decltype(PH1)>(PH1));
        });
        restoredTime = restored.time();
        core::CJsonStatePersistInserter::persist(
            restoredJson, [&restored](core::CJsonStatePersistInserter& inserter) {
                restored.persist(inserter);
            });
    }
    BOOST_REQUIRE_EQUAL(origJson.str(), restoredJson.str());
    BOOST_REQUIRE_EQUAL(origTime, restoredTime);
}

BOOST_FIXTURE_TEST_CASE(testVarp, CTestFixture) {
    core_t::TTime startTime = 100000;
    constexpr core_t::TTime bucketLength = 1000;
    const std::string person("p");
    const std::string inf1("i1");
    const std::string inf2("i2");
    const std::string inf3("i3");
    TDoubleVec values;
    SModelParams const params(bucketLength);

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualVarianceByPerson);
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .sampleCountOverride(2U)
                                     .build();
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson(person, gatherer, m_ResourceMonitor));

        BOOST_REQUIRE_EQUAL(1, gatherer.numberFeatures());

        {
            values.assign({5.0, 6.0, 3.0, 2.0, 4.0});
            addArrivals(gatherer, m_ResourceMonitor, startTime, 10, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            // Expect only 1 feature
            BOOST_REQUIRE_EQUAL(1, featureData.size());
            TFeatureSizeFeatureDataPrVecPr const fsfd = featureData[0];
            BOOST_REQUIRE_EQUAL(model_t::E_IndividualVarianceByPerson, fsfd.first);
            CSample::TDouble1Vec v =
                featureData[0].second[0].second.s_BucketValue->value();
            double expectedMean = 0;
            double const expectedVariance = variance(values, expectedMean);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[0], expectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[1], expectedMean, 0.0001);
        }
        startTime += bucketLength;
        {
            values.assign({115.0, 116.0, 117.0, 1111.5, 22.45, 2526.55634, 55.55, 14.723});
            addArrivals(gatherer, m_ResourceMonitor, startTime, 100, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            CSample::TDouble1Vec v =
                featureData[0].second[0].second.s_BucketValue->value();
            double expectedMean = 0;
            double const expectedVariance = variance(values, expectedMean);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[0], expectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[1], expectedMean, 0.0001);
        }
        startTime += bucketLength;
        gatherer.sampleNow(startTime);
        startTime += bucketLength;
        {
            values.assign({0.0});
            addArrivals(gatherer, m_ResourceMonitor, startTime, 100, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData[0].second[0].second.s_BucketValue);
        }
    }

    // Now test with influencers
    LOG_DEBUG(<< "Testing influencers");
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualVarianceByPerson);
        TStrVec influencerFieldNames;
        influencerFieldNames.emplace_back("i");
        influencerFieldNames.emplace_back("j");
        CDataGatherer gatherer = CDataGathererBuilder(model_t::E_Metric, features,
                                                      params, KEY, startTime)
                                     .influenceFieldNames(influencerFieldNames)
                                     .sampleCountOverride(2U)
                                     .build();
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson(person, gatherer, m_ResourceMonitor,
                                         influencerFieldNames.size()));

        TStrVec const testInf(gatherer.beginInfluencers(), gatherer.endInfluencers());

        LOG_DEBUG(<< "Influencer fields: " << testInf);
        LOG_DEBUG(<< "FOI: " << gatherer.fieldsOfInterest());

        BOOST_REQUIRE_EQUAL(1, gatherer.numberFeatures());
        {
            addArrival(gatherer, m_ResourceMonitor, startTime + 0, person, 5.0, inf1, inf2);
            addArrival(gatherer, m_ResourceMonitor, startTime + 100, person, 5.5, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 200, person, 5.9, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 300, person, 5.2, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 350, person, 5.1, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 400, person, 2.2, inf1, inf2);
            addArrival(gatherer, m_ResourceMonitor, startTime + 500, person, 4.9, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 600, person, 5.1, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 650, person, 1.0, "", "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 700, person, 5.0, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 800, person,
                       12.12, inf1, inf2);
            addArrival(gatherer, m_ResourceMonitor, startTime + 900, person, 5.2, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 950, person, 5.0, inf1, inf3);

            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            TFeatureSizeFeatureDataPrVecPr fsfd = featureData[0];
            BOOST_REQUIRE_EQUAL(model_t::E_IndividualVarianceByPerson, fsfd.first);

            CSample::TDouble1Vec v =
                featureData[0].second[0].second.s_BucketValue->value();
            values.assign({5.0, 5.5, 5.9, 5.2, 5.1, 2.2, 4.9, 5.1, 5.0, 12.12, 5.2, 5.0, 1.0});
            double expectedMean = 0;
            double const expectedVariance = variance(values, expectedMean);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[0], expectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[1], expectedMean, 0.0001);

            values.pop_back();
            double i1ExpectedMean = 0;
            double const i1ExpectedVariance = variance(values, i1ExpectedMean);

            values.clear();
            values.push_back(5.0);
            values.push_back(2.2);
            values.push_back(12.12);
            double i2ExpectedMean = 0;
            double const i2ExpectedVariance = variance(values, i2ExpectedMean);

            values.clear();
            values.push_back(5.0);
            double i3ExpectedMean = 0;
            double const i3ExpectedVariance = variance(values, i3ExpectedMean);

            SMetricFeatureData const mfd = fsfd.second[0].second;
            SMetricFeatureData::TStrCRefDouble1VecDoublePrPrVecVec ivs = mfd.s_InfluenceValues;
            LOG_DEBUG(<< "IVs: " << ivs);
            BOOST_REQUIRE_EQUAL(2, ivs.size());
            BOOST_REQUIRE_EQUAL(1, ivs[0].size());
            BOOST_REQUIRE_EQUAL(2, ivs[1].size());

            const SMetricFeatureData::TStrCRefDouble1VecDoublePrPr& ivs1 = ivs[0][0];
            BOOST_REQUIRE_EQUAL(inf1, ivs1.first.get());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(12.0, ivs1.second.second, 0.0001);
            BOOST_REQUIRE_EQUAL(2, ivs1.second.first.size());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(ivs1.second.first[0], i1ExpectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(ivs1.second.first[1], i1ExpectedMean, 0.0001);

            // The order of ivs2 and ivs3 seems to be backwards...
            const SMetricFeatureData::TStrCRefDouble1VecDoublePrPr& ivs2 = ivs[1][1];
            BOOST_REQUIRE_EQUAL(inf2, ivs2.first.get());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(3.0, ivs2.second.second, 0.0001);
            BOOST_REQUIRE_EQUAL(2, ivs2.second.first.size());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(ivs2.second.first[0], i2ExpectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(ivs2.second.first[1], i2ExpectedMean, 0.0001);

            const SMetricFeatureData::TStrCRefDouble1VecDoublePrPr& ivs3 = ivs[1][0];
            BOOST_REQUIRE_EQUAL(inf3, ivs3.first.get());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(1.0, ivs3.second.second, 0.0001);
            BOOST_REQUIRE_EQUAL(2, ivs3.second.first.size());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(ivs3.second.first[0], i3ExpectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(ivs3.second.first[1], i3ExpectedMean, 0.0001);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
