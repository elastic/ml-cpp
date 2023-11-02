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
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>
#include <core/UnwrapRef.h>

#include <maths/common/COrderings.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CFeatureData.h>
#include <model/CMetricBucketGatherer.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>
#include <model/SModelParams.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <optional>
#include <tuple>

namespace {

BOOST_AUTO_TEST_SUITE(CMetricDataGathererTest)

using namespace ml;
using namespace model;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeVecVec = std::vector<TTimeVec>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TFeatureVec = std::vector<model_t::EFeature>;
using TSizeUInt64Pr = std::pair<std::size_t, std::uint64_t>;
using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TSizeFeatureDataPr = std::pair<std::size_t, SMetricFeatureData>;
using TSizeFeatureDataPrVec = std::vector<TSizeFeatureDataPr>;
using TFeatureSizeFeatureDataPrVecPr = std::pair<model_t::EFeature, TSizeFeatureDataPrVec>;
using TFeatureSizeFeatureDataPrVecPrVec = std::vector<TFeatureSizeFeatureDataPrVecPr>;
using TOptionalDouble = std::optional<double>;
using TOptionalStr = std::optional<std::string>;
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
    std::string i("i");
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
    std::string valueAsString(core::CStringUtils::typeToStringPrecise(
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
                const std::string& delimiter) {
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
    std::string valueAsString(core::CStringUtils::typeToString(value));
    fieldValues.push_back(&valueAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

double doubleToStringToDouble(double value) {
    std::string valueAsString(core::CStringUtils::typeToStringPrecise(
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
    for (std::size_t i = 0; i < values.size(); ++i) {
        total += values[i];
    }
    mean = total / static_cast<double>(values.size());

    total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        double x = values[i] - mean;
        total += (x * x);
    }
    // Population variance, not sample variance (which is / n - 1)
    return total / static_cast<double>(values.size());
}

const CSearchKey KEY;
const std::string EMPTY_STRING;

void testPersistence(const SModelParams& params, const CDataGatherer& origGatherer) {
    // Test persistence. (We check for idempotency.)
    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origGatherer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "gatherer XML size " << origXml.size());
    LOG_TRACE(<< "gatherer XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CDataGatherer restoredGatherer(model_t::E_Metric, model_t::E_None, params,
                                   EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                   EMPTY_STRING, EMPTY_STRING, {}, KEY, traverser);

    BOOST_REQUIRE_EQUAL(origGatherer.checksum(), restoredGatherer.checksum());

    // The XML representation of the new filter should be the
    // same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredGatherer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testSingleSeries, CTestFixture) {
    // Test mean, min, max and sum gathering.

    const core_t::TTime startTime{0};
    const core_t::TTime bucketLength{600};
    TTimeDoublePrVec bucket1{{1, 1.0},   {15, 2.1},  {180, 0.9},
                             {190, 1.5}, {400, 1.5}, {550, 2.0}};
    TTimeDoublePrVec bucket2{{600, 2.0}, {799, 2.2}, {1199, 1.8}};

    TFeatureVec features{model_t::E_IndividualMeanByPerson,
                         model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                         model_t::E_IndividualSumByBucketAndPerson,
                         model_t::E_IndividualCountByBucketAndPerson}; // Should be ignored.
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, KEY, features, startTime);
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
        gatherer.featureData(startTime, featureData);
        LOG_DEBUG(<< "featureData = " << featureData);
        BOOST_REQUIRE_EQUAL(1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(true, featureData[1].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(true, featureData[2].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
    }

    for (std::size_t i = 1; i < bucket1.size(); ++i) {
        addArrival(gatherer, m_ResourceMonitor, bucket1[i].first, "p",
                   bucket1[i].second);
    }
    {
        TFeatureSizeFeatureDataPrVecPrVec featureData;
        gatherer.sampleNow(startTime);
        gatherer.featureData(core_t::TTime(startTime + bucketLength - 1), featureData);
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
        BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);
        BOOST_REQUIRE_EQUAL(
            "[(223 [1.5] 1 6)]",
            core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            "[(180 [0.9] 1 1)]",
            core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            "[(15 [2.1] 1 1)]",
            core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            "[(300 [9] 1 6)]",
            core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
        testPersistence(params, gatherer);
    }

    gatherer.timeNow(startTime + bucketLength);
    for (const auto& value : bucket2) {
        addArrival(gatherer, m_ResourceMonitor, value.first, "p", value.second);
    }
    {
        TFeatureSizeFeatureDataPrVecPrVec featureData;
        gatherer.sampleNow(startTime + bucketLength);
        gatherer.featureData(startTime + bucketLength, featureData);
        BOOST_TEST_REQUIRE(!featureData.empty());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            2.0, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            1.8, featureData[1].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            2.2, featureData[2].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            6.0, featureData[3].second[0].second.s_BucketValue->value()[0], 1e-14);
        BOOST_REQUIRE_EQUAL(
            "[(866 [2] 1 3)]",
            core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            "[(1199 [1.8] 1 1)]",
            core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            "[(799 [2.2] 1 1)]",
            core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
        BOOST_REQUIRE_EQUAL(
            "[(900 [6] 1 3)]",
            core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
        testPersistence(params, gatherer);
    }
}

BOOST_FIXTURE_TEST_CASE(testMultipleSeries, CTestFixture) {
    // Test mean, min, max and sum gathering for multiple time series.

    const core_t::TTime startTime{0};
    const core_t::TTime bucketLength{600};
    TFeatureVec features{model_t::E_IndividualMeanByPerson,
                         model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                         model_t::E_IndividualSumByBucketAndPerson};
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, KEY, features, startTime);
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(1, addPerson("p2", gatherer, m_ResourceMonitor));

    TTimeDoublePrVecVec buckets1{
        {{1, 1.0}, {15, 2.1}, {180, 0.9}, {190, 1.5}, {400, 1.5}, {550, 2.0}},
        {{600, 2.0}, {799, 2.2}, {1199, 1.8}}};

    TTimeDoublePrVecVec buckets2{
        {{1, 1.0}, {5, 1.0}, {15, 2.1}, {25, 2.0}, {180, 0.9}, {190, 1.5}, {400, 1.5}, {550, 2.0}},
        {{600, 2.2}, {605, 2.2}, {609, 2.2}, {799, 2.4}, {1199, 2.0}}};

    for (std::size_t i = 0; i < 2; ++i) {
        LOG_DEBUG(<< "Processing bucket " << i);
        gatherer.timeNow(startTime + i * bucketLength);
        for (auto[time, value] : buckets1[i]) {
            addArrival(gatherer, m_ResourceMonitor, time, "p1", value);
        }
        for (auto[time, value] : buckets2[i]) {
            addArrival(gatherer, m_ResourceMonitor, time, "p2", value);
        }
    }

    TSizeUInt64PrVec nonZeroCounts;
    gatherer.personNonZeroCounts(startTime + bucketLength, nonZeroCounts);
    BOOST_REQUIRE_EQUAL("[(0, 3), (1, 5)]", core::CContainerPrinter::print(nonZeroCounts));

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    gatherer.sampleNow(startTime + bucketLength);
    gatherer.featureData(startTime + bucketLength, featureData);

    BOOST_TEST_REQUIRE(!featureData.empty());
    BOOST_REQUIRE_EQUAL(2, featureData[0].second.size());
    BOOST_REQUIRE_EQUAL(2, featureData[1].second.size());
    BOOST_REQUIRE_EQUAL(2, featureData[2].second.size());
    BOOST_REQUIRE_EQUAL(2, featureData[3].second.size());
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        2.0, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
    BOOST_REQUIRE_EQUAL(1.8, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.2, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

    BOOST_REQUIRE_EQUAL("[(866 [2] 1 3)]", core::CContainerPrinter::print(
                                               featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(1199 [1.8] 1 1)]",
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(799 [2.2] 1 1)]",
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(900 [6] 1 3)]", core::CContainerPrinter::print(
                                               featureData[3].second[0].second.s_Samples));

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        2.2, featureData[0].second[1].second.s_BucketValue->value()[0], 1e-10);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.4, featureData[2].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(11.0, featureData[3].second[1].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(false, featureData[0].second[1].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[1].second[1].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[2].second[1].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[3].second[1].second.s_IsInteger);

    BOOST_REQUIRE_EQUAL(
        "[(762 [2.2] 1 5)]",
        core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(1199 [2] 1 1)]",
        core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(799 [2.4] 1 1)]",
        core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(900 [11] 1 5)]",
        core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));
    testPersistence(params, gatherer);

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

    gatherer.personNonZeroCounts(startTime + 4 * bucketLength, nonZeroCounts);
    BOOST_REQUIRE_EQUAL("[(1, 5)]", core::CContainerPrinter::print(nonZeroCounts));

    gatherer.featureData(startTime + bucketLength, featureData);

    BOOST_TEST_REQUIRE(!featureData.empty());
    BOOST_REQUIRE_EQUAL(1, featureData[0].second.size());
    BOOST_REQUIRE_EQUAL(1, featureData[1].second.size());
    BOOST_REQUIRE_EQUAL(1, featureData[2].second.size());

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        2.2, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.4, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(11.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
    BOOST_REQUIRE_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

    BOOST_REQUIRE_EQUAL(
        "[(762 [2.2] 1 5)]",
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(1199 [2] 1 1)]",
        core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(799 [2.4] 1 1)]",
        core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(900 [11] 1 5)]",
        core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
}

BOOST_FIXTURE_TEST_CASE(testRemovePeople, CTestFixture) {
    // Test various combinations of removed people.

    const core_t::TTime startTime{0};
    const core_t::TTime bucketLength{3600};

    TFeatureVec features{model_t::E_IndividualMeanByPerson,
                         model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                         model_t::E_IndividualSumByBucketAndPerson};
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, KEY, features, startTime);
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(1, addPerson("p2", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(2, addPerson("p3", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(3, addPerson("p4", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(4, addPerson("p5", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(5, addPerson("p6", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(6, addPerson("p7", gatherer, m_ResourceMonitor));
    BOOST_REQUIRE_EQUAL(7, addPerson("p8", gatherer, m_ResourceMonitor));

    TTimeVecVec times{{0, 0, 0, 0, 0, 0, 0, 0},
                      {10, 20, 100, 0, 0, 0, 0, 0},
                      {110, 120, 150, 170, 200, 0, 0, 0},
                      {210, 220, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0},
                      {400, 410, 480, 510, 530, 0, 0, 0},
                      {1040, 1100, 1080, 1200, 1300, 1311, 2100, 0},
                      {2200, 2500, 2600, 2610, 2702, 2731, 2710, 2862}};
    TDoubleVecVec values{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {1.0, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {2.0, 5.0, 6.0, 1.0, 0.2, 0.0, 0.0, 0.0},
                         {2.1, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {4.0, 1.0, 8.0, 1.0, 0.3, 0.0, 0.0, 0.0},
                         {4.0, 1.0, 8.0, 1.0, 0.3, 1.1, 10.3, 0.0},
                         {2.0, 5.0, 6.0, 1.0, 0.2, 3.1, 7.1, 6.2}};
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t j = 0; j < values[i].size(); ++j) {
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

        CDataGatherer expectedGatherer(model_t::E_Metric, model_t::E_None, params,
                                       EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                       EMPTY_STRING, {}, KEY, features, startTime);
        BOOST_REQUIRE_EQUAL(0, addPerson("p3", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(1, addPerson("p4", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(2, addPerson("p5", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(3, addPerson("p6", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(4, addPerson("p7", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(5, addPerson("p8", expectedGatherer, m_ResourceMonitor));

        std::size_t people[] = {2, 3, 4, 5, 6, 7};
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

        CDataGatherer expectedGatherer(model_t::E_Metric, model_t::E_None, params,
                                       EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                       EMPTY_STRING, {}, KEY, features, startTime);
        BOOST_REQUIRE_EQUAL(0, addPerson("p3", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(1, addPerson("p6", expectedGatherer, m_ResourceMonitor));
        BOOST_REQUIRE_EQUAL(2, addPerson("p7", expectedGatherer, m_ResourceMonitor));

        std::size_t people[] = {2, 5, 6};
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

        CDataGatherer expectedGatherer(model_t::E_Metric, model_t::E_None, params,
                                       EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                       EMPTY_STRING, {}, KEY, features, startTime);

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

    const core_t::TTime bucketLength = 600;
    const TSizeVec bucketCounts{2, 5, 2, 1, 0, 0, 4, 8, 0, 1};
    const core_t::TTime startTime = 0;

    test::CRandomNumbers rng;

    TFeatureVec sumFeatures{model_t::E_IndividualSumByBucketAndPerson};
    SModelParams params(bucketLength);
    CDataGatherer sum(model_t::E_Metric, model_t::E_None, params, EMPTY_STRING,
                      EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                      {}, KEY, sumFeatures, startTime);
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", sum, m_ResourceMonitor));

    TFeatureVec nonNullSumFeatures{model_t::E_IndividualNonNullSumByBucketAndPerson};

    CDataGatherer nonNullSum(model_t::E_Metric, model_t::E_None, params,
                             EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                             EMPTY_STRING, {}, KEY, nonNullSumFeatures, startTime);
    BOOST_REQUIRE_EQUAL(0, addPerson("p1", nonNullSum, m_ResourceMonitor));

    core_t::TTime bucketStart = startTime;
    for (auto count : bucketCounts) {
        TDoubleVec times;
        rng.generateUniformSamples(0.0, static_cast<double>(bucketLength - 0.1), count, times);
        std::sort(times.begin(), times.end());

        TDoubleVec values;
        rng.generateNormalSamples(5.0, 4.0, count, values);

        double expected = 0.0;
        for (std::size_t j = 0; j < times.size(); ++j) {
            addArrival(sum, m_ResourceMonitor,
                       bucketStart + static_cast<core_t::TTime>(times[j]), "p1",
                       values[j]);
            addArrival(nonNullSum, m_ResourceMonitor,
                       bucketStart + static_cast<core_t::TTime>(times[j]), "p1",
                       values[j]);
            expected += doubleToStringToDouble(values[j]);
        }

        LOG_DEBUG(<< "bucket: count = " << count << ", sum = " << expected);
        {
            TFeatureSizeFeatureDataPrVecPrVec data;
            sum.featureData(bucketStart, data);
            BOOST_REQUIRE_EQUAL(1, data.size());
            for (std::size_t j = 0; j < data.size(); ++j) {
                const TSizeFeatureDataPrVec& featureData = data[j].second;
                BOOST_REQUIRE_EQUAL(1, featureData.size());
                BOOST_REQUIRE_EQUAL(
                    expected, featureData[j].second.s_BucketValue->value()[0]);
                BOOST_REQUIRE_EQUAL(
                    std::size_t(1),
                    core::unwrap_ref(featureData[j].second.s_Samples).size());
                BOOST_REQUIRE_EQUAL(
                    expected,
                    core::unwrap_ref(featureData[j].second.s_Samples)[0].value()[0]);
            }
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec data;
            nonNullSum.featureData(bucketStart, data);
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
                        std::size_t(1),
                        core::unwrap_ref(featureData[j].second.s_Samples).size());
                    BOOST_REQUIRE_EQUAL(
                        expected,
                        core::unwrap_ref(featureData[j].second.s_Samples)[0].value()[0]);
                }
            }
        }

        bucketStart += bucketLength;
        sum.timeNow(bucketStart);
        nonNullSum.timeNow(bucketStart);
    }
}

BOOST_FIXTURE_TEST_CASE(testSingleSeriesOutOfOrder, CTestFixture) {
    // Test that the various statistics come back as we suspect.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 1;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TTimeDoublePrVec bucket1{{1, 1.0}, {15, 2.1}, {180, 0.9}, {400, 1.5}, {550, 2.0}};
    TTimeDoublePrVec bucket2{{600, 2.0}, {190, 1.5}, {799, 2.2}, {1199, 1.8}};

    {
        TFeatureVec features{model_t::E_IndividualMeanByPerson,
                             model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                             model_t::E_IndividualSumByBucketAndPerson,
                             model_t::E_IndividualCountByBucketAndPerson};
        CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, {}, KEY, features, startTime);
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
            gatherer.featureData(startTime, featureData);
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

        for (std::size_t i = 1; i < bucket1.size(); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[i].first, "p",
                       bucket1[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime + bucketLength - 1, featureData);
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
            BOOST_REQUIRE_EQUAL("[(229 [1.5] 1 5)]",
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL("[(180 [0.9] 1 1)]",
                                core::CContainerPrinter::print(
                                    featureData[1].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL("[(15 [2.1] 1 1)]",
                                core::CContainerPrinter::print(
                                    featureData[2].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL("[(300 [7.5] 1 5)]",
                                core::CContainerPrinter::print(
                                    featureData[3].second[0].second.s_Samples));
            testPersistence(params, gatherer);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (const auto& value : bucket2) {
            addArrival(gatherer, m_ResourceMonitor, value.first, "p", value.second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData(startTime, featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_EQUAL(
                1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                0.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                2.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                9.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL("[(223 [1.5] 1 6)]",
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL("[(180 [0.9] 1 1)]",
                                core::CContainerPrinter::print(
                                    featureData[1].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL("[(15 [2.1] 1 1)]",
                                core::CContainerPrinter::print(
                                    featureData[2].second[0].second.s_Samples));
            BOOST_REQUIRE_EQUAL("[(300 [9] 1 6)]",
                                core::CContainerPrinter::print(
                                    featureData[3].second[0].second.s_Samples));
            testPersistence(params, gatherer);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testResetBucketSingleSeries, CTestFixture) {
    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TTimeDoublePrVec data{{1, 1.0},   {550, 2.0},  {600, 3.0},
                          {700, 4.0}, {1000, 5.0}, {1200, 6.0}};

    TFeatureVec features{model_t::E_IndividualMeanByPerson,
                         model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                         model_t::E_IndividualSumByBucketAndPerson};
    CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, KEY, features, startTime);
    addPerson("p", gatherer, m_ResourceMonitor);

    for (const auto& value : data) {
        addArrival(gatherer, m_ResourceMonitor, value.first, "p", value.second);
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    TSizeSizePr pidCidPr(0, 0);

    gatherer.featureData(0, featureData);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(600, featureData);
    BOOST_REQUIRE_EQUAL(4.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(12.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(std::uint64_t(3),
                        gatherer.bucketCounts(600).find(pidCidPr)->second);

    gatherer.featureData(1200, featureData);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr)->second);

    gatherer.resetBucket(600);
    addArrival(gatherer, m_ResourceMonitor, 610, "p", 2.0);
    addArrival(gatherer, m_ResourceMonitor, 620, "p", 3.0);

    gatherer.featureData(0, featureData);
    BOOST_REQUIRE_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(600, featureData);
    BOOST_REQUIRE_EQUAL(2.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(2.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(3.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(5.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(1200, featureData);
    BOOST_REQUIRE_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr)->second);

    gatherer.sampleNow(0);
    gatherer.featureData(0, featureData);

    BOOST_REQUIRE_EQUAL(
        "[(276 [1.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(1 [1] 1 1)]", core::CContainerPrinter::print(
                                             featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(550 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(300 [3] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[0].second.s_Samples));

    gatherer.sampleNow(600);
    gatherer.featureData(600, featureData);

    BOOST_REQUIRE_EQUAL(
        "[(615 [2.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(610 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(620 [3] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(900 [5] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[0].second.s_Samples));
}

BOOST_FIXTURE_TEST_CASE(testResetBucketMultipleSeries, CTestFixture) {
    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TTimeDoublePrVec data{{1, 1.0},   {550, 2.0},  {600, 3.0},
                          {700, 4.0}, {1000, 5.0}, {1200, 6.0}};

    TFeatureVec features{model_t::E_IndividualMeanByPerson,
                         model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                         model_t::E_IndividualSumByBucketAndPerson};
    CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, KEY, features, startTime);
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
    TSizeSizePr pidCidPr0(0, 0);
    TSizeSizePr pidCidPr1(1, 0);
    TSizeSizePr pidCidPr2(2, 0);

    gatherer.featureData(0, featureData);
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
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr2)->second);

    gatherer.featureData(600, featureData);
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
    BOOST_REQUIRE_EQUAL(std::uint64_t(3),
                        gatherer.bucketCounts(600).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(3),
                        gatherer.bucketCounts(600).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(3),
                        gatherer.bucketCounts(600).find(pidCidPr2)->second);

    gatherer.featureData(1200, featureData);
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
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr2)->second);

    gatherer.resetBucket(600);
    for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
        addArrival(gatherer, m_ResourceMonitor, 610, gatherer.personName(pid), 2.0);
        addArrival(gatherer, m_ResourceMonitor, 620, gatherer.personName(pid), 3.0);
    }

    gatherer.featureData(0, featureData);
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
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(0).find(pidCidPr2)->second);

    gatherer.featureData(600, featureData);
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
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(600).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(600).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(2),
                        gatherer.bucketCounts(600).find(pidCidPr2)->second);

    gatherer.featureData(1200, featureData);
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
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr0)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr1)->second);
    BOOST_REQUIRE_EQUAL(std::uint64_t(1),
                        gatherer.bucketCounts(1200).find(pidCidPr2)->second);

    gatherer.sampleNow(0);
    gatherer.featureData(0, featureData);

    BOOST_REQUIRE_EQUAL(
        "[(276 [1.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(276 [1.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(276 [1.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(1 [1] 1 1)]", core::CContainerPrinter::print(
                                             featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(1 [1] 1 1)]", core::CContainerPrinter::print(
                                             featureData[1].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(1 [1] 1 1)]", core::CContainerPrinter::print(
                                             featureData[1].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(550 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(550 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(550 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(300 [3] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(300 [3] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(300 [3] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[2].second.s_Samples));

    gatherer.sampleNow(600);
    gatherer.featureData(600, featureData);

    BOOST_REQUIRE_EQUAL(
        "[(615 [2.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(615 [2.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL(
        "[(615 [2.5] 1 2)]",
        core::CContainerPrinter::print(featureData[0].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(610 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[1].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(610 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[1].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(610 [2] 1 1)]", core::CContainerPrinter::print(
                                               featureData[1].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(620 [3] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(620 [3] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(620 [3] 1 1)]", core::CContainerPrinter::print(
                                               featureData[2].second[2].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(900 [5] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[0].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(900 [5] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[1].second.s_Samples));
    BOOST_REQUIRE_EQUAL("[(900 [5] 1 2)]", core::CContainerPrinter::print(
                                               featureData[3].second[2].second.s_Samples));
}

BOOST_FIXTURE_TEST_CASE(testInfluenceStatistics, CTestFixture) {
    using TTimeDoubleStrStrTuple = std::tuple<core_t::TTime, double, std::string, std::string>;
    using TTimeDoubleStrStrTupleVec = std::vector<TTimeDoubleStrStrTuple>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TStrDoubleDoublePrPr = std::pair<std::string, TDoubleDoublePr>;
    using TStrDoubleDoublePrPrVec = std::vector<TStrDoubleDoublePrPr>;

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TStrVec influencerNames{"i1", "i2"};
    TStrVecVec influencerValues{{"i11", "i12", "i13"}, {"i21", "i22", "i23"}};

    TTimeDoubleStrStrTupleVec data{
        {1, 1.0, influencerValues[0][0], influencerValues[1][0]}, // Bucket 1
        {150, 5.0, influencerValues[0][1], influencerValues[1][1]},
        {150, 3.0, influencerValues[0][2], influencerValues[1][2]},
        {550, 2.0, influencerValues[0][0], influencerValues[1][0]},
        {551, 2.1, influencerValues[0][1], influencerValues[1][1]},
        {552, 4.0, influencerValues[0][2], influencerValues[1][2]},
        {554, 2.3, influencerValues[0][2], influencerValues[1][2]},
        {600, 3.0, influencerValues[0][1], influencerValues[1][0]}, // Bucket 2
        {660, 3.0, influencerValues[0][0], influencerValues[1][2]},
        {690, 7.1, influencerValues[0][1], ""},
        {700, 4.0, influencerValues[0][0], influencerValues[1][2]},
        {800, 2.1, influencerValues[0][2], influencerValues[1][0]},
        {900, 2.5, influencerValues[0][1], influencerValues[1][0]},
        {1000, 5.0, influencerValues[0][1], influencerValues[1][0]},
        {1200, 6.4, "", influencerValues[1][2]}, // Bucket 3
        {1210, 6.0, "", influencerValues[1][2]},
        {1240, 7.0, "", influencerValues[1][1]},
        {1600, 11.0, "", influencerValues[1][0]},
        TTimeDoubleStrStrTuple(1800, 11.0, "", "") // Sentinel
    };

    TStrVec expectedStatistics{
        "[(i11, (1.5, 2)), (i12, (3.55, 2)), (i13, (3.1, 3)), (i21, (1.5, 2)), (i22, (3.55, 2)), (i23, (3.1, 3))]", // Bucket 1
        "[(i11, (1.5, 2)), (i12, (3.55, 2)), (i13, (3.1, 3)), (i21, (1.5, 2)), (i22, (3.55, 2)), (i23, (3.1, 3))]",
        "[(i11, (1, 1)), (i12, (2.1, 1)), (i13, (2.3, 1)), (i21, (1, 1)), (i22, (2.1, 1)), (i23, (2.3, 1))]",
        "[(i11, (1, 1)), (i12, (2.1, 1)), (i13, (2.3, 1)), (i21, (1, 1)), (i22, (2.1, 1)), (i23, (2.3, 1))]",
        "[(i11, (2, 1)), (i12, (5, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (5, 1)), (i23, (4, 1))]",
        "[(i11, (2, 1)), (i12, (5, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (5, 1)), (i23, (4, 1))]",
        "[(i11, (3, 2)), (i12, (7.1, 2)), (i13, (9.3, 3)), (i21, (3, 2)), (i22, (7.1, 2)), (i23, (9.3, 3))]",
        "[(i11, (3, 2)), (i12, (7.1, 2)), (i13, (9.3, 3)), (i21, (3, 2)), (i22, (7.1, 2)), (i23, (9.3, 3))]",
        "[(i11, (3.5, 2)), (i12, (4.4, 4)), (i13, (2.1, 1)), (i21, (3.15, 4)), (i23, (3.5, 2))]", // Bucket 2
        "[(i11, (3.5, 2)), (i12, (4.4, 4)), (i13, (2.1, 1)), (i21, (3.15, 4)), (i23, (3.5, 2))]",
        "[(i11, (3, 1)), (i12, (2.5, 1)), (i13, (2.1, 1)), (i21, (2.1, 1)), (i23, (3, 1))]",
        "[(i11, (3, 1)), (i12, (2.5, 1)), (i13, (2.1, 1)), (i21, (2.1, 1)), (i23, (3, 1))]",
        "[(i11, (4, 1)), (i12, (7.1, 1)), (i13, (2.1, 1)), (i21, (5, 1)), (i23, (4, 1))]",
        "[(i11, (4, 1)), (i12, (7.1, 1)), (i13, (2.1, 1)), (i21, (5, 1)), (i23, (4, 1))]",
        "[(i11, (7, 2)), (i12, (17.6, 4)), (i13, (2.1, 1)), (i21, (12.6, 4)), (i23, (7, 2))]",
        "[(i11, (7, 2)), (i12, (17.6, 4)), (i13, (2.1, 1)), (i21, (12.6, 4)), (i23, (7, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.2, 2))]", // Bucket 3
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.2, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 2))]"};

    TFeatureVec features{model_t::E_IndividualMeanByPerson,
                         model_t::E_IndividualMinByPerson, model_t::E_IndividualMaxByPerson,
                         model_t::E_IndividualSumByBucketAndPerson};
    CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params, EMPTY_STRING,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           influencerNames, KEY, features, startTime);

    addPerson("p1", gatherer, m_ResourceMonitor, influencerNames.size());
    addPerson("p2", gatherer, m_ResourceMonitor, influencerNames.size());

    core_t::TTime bucketStart = startTime;
    auto expected = expectedStatistics.begin();
    for (std::size_t i = 0; i < std::size(data); ++i) {
        if (std::get<0>(data[i]) >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "*** processing bucket ***");
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(bucketStart, featureData);
            for (std::size_t j = 0; j < featureData.size(); ++j) {
                model_t::EFeature feature = featureData[j].first;
                LOG_DEBUG(<< "feature = " << model_t::print(feature));

                const TSizeFeatureDataPrVec& data_ = featureData[j].second;
                for (std::size_t k = 0; k < data_.size(); ++k) {
                    TStrDoubleDoublePrPrVec statistics;
                    for (std::size_t m = 0;
                         m < data_[k].second.s_InfluenceValues.size(); ++m) {
                        for (std::size_t n = 0;
                             n < data_[k].second.s_InfluenceValues[m].size(); ++n) {
                            statistics.emplace_back(
                                data_[k].second.s_InfluenceValues[m][n].first,
                                TDoubleDoublePr(
                                    data_[k]
                                        .second.s_InfluenceValues[m][n]
                                        .second.first[0],
                                    data_[k]
                                        .second.s_InfluenceValues[m][n]
                                        .second.second));
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
            addArrival(gatherer, m_ResourceMonitor, std::get<0>(data[i]),
                       gatherer.personName(pid), std::get<1>(data[i]),
                       std::get<2>(data[i]), std::get<3>(data[i]));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testMultivariate, CTestFixture) {
    using TTimeDoubleDoubleTuple = std::tuple<core_t::TTime, double, double>;
    using TTimeDoubleDoubleTupleVec = std::vector<TTimeDoubleDoubleTuple>;
    using TTimeDoubleDoubleTupleVecVec = std::vector<TTimeDoubleDoubleTupleVec>;

    static const std::string DELIMITER("__");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;

    SModelParams params(bucketLength);
    params.s_MultivariateComponentDelimiter = DELIMITER;

    TTimeDoubleDoubleTupleVec bucket1{{1, 1.0, 1.0},   {15, 2.1, 2.0},
                                      {180, 0.9, 0.8}, {190, 1.5, 1.4},
                                      {400, 1.5, 1.4}, {550, 2.0, 1.8}};
    TTimeDoubleDoubleTupleVec bucket2{{600, 2.0, 1.8}, {799, 2.2, 2.0}, {1199, 1.8, 1.6}};
    TTimeDoubleDoubleTupleVec bucket3{{1200, 2.1, 2.0}, {1250, 2.5, 2.4}};
    TTimeDoubleDoubleTupleVec bucket4{{1900, 3.5, 3.2}};
    TTimeDoubleDoubleTupleVec bucket5{{2420, 3.5, 3.2}, {2480, 3.2, 3.0}, {2490, 3.8, 3.8}};

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        TStrVec influencerNames;
        CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               influencerNames, KEY, features, startTime);
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
            addArrival(gatherer, m_ResourceMonitor, std::get<0>(bucket1[0]), "p",
                       std::get<1>(bucket1[0]), std::get<2>(bucket1[0]), DELIMITER);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            BOOST_REQUIRE_EQUAL(
                1.0, featureData[0].second[0].second.s_BucketValue->value()[1]);
            BOOST_REQUIRE_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
        }

        for (std::size_t i = 1; i < bucket1.size(); ++i) {
            addArrival(gatherer, m_ResourceMonitor, std::get<0>(bucket1[i]), "p",
                       std::get<1>(bucket1[i]), std::get<2>(bucket1[i]), DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData(core_t::TTime(startTime + bucketLength - 1), featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.4, featureData[0].second[0].second.s_BucketValue->value()[1], 1e-10);
            BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
            BOOST_REQUIRE_EQUAL("[(223 [1.5, 1.4] 1 6)]",
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            testPersistence(params, gatherer);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (const auto& value : bucket2) {
            addArrival(gatherer, m_ResourceMonitor, std::get<0>(value), "p",
                       std::get<1>(value), std::get<2>(value), DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + bucketLength);
            gatherer.featureData(startTime + bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.0, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                1.8, featureData[0].second[0].second.s_BucketValue->value()[1], 1e-10);
            BOOST_REQUIRE_EQUAL("[(866 [2, 1.8] 1 3)]",
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
            testPersistence(params, gatherer);
        }

        gatherer.timeNow(startTime + 2 * bucketLength);
        for (const auto& value : bucket3) {
            addArrival(gatherer, m_ResourceMonitor, std::get<0>(value), "p",
                       std::get<1>(value), std::get<2>(value), DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + 2 * bucketLength);
            gatherer.featureData(startTime + 2 * bucketLength, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            BOOST_TEST_REQUIRE(!featureData.empty());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.3, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                2.2, featureData[0].second[0].second.s_BucketValue->value()[1], 1e-10);
            BOOST_REQUIRE_EQUAL("[(1225 [2.3, 2.2] 1 2)]",
                                core::CContainerPrinter::print(
                                    featureData[0].second[0].second.s_Samples));
        }
    }

    // Test capture of sample measurement count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, {}, KEY, features, startTime);
        BOOST_REQUIRE_EQUAL(0, addPerson("p", gatherer, m_ResourceMonitor));

        TTimeDoubleDoubleTupleVecVec buckets;
        buckets.emplace_back(std::begin(bucket1), std::end(bucket1));
        buckets.emplace_back(std::begin(bucket2), std::end(bucket2));
        buckets.emplace_back(std::begin(bucket3), std::end(bucket3));
        buckets.emplace_back(std::begin(bucket4), std::end(bucket4));
        buckets.emplace_back(std::begin(bucket5), std::end(bucket5));

        for (std::size_t i = 0; i < buckets.size(); ++i) {
            LOG_DEBUG(<< "Processing bucket " << i);
            gatherer.timeNow(startTime + i * bucketLength);
            const TTimeDoubleDoubleTupleVec& bucket = buckets[i];
            for (std::size_t j = 0; j < bucket.size(); ++j) {
                addArrival(gatherer, m_ResourceMonitor, std::get<0>(bucket[j]), "p",
                           std::get<1>(bucket[j]), std::get<2>(bucket[j]), DELIMITER);
            }
        }

        TFeatureSizeFeatureDataPrVecPrVec featureData;
        core_t::TTime featureBucketStart = core_t::TTime(startTime + 4 * bucketLength);
        gatherer.sampleNow(featureBucketStart);
        gatherer.featureData(featureBucketStart, featureData);
        BOOST_TEST_REQUIRE(!featureData.empty());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(
            3.5, featureData[0].second[0].second.s_BucketValue->value()[0], 1e-10);
        BOOST_REQUIRE_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
        LOG_DEBUG(<< "featureData = " << featureData);
        BOOST_REQUIRE_EQUAL(
            "[(2463 [3.5, 3.333333] 1 3)]",
            core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    }
}

BOOST_FIXTURE_TEST_CASE(testVarp, CTestFixture) {
    core_t::TTime startTime = 100000;
    const core_t::TTime bucketLength = 1000;
    const std::string person("p");
    const std::string person2("q");
    const std::string person3("r");
    const std::string inf1("i1");
    const std::string inf2("i2");
    const std::string inf3("i3");
    TDoubleVec values;
    SModelParams params(bucketLength);

    {
        TFeatureVec features{model_t::E_IndividualVarianceByPerson};
        CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, {}, KEY, features, startTime);
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson(person, gatherer, m_ResourceMonitor));

        BOOST_REQUIRE_EQUAL(1, gatherer.numberFeatures());

        {
            values.assign({5.0, 6.0, 3.0, 2.0, 4.0});
            addArrivals(gatherer, m_ResourceMonitor, startTime, 10, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, featureData);
            // Expect only 1 feature
            BOOST_REQUIRE_EQUAL(1, featureData.size());
            TFeatureSizeFeatureDataPrVecPr fsfd = featureData[0];
            BOOST_REQUIRE_EQUAL(model_t::E_IndividualVarianceByPerson, fsfd.first);
            CSample::TDouble1Vec v =
                featureData[0].second[0].second.s_BucketValue->value();
            double expectedMean = 0;
            double expectedVariance = variance(values, expectedMean);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[0], expectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[1], expectedMean, 0.0001);
        }
        startTime += bucketLength;
        {
            values.assign({115.0, 116.0, 117.0, 1111.5, 22.45, 2526.55634, 55.55, 14.723});
            addArrivals(gatherer, m_ResourceMonitor, startTime, 100, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, featureData);
            LOG_DEBUG(<< "featureData = " << featureData);
            auto v = featureData[0].second[0].second.s_BucketValue->value();
            double expectedMean = 0;
            double expectedVariance = variance(values, expectedMean);
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
            gatherer.featureData(startTime, featureData);
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
        influencerFieldNames.push_back("i");
        influencerFieldNames.push_back("j");
        CDataGatherer gatherer(model_t::E_Metric, model_t::E_None, params, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               influencerFieldNames, KEY, features, startTime);
        BOOST_TEST_REQUIRE(!gatherer.isPopulation());
        BOOST_REQUIRE_EQUAL(0, addPerson(person, gatherer, m_ResourceMonitor,
                                         influencerFieldNames.size()));

        TStrVec testInf(gatherer.beginInfluencers(), gatherer.endInfluencers());

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
            gatherer.featureData(startTime, featureData);
            TFeatureSizeFeatureDataPrVecPr fsfd = featureData[0];
            BOOST_REQUIRE_EQUAL(model_t::E_IndividualVarianceByPerson, fsfd.first);

            CSample::TDouble1Vec v =
                featureData[0].second[0].second.s_BucketValue->value();
            values.assign({5.0, 5.5, 5.9, 5.2, 5.1, 2.2, 4.9, 5.1, 5.0, 12.12, 5.2, 5.0, 1.0});
            double expectedMean = 0;
            double expectedVariance = variance(values, expectedMean);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[0], expectedVariance, 0.0001);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(v[1], expectedMean, 0.0001);

            values.pop_back();
            double i1ExpectedMean = 0;
            double i1ExpectedVariance = variance(values, i1ExpectedMean);

            values.clear();
            values.push_back(5.0);
            values.push_back(2.2);
            values.push_back(12.12);
            double i2ExpectedMean = 0;
            double i2ExpectedVariance = variance(values, i2ExpectedMean);

            values.clear();
            values.push_back(5.0);
            double i3ExpectedMean = 0;
            double i3ExpectedVariance = variance(values, i3ExpectedMean);

            SMetricFeatureData mfd = fsfd.second[0].second;
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
}
