/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include "CMetricDataGathererTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CGathererTools.h>
#include <model/CMetricBucketGatherer.h>
#include <model/CModelParams.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>

#include <test/CRandomNumbers.h>

#include <boost/optional.hpp>
#include <boost/range.hpp>

using namespace ml;
using namespace model;

namespace {
typedef std::vector<double> TDoubleVec;
typedef std::vector<std::size_t> TSizeVec;
typedef std::pair<std::size_t, std::size_t> TSizeSizePr;
typedef std::vector<model_t::EFeature> TFeatureVec;
typedef std::pair<std::size_t, uint64_t> TSizeUInt64Pr;
typedef std::vector<TSizeUInt64Pr> TSizeUInt64PrVec;
typedef std::vector<std::string> TStrVec;
typedef std::pair<std::size_t, SMetricFeatureData> TSizeFeatureDataPr;
typedef std::vector<TSizeFeatureDataPr> TSizeFeatureDataPrVec;
typedef std::pair<model_t::EFeature, TSizeFeatureDataPrVec> TFeatureSizeFeatureDataPrVecPr;
typedef std::vector<TFeatureSizeFeatureDataPrVecPr> TFeatureSizeFeatureDataPrVecPrVec;
typedef boost::optional<double> TOptionalDouble;
typedef boost::optional<std::string> TOptionalStr;
typedef std::pair<core_t::TTime, double> TTimeDoublePr;
typedef std::vector<TTimeDoublePr> TTimeDoublePrVec;
typedef std::vector<TTimeDoublePrVec> TTimeDoublePrVecVec;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

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
    person.resize(gatherer.fieldsOfInterest().size(), 0);
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
    std::string valueAsString(
        core::CStringUtils::typeToStringPrecise(value, core::CIEEE754::E_DoublePrecision));
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
    latlngAsString +=
        core::CStringUtils::typeToStringPrecise(lat, core::CIEEE754::E_DoublePrecision);
    latlngAsString += delimiter;
    latlngAsString +=
        core::CStringUtils::typeToStringPrecise(lng, core::CIEEE754::E_DoublePrecision);
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
    fieldValues.push_back(influencer1.empty() ? 0 : &influencer1);
    fieldValues.push_back(influencer2.empty() ? 0 : &influencer2);
    std::string valueAsString(core::CStringUtils::typeToString(value));
    fieldValues.push_back(&valueAsString);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

double doubleToStringToDouble(double value) {
    std::string valueAsString(
        core::CStringUtils::typeToStringPrecise(value, core::CIEEE754::E_DoublePrecision));
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
}

void CMetricDataGathererTest::singleSeriesTests(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::singleSeriesTests ***");

    // Test that the various statistics come back as we suspect.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;

    TTimeDoublePr bucket1[] = {TTimeDoublePr(1, 1.0),
                               TTimeDoublePr(15, 2.1),
                               TTimeDoublePr(180, 0.9),
                               TTimeDoublePr(190, 1.5),
                               TTimeDoublePr(400, 1.5),
                               TTimeDoublePr(550, 2.0)};
    TTimeDoublePr bucket2[] = {TTimeDoublePr(600, 2.0),
                               TTimeDoublePr(799, 2.2),
                               TTimeDoublePr(1199, 1.8)};
    TTimeDoublePr bucket3[] = {TTimeDoublePr(1200, 2.1), TTimeDoublePr(1250, 2.5)};
    TTimeDoublePr bucket4[] = {
        TTimeDoublePr(1900, 3.5),
    };
    TTimeDoublePr bucket5[] = {TTimeDoublePr(2420, 3.5),
                               TTimeDoublePr(2480, 3.2),
                               TTimeDoublePr(2490, 3.8)};
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        SModelParams params(bucketLength);
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               KEY,
                               features,
                               startTime,
                               2u);
        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

        CPPUNIT_ASSERT_EQUAL(std::size_t(4), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 4; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), pid);
        CPPUNIT_ASSERT(!gatherer.personId("a.n.other p", pid));

        {
            addArrival(gatherer, m_ResourceMonitor, bucket1[0].first, "p", bucket1[0].second);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[1].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[2].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
        }

        for (size_t i = 1; i < boost::size(bucket1); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[i].first, "p", bucket1[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData(core_t::TTime(startTime + bucketLength - 1),
                                 bucketLength,
                                 featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(0.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(2.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(9.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(std::string("[(8 [1.55] 1 2), (185 [1.2] 1 2), (475 [1.75] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[0].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(8 [1] 1 2), (185 [0.9] 1 2), (475 [1.5] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[1].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(8 [2.1] 1 2), (185 [1.5] 1 2), (475 [2] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[2].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(0 [9] 1 6)]"),
                                 core::CContainerPrinter::print(
                                     featureData[3].second[0].second.s_Samples));

            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                gatherer.acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            LOG_DEBUG("gatherer XML representation:\n" << origXml);

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);

            CDataGatherer restoredGatherer(model_t::E_Metric,
                                           model_t::E_None,
                                           params,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           TStrVec(),
                                           false,
                                           KEY,
                                           traverser);

            // The XML representation of the new filter should be the
            // same as the original
            std::string newXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                restoredGatherer.acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (size_t i = 0; i < boost::size(bucket2); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket2[i].first, "p", bucket2[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + bucketLength);
            gatherer.featureData(startTime + bucketLength, bucketLength, featureData);
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_EQUAL(2.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.8, featureData[1].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(2.2, featureData[2].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(std::string("[(700 [2.1] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[0].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(700 [2] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[1].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(700 [2.2] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[2].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(600 [6] 1 3)]"),
                                 core::CContainerPrinter::print(
                                     featureData[3].second[0].second.s_Samples));

            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                gatherer.acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            LOG_DEBUG("model XML representation:\n" << origXml);

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);

            CDataGatherer restoredGatherer(model_t::E_Metric,
                                           model_t::E_None,
                                           params,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           TStrVec(),
                                           false,
                                           KEY,
                                           traverser);

            // The XML representation of the new filter should be the
            // same as the original
            std::string newXml;
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                restoredGatherer.acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }
    }

    // Test capture of sample measurement count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        SModelParams params(bucketLength);
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               KEY,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

        TTimeDoublePrVecVec buckets;
        buckets.push_back(TTimeDoublePrVec(boost::begin(bucket1), boost::end(bucket1)));
        buckets.push_back(TTimeDoublePrVec(boost::begin(bucket2), boost::end(bucket2)));
        buckets.push_back(TTimeDoublePrVec(boost::begin(bucket3), boost::end(bucket3)));
        buckets.push_back(TTimeDoublePrVec(boost::begin(bucket4), boost::end(bucket4)));
        buckets.push_back(TTimeDoublePrVec(boost::begin(bucket5), boost::end(bucket5)));

        for (std::size_t i = 0u; i < buckets.size(); ++i) {
            LOG_DEBUG("Processing bucket " << i);
            gatherer.timeNow(startTime + i * bucketLength);
            const TTimeDoublePrVec& bucket = buckets[i];
            for (std::size_t j = 0u; j < bucket.size(); ++j) {
                addArrival(gatherer, m_ResourceMonitor, bucket[j].first, "p", bucket[j].second);
            }
        }

        CPPUNIT_ASSERT_EQUAL(4.0, gatherer.effectiveSampleCount(0));
        TFeatureSizeFeatureDataPrVecPrVec featureData;
        core_t::TTime featureBucketStart = core_t::TTime(startTime + 4 * bucketLength);
        gatherer.sampleNow(featureBucketStart);
        gatherer.featureData(featureBucketStart, bucketLength, featureData);
        CPPUNIT_ASSERT(!featureData.empty());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.5,
                                     featureData[0].second[0].second.s_BucketValue->value()[0],
                                     1e-10);
        CPPUNIT_ASSERT_EQUAL(3.2, featureData[1].second[0].second.s_BucketValue->value()[0]);
        CPPUNIT_ASSERT_EQUAL(3.8, featureData[2].second[0].second.s_BucketValue->value()[0]);
        CPPUNIT_ASSERT_EQUAL(10.5, featureData[3].second[0].second.s_BucketValue->value()[0]);
        CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
        CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
        CPPUNIT_ASSERT_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
        CPPUNIT_ASSERT_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

        CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.5] 1 4)]"),
                             core::CContainerPrinter::print(
                                 featureData[0].second[0].second.s_Samples));
        CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.2] 1 4)]"),
                             core::CContainerPrinter::print(
                                 featureData[1].second[0].second.s_Samples));
        CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.8] 1 4)]"),
                             core::CContainerPrinter::print(
                                 featureData[2].second[0].second.s_Samples));
        CPPUNIT_ASSERT_EQUAL(std::string("[(2400 [10.5] 1 3)]"),
                             core::CContainerPrinter::print(
                                 featureData[3].second[0].second.s_Samples));
    }
}

void CMetricDataGathererTest::multipleSeriesTests(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::multipleSeriesTests ***");

    // Test that the various statistics come back as we suspect
    // for multiple people.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_Metric,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           TStrVec(),
                           false,
                           KEY,
                           features,
                           startTime,
                           0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p1", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson("p2", gatherer, m_ResourceMonitor));

    TTimeDoublePr bucket11[] = {TTimeDoublePr(1, 1.0),
                                TTimeDoublePr(15, 2.1),
                                TTimeDoublePr(180, 0.9),
                                TTimeDoublePr(190, 1.5),
                                TTimeDoublePr(400, 1.5),
                                TTimeDoublePr(550, 2.0)};
    TTimeDoublePr bucket12[] = {TTimeDoublePr(600, 2.0),
                                TTimeDoublePr(799, 2.2),
                                TTimeDoublePr(1199, 1.8)};
    TTimeDoublePr bucket13[] = {TTimeDoublePr(1200, 2.1), TTimeDoublePr(1250, 2.5)};
    TTimeDoublePr bucket14[] = {
        TTimeDoublePr(1900, 3.5),
    };
    TTimeDoublePr bucket15[] = {TTimeDoublePr(2420, 3.5),
                                TTimeDoublePr(2480, 3.2),
                                TTimeDoublePr(2490, 3.8)};
    TTimeDoublePrVecVec buckets1;
    buckets1.push_back(TTimeDoublePrVec(boost::begin(bucket11), boost::end(bucket11)));
    buckets1.push_back(TTimeDoublePrVec(boost::begin(bucket12), boost::end(bucket12)));
    buckets1.push_back(TTimeDoublePrVec(boost::begin(bucket13), boost::end(bucket13)));
    buckets1.push_back(TTimeDoublePrVec(boost::begin(bucket14), boost::end(bucket14)));
    buckets1.push_back(TTimeDoublePrVec(boost::begin(bucket15), boost::end(bucket15)));

    TTimeDoublePr bucket21[] = {TTimeDoublePr(1, 1.0),
                                TTimeDoublePr(5, 1.0),
                                TTimeDoublePr(15, 2.1),
                                TTimeDoublePr(25, 2.0),
                                TTimeDoublePr(180, 0.9),
                                TTimeDoublePr(190, 1.5),
                                TTimeDoublePr(400, 1.5),
                                TTimeDoublePr(550, 2.0)};
    TTimeDoublePr bucket22[] = {TTimeDoublePr(600, 2.0),
                                TTimeDoublePr(605, 2.0),
                                TTimeDoublePr(609, 2.0),
                                TTimeDoublePr(799, 2.2),
                                TTimeDoublePr(1199, 1.8)};
    TTimeDoublePr bucket23[] = {TTimeDoublePr(1200, 2.1),
                                TTimeDoublePr(1250, 2.5),
                                TTimeDoublePr(1255, 2.2),
                                TTimeDoublePr(1256, 2.4),
                                TTimeDoublePr(1300, 2.2),
                                TTimeDoublePr(1400, 2.5)};
    TTimeDoublePr bucket24[] = {TTimeDoublePr(1900, 3.5), TTimeDoublePr(1950, 3.5)};
    TTimeDoublePr bucket25[] = {TTimeDoublePr(2420, 3.5),
                                TTimeDoublePr(2480, 2.9),
                                TTimeDoublePr(2490, 3.9),
                                TTimeDoublePr(2500, 3.4),
                                TTimeDoublePr(2550, 4.1),
                                TTimeDoublePr(2600, 3.8)};
    TTimeDoublePrVecVec buckets2;
    buckets2.push_back(TTimeDoublePrVec(boost::begin(bucket21), boost::end(bucket21)));
    buckets2.push_back(TTimeDoublePrVec(boost::begin(bucket22), boost::end(bucket22)));
    buckets2.push_back(TTimeDoublePrVec(boost::begin(bucket23), boost::end(bucket23)));
    buckets2.push_back(TTimeDoublePrVec(boost::begin(bucket24), boost::end(bucket24)));
    buckets2.push_back(TTimeDoublePrVec(boost::begin(bucket25), boost::end(bucket25)));

    for (std::size_t i = 0u; i < 5; ++i) {
        LOG_DEBUG("Processing bucket " << i);
        gatherer.timeNow(startTime + i * bucketLength);

        const TTimeDoublePrVec& bucket1 = buckets1[i];
        for (std::size_t j = 0u; j < bucket1.size(); ++j) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[j].first, "p1", bucket1[j].second);
        }

        const TTimeDoublePrVec& bucket2 = buckets2[i];
        TMeanAccumulator a;
        for (std::size_t j = 0u; j < bucket2.size(); ++j) {
            addArrival(gatherer, m_ResourceMonitor, bucket2[j].first, "p2", bucket2[j].second);
            a.add(bucket2[j].second);
        }
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, gatherer.effectiveSampleCount(0), 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, gatherer.effectiveSampleCount(1), 1e-10);

    TSizeUInt64PrVec nonZeroCounts;
    gatherer.personNonZeroCounts(startTime + 4 * bucketLength, nonZeroCounts);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 3), (1, 6)]"),
                         core::CContainerPrinter::print(nonZeroCounts));

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    core_t::TTime featureBucketStart = core_t::TTime(startTime + 4 * bucketLength);
    gatherer.sampleNow(featureBucketStart);
    gatherer.featureData(featureBucketStart, bucketLength, featureData);

    CPPUNIT_ASSERT(!featureData.empty());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), featureData[0].second.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), featureData[1].second.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), featureData[2].second.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), featureData[3].second.size());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.5,
                                 featureData[0].second[0].second.s_BucketValue->value()[0],
                                 1e-10);
    CPPUNIT_ASSERT_EQUAL(3.2, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.8, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(10.5, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

    CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.5] 1 4)]"),
                         core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.2] 1 4)]"),
                         core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.8] 1 4)]"),
                         core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2400 [10.5] 1 3)]"),
                         core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.6,
                                 featureData[0].second[1].second.s_BucketValue->value()[0],
                                 1e-10);
    CPPUNIT_ASSERT_EQUAL(2.9, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(4.1, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(21.6, featureData[3].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[1].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[1].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[2].second[1].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[3].second[1].second.s_IsInteger);

    CPPUNIT_ASSERT_EQUAL(std::string("[(2290 [3.45] 1 6)]"),
                         core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2290 [2.9] 1 6)]"),
                         core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2290 [3.9] 1 6)]"),
                         core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2400 [21.6] 1 6)]"),
                         core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));

    // Test persistence. (We check for idempotency.)
    std::string origXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        gatherer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG("model XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CDataGatherer restoredGatherer(model_t::E_Metric,
                                   model_t::E_None,
                                   params,
                                   EMPTY_STRING,
                                   EMPTY_STRING,
                                   EMPTY_STRING,
                                   EMPTY_STRING,
                                   EMPTY_STRING,
                                   EMPTY_STRING,
                                   TStrVec(),
                                   false,
                                   KEY,
                                   traverser);

    // The XML representation of the new filter should be the
    // same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredGatherer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);

    // Remove person p1.
    TSizeVec peopleToRemove;
    peopleToRemove.push_back(0);
    gatherer.recyclePeople(peopleToRemove);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
    CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(0));
    CPPUNIT_ASSERT_EQUAL(std::string("p2"), gatherer.personName(1));
    std::size_t pid;
    CPPUNIT_ASSERT(gatherer.personId("p2", pid));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), pid);
    CPPUNIT_ASSERT(!gatherer.personId("p1", pid));

    CPPUNIT_ASSERT_EQUAL(std::size_t(0), gatherer.numberActiveAttributes());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), gatherer.numberOverFieldValues());

    gatherer.personNonZeroCounts(startTime + 4 * bucketLength, nonZeroCounts);
    CPPUNIT_ASSERT_EQUAL(std::string("[(1, 6)]"), core::CContainerPrinter::print(nonZeroCounts));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, gatherer.effectiveSampleCount(1), 1e-10);

    gatherer.featureData(core_t::TTime(startTime + 4 * bucketLength), bucketLength, featureData);

    CPPUNIT_ASSERT(!featureData.empty());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[1].second.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[2].second.size());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.6,
                                 featureData[0].second[0].second.s_BucketValue->value()[0],
                                 1e-10);
    CPPUNIT_ASSERT_EQUAL(2.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(4.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(21.6, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
    CPPUNIT_ASSERT_EQUAL(false, featureData[3].second[0].second.s_IsInteger);

    CPPUNIT_ASSERT_EQUAL(std::string("[(2290 [3.45] 1 6)]"),
                         core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2290 [2.9] 1 6)]"),
                         core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2290 [3.9] 1 6)]"),
                         core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(2400 [21.6] 1 6)]"),
                         core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
}

void CMetricDataGathererTest::testSampleCount(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::testSampleCount ***");

    // Test that we set sensible sample counts for each person.

    // Person 1 has constant update rate of 4 values per bucket.
    // Person 2 has variable rate with mean of 2 values per bucket.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    const std::size_t numberBuckets = 3u;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_Metric,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           TStrVec(),
                           false,
                           KEY,
                           features,
                           startTime,
                           0);

    std::size_t pid1 = addPerson("p1", gatherer, m_ResourceMonitor);
    std::size_t pid2 = addPerson("p2", gatherer, m_ResourceMonitor);

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < numberBuckets; ++i) {
        LOG_DEBUG("Processing bucket " << i);
        gatherer.timeNow(startTime + i * bucketLength);

        {
            LOG_DEBUG("count p1 = 6");
            addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 20, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 40, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 60, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 80, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 100, "p1", 1.0);
            addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 120, "p1", 1.0);
        }
        {
            TDoubleVec count;
            rng.generateUniformSamples(1.0, 5.0, 1, count);
            LOG_DEBUG("count p2 = " << ::floor(count[0]));
            for (std::size_t j = 0u; j < static_cast<std::size_t>(count[0]); ++j) {
                addArrival(gatherer,
                           m_ResourceMonitor,
                           startTime + i * bucketLength + 100 * (j + 1),
                           "p2",
                           1.0);
            }
        }
    }
    gatherer.timeNow(startTime + numberBuckets * bucketLength);

    LOG_DEBUG("p1 sample count = " << gatherer.effectiveSampleCount(pid1));
    LOG_DEBUG("p2 sample count = " << gatherer.effectiveSampleCount(pid2));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, gatherer.effectiveSampleCount(pid1), 1e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, gatherer.effectiveSampleCount(pid2), 1.0 + 1e-5);

    for (std::size_t i = numberBuckets; i < 100; ++i) {
        LOG_DEBUG("Processing bucket " << i);
        gatherer.timeNow(startTime + i * bucketLength);
        addArrival(gatherer, m_ResourceMonitor, startTime + i * bucketLength + 10, "p1", 1.0);
    }
    LOG_DEBUG("p1 sample count = " << gatherer.effectiveSampleCount(pid1));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, gatherer.effectiveSampleCount(pid1), 0.5);
}

void CMetricDataGathererTest::testRemovePeople(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::testRemovePeople ***");

    // Test various combinations of removed people.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 3600;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_Metric,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           TStrVec(),
                           false,
                           KEY,
                           features,
                           startTime,
                           0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p1", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson("p2", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson("p3", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), addPerson("p4", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), addPerson("p5", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), addPerson("p6", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(6), addPerson("p7", gatherer, m_ResourceMonitor));
    CPPUNIT_ASSERT_EQUAL(std::size_t(7), addPerson("p8", gatherer, m_ResourceMonitor));

    core_t::TTime times[][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {10, 20, 100, 0, 0, 0, 0, 0},
        {110, 120, 150, 170, 200, 0, 0, 0},
        {210, 220, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {400, 410, 480, 510, 530, 0, 0, 0},
        {1040, 1100, 1080, 1200, 1300, 1311, 2100, 0},
        {2200, 2500, 2600, 2610, 2702, 2731, 2710, 2862},
    };
    double values[][8] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.0, 5.0, 6.0, 1.0, 0.2, 0.0, 0.0, 0.0},
        {2.1, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {4.0, 1.0, 8.0, 1.0, 0.3, 0.0, 0.0, 0.0},
        {4.0, 1.0, 8.0, 1.0, 0.3, 1.1, 10.3, 0.0},
        {2.0, 5.0, 6.0, 1.0, 0.2, 3.1, 7.1, 6.2},
    };
    for (std::size_t i = 0u; i < boost::size(values); ++i) {
        for (std::size_t j = 0u; j < boost::size(values[i]); ++j) {
            if (values[i][j] > 0.0) {
                addArrival(gatherer,
                           m_ResourceMonitor,
                           startTime + times[i][j],
                           gatherer.personName(i),
                           values[i][j]);
            }
        }
    }

    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(0);
        peopleToRemove.push_back(1);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer(model_t::E_Metric,
                                       model_t::E_None,
                                       params,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       TStrVec(),
                                       false,
                                       KEY,
                                       features,
                                       startTime,
                                       0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p3", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson("p4", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson("p5", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), addPerson("p6", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), addPerson("p7", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), addPerson("p8", expectedGatherer, m_ResourceMonitor));

        std::size_t people[] = {2, 3, 4, 5, 6, 7};
        for (std::size_t i = 0u; i < boost::size(people); ++i) {
            for (std::size_t j = 0u; j < boost::size(values[people[i]]); ++j) {
                if (values[people[i]][j] > 0.0) {
                    addArrival(expectedGatherer,
                               m_ResourceMonitor,
                               startTime + times[people[i]][j],
                               expectedGatherer.personName(i),
                               values[people[i]][j]);
                }
            }
        }

        LOG_DEBUG("checksum          = " << gatherer.checksum());
        LOG_DEBUG("expected checksum = " << expectedGatherer.checksum());
        CPPUNIT_ASSERT_EQUAL(expectedGatherer.checksum(), gatherer.checksum());
    }
    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(3);
        peopleToRemove.push_back(4);
        peopleToRemove.push_back(7);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer(model_t::E_Metric,
                                       model_t::E_None,
                                       params,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       TStrVec(),
                                       false,
                                       KEY,
                                       features,
                                       startTime,
                                       0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p3", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson("p6", expectedGatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson("p7", expectedGatherer, m_ResourceMonitor));

        std::size_t people[] = {2, 5, 6};
        for (std::size_t i = 0u; i < boost::size(people); ++i) {
            for (std::size_t j = 0u; j < boost::size(values[people[i]]); ++j) {
                if (values[people[i]][j] > 0.0) {
                    addArrival(expectedGatherer,
                               m_ResourceMonitor,
                               startTime + times[people[i]][j],
                               expectedGatherer.personName(i),
                               values[people[i]][j]);
                }
            }
        }

        LOG_DEBUG("checksum          = " << gatherer.checksum());
        LOG_DEBUG("expected checksum = " << expectedGatherer.checksum());
        CPPUNIT_ASSERT_EQUAL(expectedGatherer.checksum(), gatherer.checksum());
    }
    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(2);
        peopleToRemove.push_back(5);
        peopleToRemove.push_back(6);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer(model_t::E_Metric,
                                       model_t::E_None,
                                       params,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       EMPTY_STRING,
                                       TStrVec(),
                                       false,
                                       KEY,
                                       features,
                                       startTime,
                                       0);

        LOG_DEBUG("checksum          = " << gatherer.checksum());
        LOG_DEBUG("expected checksum = " << expectedGatherer.checksum());
        CPPUNIT_ASSERT_EQUAL(expectedGatherer.checksum(), gatherer.checksum());
    }

    TSizeVec expectedRecycled;
    expectedRecycled.push_back(addPerson("p1", gatherer, m_ResourceMonitor));
    expectedRecycled.push_back(addPerson("p7", gatherer, m_ResourceMonitor));

    LOG_DEBUG(
        "recycled          = " << core::CContainerPrinter::print(gatherer.recycledPersonIds()));
    LOG_DEBUG("expected recycled = " << core::CContainerPrinter::print(expectedRecycled));
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedRecycled),
                         core::CContainerPrinter::print(gatherer.recycledPersonIds()));
}

void CMetricDataGathererTest::testSum(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::testSum ***");

    // Test sum and non-zero sum work as expected.

    const core_t::TTime bucketLength = 600;
    const std::size_t bucketCounts[] = {2, 5, 2, 1, 0, 0, 4, 8, 0, 1};
    const core_t::TTime startTime = 0;

    test::CRandomNumbers rng;

    TFeatureVec sumFeatures;
    sumFeatures.push_back(model_t::E_IndividualSumByBucketAndPerson);
    SModelParams params(bucketLength);
    CDataGatherer sum(model_t::E_Metric,
                      model_t::E_None,
                      params,
                      EMPTY_STRING,
                      EMPTY_STRING,
                      EMPTY_STRING,
                      EMPTY_STRING,
                      EMPTY_STRING,
                      EMPTY_STRING,
                      TStrVec(),
                      false,
                      KEY,
                      sumFeatures,
                      startTime,
                      0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p1", sum, m_ResourceMonitor));

    TFeatureVec nonZeroSumFeatures;
    nonZeroSumFeatures.push_back(model_t::E_IndividualNonNullSumByBucketAndPerson);

    CDataGatherer nonZeroSum(model_t::E_Metric,
                             model_t::E_None,
                             params,
                             EMPTY_STRING,
                             EMPTY_STRING,
                             EMPTY_STRING,
                             EMPTY_STRING,
                             EMPTY_STRING,
                             EMPTY_STRING,
                             TStrVec(),
                             false,
                             KEY,
                             nonZeroSumFeatures,
                             startTime,
                             0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p1", nonZeroSum, m_ResourceMonitor));

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        std::size_t count = bucketCounts[i];

        TDoubleVec times;
        rng.generateUniformSamples(0.0, static_cast<double>(bucketLength - 0.1), count, times);
        std::sort(times.begin(), times.end());

        TDoubleVec values;
        rng.generateNormalSamples(5.0, 4.0, count, values);

        double expected = 0.0;
        for (std::size_t j = 0u; j < times.size(); ++j) {
            addArrival(sum,
                       m_ResourceMonitor,
                       bucketStart + static_cast<core_t::TTime>(times[j]),
                       "p1",
                       values[j]);
            addArrival(nonZeroSum,
                       m_ResourceMonitor,
                       bucketStart + static_cast<core_t::TTime>(times[j]),
                       "p1",
                       values[j]);
            expected += doubleToStringToDouble(values[j]);
        }

        LOG_DEBUG("bucket: count = " << count << ", sum = " << expected);
        {
            TFeatureSizeFeatureDataPrVecPrVec data;
            sum.featureData(bucketStart, bucketLength, data);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), data.size());
            for (std::size_t j = 0u; j < data.size(); ++j) {
                const TSizeFeatureDataPrVec& featureData = data[j].second;
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(expected, featureData[j].second.s_BucketValue->value()[0]);
                CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                                     boost::unwrap_ref(featureData[j].second.s_Samples).size());
                CPPUNIT_ASSERT_EQUAL(expected,
                                     boost::unwrap_ref(featureData[j].second.s_Samples)[0]
                                         .value()[0]);
            }
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec data;
            nonZeroSum.featureData(bucketStart, bucketLength, data);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), data.size());
            for (std::size_t j = 0u; j < data.size(); ++j) {
                const TSizeFeatureDataPrVec& featureData = data[j].second;
                if (count == 0) {
                    CPPUNIT_ASSERT_EQUAL(std::size_t(0), featureData.size());
                } else {
                    CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                    CPPUNIT_ASSERT_EQUAL(expected, featureData[j].second.s_BucketValue->value()[0]);
                    CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                                         boost::unwrap_ref(featureData[j].second.s_Samples).size());
                    CPPUNIT_ASSERT_EQUAL(expected,
                                         boost::unwrap_ref(featureData[j].second.s_Samples)[0]
                                             .value()[0]);
                }
            }
        }

        bucketStart += bucketLength;
        sum.timeNow(bucketStart);
        nonZeroSum.timeNow(bucketStart);
    }
}

void CMetricDataGathererTest::singleSeriesOutOfOrderTests(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::singleSeriesOutOfOrderTests ***");

    // Test that the various statistics come back as we suspect.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 1;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TTimeDoublePr bucket1[] = {TTimeDoublePr(1, 1.0),
                               TTimeDoublePr(15, 2.1),
                               TTimeDoublePr(180, 0.9),
                               TTimeDoublePr(400, 1.5),
                               TTimeDoublePr(550, 2.0)};
    TTimeDoublePr bucket2[] = {TTimeDoublePr(600, 2.0),
                               TTimeDoublePr(190, 1.5),
                               TTimeDoublePr(799, 2.2),
                               TTimeDoublePr(1199, 1.8)};

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        features.push_back(model_t::E_IndividualSumByBucketAndPerson);
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               KEY,
                               features,
                               startTime,
                               2u);
        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

        CPPUNIT_ASSERT_EQUAL(std::size_t(4), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 4; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), pid);
        CPPUNIT_ASSERT(!gatherer.personId("a.n.other p", pid));

        {
            addArrival(gatherer, m_ResourceMonitor, bucket1[0].first, "p", bucket1[0].second);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[1].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[2].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
        }

        for (size_t i = 1; i < boost::size(bucket1); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket1[i].first, "p", bucket1[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(core_t::TTime(startTime + bucketLength - 1),
                                 bucketLength,
                                 featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(0.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(2.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(7.5, featureData[3].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(false, featureData[1].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(false, featureData[2].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(false, featureData[3].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(std::string("[]"),
                                 core::CContainerPrinter::print(
                                     featureData[0].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[]"),
                                 core::CContainerPrinter::print(
                                     featureData[1].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[]"),
                                 core::CContainerPrinter::print(
                                     featureData[2].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(0 [7.5] 1 5)]"),
                                 core::CContainerPrinter::print(
                                     featureData[3].second[0].second.s_Samples));

            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                gatherer.acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            LOG_DEBUG("gatherer XML representation:\n" << origXml);

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);

            CDataGatherer restoredGatherer(model_t::E_Metric,
                                           model_t::E_None,
                                           params,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           TStrVec(),
                                           false,
                                           KEY,
                                           traverser);

            // The XML representation of the new filter should be the
            // same as the original
            std::string newXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                restoredGatherer.acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (size_t i = 0; i < boost::size(bucket2); ++i) {
            addArrival(gatherer, m_ResourceMonitor, bucket2[i].first, "p", bucket2[i].second);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData(startTime, bucketLength, featureData);
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(0.9, featureData[1].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(2.1, featureData[2].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(9.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(true, featureData[3].second[0].second.s_IsInteger);
            CPPUNIT_ASSERT_EQUAL(std::string("[(8 [1.55] 1 2), (257 [1.3] 0.666667 3)]"),
                                 core::CContainerPrinter::print(
                                     featureData[0].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(8 [1] 1 2), (257 [0.9] 1 3)]"),
                                 core::CContainerPrinter::print(
                                     featureData[1].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(8 [2.1] 1 2), (257 [1.5] 1 3)]"),
                                 core::CContainerPrinter::print(
                                     featureData[2].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(std::string("[(0 [9] 1 6)]"),
                                 core::CContainerPrinter::print(
                                     featureData[3].second[0].second.s_Samples));

            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                gatherer.acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            LOG_DEBUG("model XML representation:\n" << origXml);

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);

            CDataGatherer restoredGatherer(model_t::E_Metric,
                                           model_t::E_None,
                                           params,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           TStrVec(),
                                           false,
                                           KEY,
                                           traverser);

            // The XML representation of the new filter should be the
            // same as the original
            std::string newXml;
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                restoredGatherer.acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }
    }
}

void CMetricDataGathererTest::testResetBucketGivenSingleSeries(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::testResetBucketGivenSingleSeries ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TTimeDoublePr data[] = {
        TTimeDoublePr(1, 1.0), // Bucket 1
        TTimeDoublePr(550, 2.0),
        TTimeDoublePr(600, 3.0), // Bucket 2
        TTimeDoublePr(700, 4.0),
        TTimeDoublePr(1000, 5.0),
        TTimeDoublePr(1200, 6.0) // Bucket 3
    };

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_Metric,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           TStrVec(),
                           false,
                           KEY,
                           features,
                           startTime,
                           2u);
    addPerson("p", gatherer, m_ResourceMonitor);

    for (std::size_t i = 0; i < boost::size(data); ++i) {
        addArrival(gatherer, m_ResourceMonitor, data[i].first, "p", data[i].second);
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    TSizeSizePr pidCidPr(0, 0);

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(4.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(12.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(3), gatherer.bucketCounts(600).find(pidCidPr)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr)->second);

    gatherer.resetBucket(600);
    addArrival(gatherer, m_ResourceMonitor, 610, "p", 2.0);
    addArrival(gatherer, m_ResourceMonitor, 620, "p", 3.0);

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(2.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr)->second);

    gatherer.sampleNow(0);
    gatherer.featureData(0, bucketLength, featureData);

    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(0 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));

    gatherer.sampleNow(600);
    gatherer.featureData(600, bucketLength, featureData);

    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(600 [5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
}

void CMetricDataGathererTest::testResetBucketGivenMultipleSeries(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::testResetBucketGivenMultipleSeries ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    TTimeDoublePr data[] = {
        TTimeDoublePr(1, 1.0), // Bucket 1
        TTimeDoublePr(550, 2.0),
        TTimeDoublePr(600, 3.0), // Bucket 2
        TTimeDoublePr(700, 4.0),
        TTimeDoublePr(1000, 5.0),
        TTimeDoublePr(1200, 6.0) // Bucket 3
    };

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_Metric,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           TStrVec(),
                           false,
                           KEY,
                           features,
                           startTime,
                           2u);
    addPerson("p1", gatherer, m_ResourceMonitor);
    addPerson("p2", gatherer, m_ResourceMonitor);
    addPerson("p3", gatherer, m_ResourceMonitor);

    for (std::size_t i = 0; i < boost::size(data); ++i) {
        for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
            addArrival(gatherer,
                       m_ResourceMonitor,
                       data[i].first,
                       gatherer.personName(pid),
                       data[i].second);
        }
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;
    TSizeSizePr pidCidPr0(0, 0);
    TSizeSizePr pidCidPr1(1, 0);
    TSizeSizePr pidCidPr2(2, 0);

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr0)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr1)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr2)->second);

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(4.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(4.0, featureData[0].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(4.0, featureData[0].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(12.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(12.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(12.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(3), gatherer.bucketCounts(600).find(pidCidPr0)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(3), gatherer.bucketCounts(600).find(pidCidPr1)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(3), gatherer.bucketCounts(600).find(pidCidPr2)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr0)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr1)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr2)->second);

    gatherer.resetBucket(600);
    for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
        addArrival(gatherer, m_ResourceMonitor, 610, gatherer.personName(pid), 2.0);
        addArrival(gatherer, m_ResourceMonitor, 620, gatherer.personName(pid), 3.0);
    }

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.5, featureData[0].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(1.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr0)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr1)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(0).find(pidCidPr2)->second);

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(2.5, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.5, featureData[0].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.5, featureData[0].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(2.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(3.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(5.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(600).find(pidCidPr0)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(600).find(pidCidPr1)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(2), gatherer.bucketCounts(600).find(pidCidPr2)->second);

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[0].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[1].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[0].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[1].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[2].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(6.0, featureData[3].second[2].second.s_BucketValue->value()[0]);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr0)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr1)->second);
    CPPUNIT_ASSERT_EQUAL(uint64_t(1), gatherer.bucketCounts(1200).find(pidCidPr2)->second);

    gatherer.sampleNow(0);
    gatherer.featureData(0, bucketLength, featureData);

    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[2].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [1] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[2].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(276 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[2].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(0 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(0 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(0 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[2].second.s_Samples));

    gatherer.sampleNow(600);
    gatherer.featureData(600, bucketLength, featureData);

    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2.5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[0].second[2].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [2] 1 2)]"),
                         core::CContainerPrinter::print(featureData[1].second[2].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(615 [3] 1 2)]"),
                         core::CContainerPrinter::print(featureData[2].second[2].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(600 [5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[0].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(600 [5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[1].second.s_Samples));
    CPPUNIT_ASSERT_EQUAL(std::string("[(600 [5] 1 2)]"),
                         core::CContainerPrinter::print(featureData[3].second[2].second.s_Samples));
}

void CMetricDataGathererTest::testInfluenceStatistics(void) {
    LOG_DEBUG("*** CMetricDataGathererTest::testInfluenceStatistics ***");

    typedef boost::tuple<core_t::TTime, double, std::string, std::string> TTimeDoubleStrStrTuple;
    typedef std::pair<double, double> TDoubleDoublePr;
    typedef std::pair<std::string, TDoubleDoublePr> TStrDoubleDoublePrPr;
    typedef std::vector<TStrDoubleDoublePrPr> TStrDoubleDoublePrPrVec;

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 2;
    params.s_SampleCountFactor = 1;
    params.s_SampleQueueGrowthFactor = 0.1;

    std::string influencerNames_[] = {"i1", "i2"};
    std::string influencerValues[][3] = {{"i11", "i12", "i13"}, {"i21", "i22", "i23"}};

    TTimeDoubleStrStrTuple data[] = {
        TTimeDoubleStrStrTuple(1, 1.0, influencerValues[0][0], influencerValues[1][0]), // Bucket 1
        TTimeDoubleStrStrTuple(150, 5.0, influencerValues[0][1], influencerValues[1][1]),
        TTimeDoubleStrStrTuple(150, 3.0, influencerValues[0][2], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(550, 2.0, influencerValues[0][0], influencerValues[1][0]),
        TTimeDoubleStrStrTuple(551, 2.1, influencerValues[0][1], influencerValues[1][1]),
        TTimeDoubleStrStrTuple(552, 4.0, influencerValues[0][2], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(554, 2.3, influencerValues[0][2], influencerValues[1][2]),
        TTimeDoubleStrStrTuple(600,
                               3.0,
                               influencerValues[0][1],
                               influencerValues[1][0]), // Bucket 2
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

    std::string expectedStatistics[] =
        {"[(i11, (1.5, 2)), (i12, (3.55, 2)), (i13, (3.1, 3)), (i21, (1.5, 2)), (i22, (3.55, 2)), "
         "(i23, (3.1, 3))]",
         "[(i11, (1.5, 2)), (i12, (3.55, 2)), (i13, (3.1, 3)), (i21, (1.5, 2)), (i22, (3.55, 2)), "
         "(i23, (3.1, 3))]",
         "[(i11, (1, 1)), (i12, (2.1, 1)), (i13, (2.3, 1)), (i21, (1, 1)), (i22, (2.1, 1)), (i23, "
         "(2.3, 1))]",
         "[(i11, (1, 1)), (i12, (2.1, 1)), (i13, (2.3, 1)), (i21, (1, 1)), (i22, (2.1, 1)), (i23, "
         "(2.3, 1))]",
         "[(i11, (2, 1)), (i12, (5, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (5, 1)), (i23, (4, "
         "1))]",
         "[(i11, (2, 1)), (i12, (5, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (5, 1)), (i23, (4, "
         "1))]",
         "[(i11, (3, 1)), (i12, (7.1, 1)), (i13, (9.3, 1)), (i21, (3, 1)), (i22, (7.1, 1)), (i23, "
         "(9.3, 1))]",
         "[(i11, (3, 1)), (i12, (7.1, 1)), (i13, (9.3, 1)), (i21, (3, 1)), (i22, (7.1, 1)), (i23, "
         "(9.3, 1))]",
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
    const std::string* expected = expectedStatistics;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    features.push_back(model_t::E_IndividualSumByBucketAndPerson);
    TStrVec influencerNames(boost::begin(influencerNames_), boost::end(influencerNames_));
    CDataGatherer gatherer(model_t::E_Metric,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           influencerNames,
                           false,
                           KEY,
                           features,
                           startTime,
                           2u);

    addPerson("p1", gatherer, m_ResourceMonitor, influencerNames.size());
    addPerson("p2", gatherer, m_ResourceMonitor, influencerNames.size());

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u, b = 0u; i < boost::size(data); ++i) {
        if (data[i].get<0>() >= bucketStart + bucketLength) {
            LOG_DEBUG("*** processing bucket ***");
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(bucketStart, bucketLength, featureData);
            for (std::size_t j = 0u; j < featureData.size(); ++j) {
                model_t::EFeature feature = featureData[j].first;
                LOG_DEBUG("feature = " << model_t::print(feature));

                const TSizeFeatureDataPrVec& data_ = featureData[j].second;
                for (std::size_t k = 0u; k < data_.size(); ++k) {
                    TStrDoubleDoublePrPrVec statistics;
                    for (std::size_t m = 0u; m < data_[k].second.s_InfluenceValues.size(); ++m) {
                        for (std::size_t n = 0u; n < data_[k].second.s_InfluenceValues[m].size();
                             ++n) {
                            statistics.push_back(
                                TStrDoubleDoublePrPr(data_[k].second.s_InfluenceValues[m][n].first,
                                                     TDoubleDoublePr(data_[k]
                                                                         .second
                                                                         .s_InfluenceValues[m][n]
                                                                         .second.first[0],
                                                                     data_[k]
                                                                         .second
                                                                         .s_InfluenceValues[m][n]
                                                                         .second.second)));
                        }
                    }
                    std::sort(statistics.begin(),
                              statistics.end(),
                              maths::COrderings::SFirstLess());

                    LOG_DEBUG("statistics = " << core::CContainerPrinter::print(statistics));
                    LOG_DEBUG("expected   = " << *expected);
                    CPPUNIT_ASSERT_EQUAL((*expected++), core::CContainerPrinter::print(statistics));
                }
            }

            bucketStart += bucketLength;
            ++b;
        }
        for (std::size_t pid = 0; pid < gatherer.numberActivePeople(); ++pid) {
            addArrival(gatherer,
                       m_ResourceMonitor,
                       data[i].get<0>(),
                       gatherer.personName(pid),
                       data[i].get<1>(),
                       data[i].get<2>(),
                       data[i].get<3>());
        }
    }
}

void CMetricDataGathererTest::testMultivariate(void) {
    typedef boost::tuple<core_t::TTime, double, double> TTimeDoubleDoubleTuple;
    typedef std::vector<TTimeDoubleDoubleTuple> TTimeDoubleDoubleTupleVec;
    typedef std::vector<TTimeDoubleDoubleTupleVec> TTimeDoubleDoubleTupleVecVec;

    static const std::string DELIMITER("__");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;

    SModelParams params(bucketLength);
    params.s_MultivariateComponentDelimiter = DELIMITER;

    TTimeDoubleDoubleTuple bucket1[] = {TTimeDoubleDoubleTuple(1, 1.0, 1.0),
                                        TTimeDoubleDoubleTuple(15, 2.1, 2.0),
                                        TTimeDoubleDoubleTuple(180, 0.9, 0.8),
                                        TTimeDoubleDoubleTuple(190, 1.5, 1.4),
                                        TTimeDoubleDoubleTuple(400, 1.5, 1.4),
                                        TTimeDoubleDoubleTuple(550, 2.0, 1.8)};
    TTimeDoubleDoubleTuple bucket2[] = {TTimeDoubleDoubleTuple(600, 2.0, 1.8),
                                        TTimeDoubleDoubleTuple(799, 2.2, 2.0),
                                        TTimeDoubleDoubleTuple(1199, 1.8, 1.6)};
    TTimeDoubleDoubleTuple bucket3[] = {TTimeDoubleDoubleTuple(1200, 2.1, 2.0),
                                        TTimeDoubleDoubleTuple(1250, 2.5, 2.4)};
    TTimeDoubleDoubleTuple bucket4[] = {
        TTimeDoubleDoubleTuple(1900, 3.5, 3.2),
    };
    TTimeDoubleDoubleTuple bucket5[] = {TTimeDoubleDoubleTuple(2420, 3.5, 3.2),
                                        TTimeDoubleDoubleTuple(2480, 3.2, 3.0),
                                        TTimeDoubleDoubleTuple(2490, 3.8, 3.8)};

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        TStrVec influencerNames;
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               influencerNames,
                               false,
                               KEY,
                               features,
                               startTime,
                               2u);
        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        CPPUNIT_ASSERT_EQUAL(features[0], gatherer.feature(0));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), pid);
        CPPUNIT_ASSERT(!gatherer.personId("a.n.other p", pid));

        {
            addArrival(gatherer,
                       m_ResourceMonitor,
                       bucket1[0].get<0>(),
                       "p",
                       bucket1[0].get<1>(),
                       bucket1[0].get<2>(),
                       DELIMITER);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[0].second[0].second.s_BucketValue->value()[0]);
            CPPUNIT_ASSERT_EQUAL(1.0, featureData[0].second[0].second.s_BucketValue->value()[1]);
            CPPUNIT_ASSERT_EQUAL(true, featureData[0].second[0].second.s_IsInteger);
        }

        for (size_t i = 1; i < boost::size(bucket1); ++i) {
            addArrival(gatherer,
                       m_ResourceMonitor,
                       bucket1[i].get<0>(),
                       "p",
                       bucket1[i].get<1>(),
                       bucket1[i].get<2>(),
                       DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime);
            gatherer.featureData(core_t::TTime(startTime + bucketLength - 1),
                                 bucketLength,
                                 featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5,
                                         featureData[0].second[0].second.s_BucketValue->value()[0],
                                         1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.4,
                                         featureData[0].second[0].second.s_BucketValue->value()[1],
                                         1e-10);
            CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
            LOG_DEBUG(core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));
            CPPUNIT_ASSERT_EQUAL(
                std::string("[(8 [1.55, 1.5] 1 2), (185 [1.2, 1.1] 1 2), (475 [1.75, 1.6] 1 2)]"),
                core::CContainerPrinter::print(featureData[0].second[0].second.s_Samples));

            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                gatherer.acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            LOG_DEBUG("gatherer XML representation:\n" << origXml);

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);

            CDataGatherer restoredGatherer(model_t::E_Metric,
                                           model_t::E_None,
                                           params,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           TStrVec(),
                                           false,
                                           KEY,
                                           traverser);

            // The XML representation of the new filter should be the
            // same as the original
            std::string newXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                restoredGatherer.acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }

        gatherer.timeNow(startTime + bucketLength);
        for (size_t i = 0; i < boost::size(bucket2); ++i) {
            addArrival(gatherer,
                       m_ResourceMonitor,
                       bucket2[i].get<0>(),
                       "p",
                       bucket2[i].get<1>(),
                       bucket2[i].get<2>(),
                       DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + bucketLength);
            gatherer.featureData(startTime + bucketLength, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0,
                                         featureData[0].second[0].second.s_BucketValue->value()[0],
                                         1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.8,
                                         featureData[0].second[0].second.s_BucketValue->value()[1],
                                         1e-10);
            CPPUNIT_ASSERT_EQUAL(std::string("[(700 [2.1, 1.9] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[0].second[0].second.s_Samples));

            // Test persistence. (We check for idempotency.)
            std::string origXml;
            {
                core::CRapidXmlStatePersistInserter inserter("root");
                gatherer.acceptPersistInserter(inserter);
                inserter.toXml(origXml);
            }

            LOG_DEBUG("model XML representation:\n" << origXml);

            // Restore the XML into a new filter
            core::CRapidXmlParser parser;
            CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);

            CDataGatherer restoredGatherer(model_t::E_Metric,
                                           model_t::E_None,
                                           params,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           EMPTY_STRING,
                                           TStrVec(),
                                           false,
                                           KEY,
                                           traverser);

            // The XML representation of the new filter should be the
            // same as the original
            std::string newXml;
            {
                ml::core::CRapidXmlStatePersistInserter inserter("root");
                restoredGatherer.acceptPersistInserter(inserter);
                inserter.toXml(newXml);
            }
            CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        }

        gatherer.timeNow(startTime + 2 * bucketLength);
        for (size_t i = 0; i < boost::size(bucket3); ++i) {
            addArrival(gatherer,
                       m_ResourceMonitor,
                       bucket3[i].get<0>(),
                       "p",
                       bucket3[i].get<1>(),
                       bucket3[i].get<2>(),
                       DELIMITER);
        }
        {
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.sampleNow(startTime + 2 * bucketLength);
            gatherer.featureData(startTime + 2 * bucketLength, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT(!featureData.empty());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.3,
                                         featureData[0].second[0].second.s_BucketValue->value()[0],
                                         1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.2,
                                         featureData[0].second[0].second.s_BucketValue->value()[1],
                                         1e-10);
            CPPUNIT_ASSERT_EQUAL(std::string("[(1200 [1.95, 1.8] 1 2)]"),
                                 core::CContainerPrinter::print(
                                     featureData[0].second[0].second.s_Samples));
        }
    }

    // Test capture of sample measurement count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanLatLongByPerson);
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               KEY,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer, m_ResourceMonitor));

        TTimeDoubleDoubleTupleVecVec buckets;
        buckets.push_back(TTimeDoubleDoubleTupleVec(boost::begin(bucket1), boost::end(bucket1)));
        buckets.push_back(TTimeDoubleDoubleTupleVec(boost::begin(bucket2), boost::end(bucket2)));
        buckets.push_back(TTimeDoubleDoubleTupleVec(boost::begin(bucket3), boost::end(bucket3)));
        buckets.push_back(TTimeDoubleDoubleTupleVec(boost::begin(bucket4), boost::end(bucket4)));
        buckets.push_back(TTimeDoubleDoubleTupleVec(boost::begin(bucket5), boost::end(bucket5)));

        for (std::size_t i = 0u; i < buckets.size(); ++i) {
            LOG_DEBUG("Processing bucket " << i);
            gatherer.timeNow(startTime + i * bucketLength);
            const TTimeDoubleDoubleTupleVec& bucket = buckets[i];
            for (std::size_t j = 0u; j < bucket.size(); ++j) {
                addArrival(gatherer,
                           m_ResourceMonitor,
                           bucket[j].get<0>(),
                           "p",
                           bucket[j].get<1>(),
                           bucket[j].get<2>(),
                           DELIMITER);
            }
        }

        CPPUNIT_ASSERT_EQUAL(4.0, gatherer.effectiveSampleCount(0));
        TFeatureSizeFeatureDataPrVecPrVec featureData;
        core_t::TTime featureBucketStart = core_t::TTime(startTime + 4 * bucketLength);
        gatherer.sampleNow(featureBucketStart);
        gatherer.featureData(featureBucketStart, bucketLength, featureData);
        CPPUNIT_ASSERT(!featureData.empty());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.5,
                                     featureData[0].second[0].second.s_BucketValue->value()[0],
                                     1e-10);
        CPPUNIT_ASSERT_EQUAL(false, featureData[0].second[0].second.s_IsInteger);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::string("[(2323 [3.5, 3.3] 1 4)]"),
                             core::CContainerPrinter::print(
                                 featureData[0].second[0].second.s_Samples));
    }
}

void CMetricDataGathererTest::testStatisticsPersist(void) {
    CGathererTools::TMeanGatherer::TMetricPartialStatistic stat(1);
    stat.add(TDoubleVec(1, 44.4), 1299196740, 1);
    stat.add(TDoubleVec(1, 5.5), 1299196741, 1);
    stat.add(TDoubleVec(1, 0.6), 1299196742, 1);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        stat.persist(inserter);
        inserter.toXml(origXml);
    }

    core_t::TTime origTime = stat.time();
    std::string restoredXml;
    std::string restoredPrint;
    core_t::TTime restoredTime;
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        CGathererTools::TMeanGatherer::TMetricPartialStatistic restored(1);
        traverser.traverseSubLevel(
            boost::bind(&CGathererTools::TMeanGatherer::TMetricPartialStatistic::restore,
                        boost::ref(restored),
                        _1));

        restoredTime = restored.time();
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restored.persist(inserter);
            inserter.toXml(restoredXml);
        }
    }
    CPPUNIT_ASSERT_EQUAL(origXml, restoredXml);
    CPPUNIT_ASSERT_EQUAL(origTime, restoredTime);
}

void CMetricDataGathererTest::testVarp(void) {
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
        TFeatureVec features;
        features.push_back(model_t::E_IndividualVarianceByPerson);
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               KEY,
                               features,
                               startTime,
                               2u);
        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(person, gatherer, m_ResourceMonitor));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());

        {
            values.push_back(5.0);
            values.push_back(6.0);
            values.push_back(3.0);
            values.push_back(2.0);
            values.push_back(4.0);
            addArrivals(gatherer, m_ResourceMonitor, startTime, 10, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            // Expect only 1 feature
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            TFeatureSizeFeatureDataPrVecPr fsfd = featureData[0];
            CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualVarianceByPerson, fsfd.first);
            CSample::TDouble1Vec v = featureData[0].second[0].second.s_BucketValue->value();
            double expectedMean = 0;
            double expectedVariance = ::variance(values, expectedMean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v[0], expectedVariance, 0.0001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v[1], expectedMean, 0.0001);
        }
        startTime += bucketLength;
        {
            values.clear();
            values.push_back(115.0);
            values.push_back(116.0);
            values.push_back(117.0);
            values.push_back(1111.5);
            values.push_back(22.45);
            values.push_back(2526.55634);
            values.push_back(55.55);
            values.push_back(14.723);
            addArrivals(gatherer, m_ResourceMonitor, startTime, 100, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CSample::TDouble1Vec v = featureData[0].second[0].second.s_BucketValue->value();
            double expectedMean = 0;
            double expectedVariance = ::variance(values, expectedMean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v[0], expectedVariance, 0.0001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v[1], expectedMean, 0.0001);
        }
        startTime += bucketLength;
        gatherer.sampleNow(startTime);
        startTime += bucketLength;
        {
            values.clear();
            values.push_back(0.0);
            addArrivals(gatherer, m_ResourceMonitor, startTime, 100, person, values);
            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT(!featureData[0].second[0].second.s_BucketValue);
        }
    }

    // Now test with influencers
    LOG_DEBUG("Testing influencers");
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualVarianceByPerson);
        TStrVec influencerFieldNames;
        influencerFieldNames.push_back("i");
        influencerFieldNames.push_back("j");
        CDataGatherer gatherer(model_t::E_Metric,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               influencerFieldNames,
                               false,
                               KEY,
                               features,
                               startTime,
                               2u);
        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0),
                             addPerson(person,
                                       gatherer,
                                       m_ResourceMonitor,
                                       influencerFieldNames.size()));

        TStrVec testInf(gatherer.beginInfluencers(), gatherer.endInfluencers());

        LOG_DEBUG("Influencer fields: " << core::CContainerPrinter::print(testInf));
        LOG_DEBUG("FOI: " << core::CContainerPrinter::print(gatherer.fieldsOfInterest()));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
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
            addArrival(gatherer, m_ResourceMonitor, startTime + 800, person, 12.12, inf1, inf2);
            addArrival(gatherer, m_ResourceMonitor, startTime + 900, person, 5.2, inf1, "");
            addArrival(gatherer, m_ResourceMonitor, startTime + 950, person, 5.0, inf1, inf3);

            gatherer.sampleNow(startTime);
            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(startTime, bucketLength, featureData);
            TFeatureSizeFeatureDataPrVecPr fsfd = featureData[0];
            CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualVarianceByPerson, fsfd.first);

            CSample::TDouble1Vec v = featureData[0].second[0].second.s_BucketValue->value();
            values.clear();
            values.push_back(5.0);
            values.push_back(5.5);
            values.push_back(5.9);
            values.push_back(5.2);
            values.push_back(5.1);
            values.push_back(2.2);
            values.push_back(4.9);
            values.push_back(5.1);
            values.push_back(5.0);
            values.push_back(12.12);
            values.push_back(5.2);
            values.push_back(5.0);
            values.push_back(1.0);
            double expectedMean = 0;
            double expectedVariance = ::variance(values, expectedMean);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v[0], expectedVariance, 0.0001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v[1], expectedMean, 0.0001);

            values.pop_back();
            double i1ExpectedMean = 0;
            double i1ExpectedVariance = ::variance(values, i1ExpectedMean);

            values.clear();
            values.push_back(5.0);
            values.push_back(2.2);
            values.push_back(12.12);
            double i2ExpectedMean = 0;
            double i2ExpectedVariance = ::variance(values, i2ExpectedMean);

            values.clear();
            values.push_back(5.0);
            double i3ExpectedMean = 0;
            double i3ExpectedVariance = ::variance(values, i3ExpectedMean);

            SMetricFeatureData mfd = fsfd.second[0].second;
            SMetricFeatureData::TStrCRefDouble1VecDoublePrPrVecVec ivs = mfd.s_InfluenceValues;
            LOG_DEBUG("IVs: " << core::CContainerPrinter::print(ivs));
            CPPUNIT_ASSERT_EQUAL(std::size_t(2), ivs.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), ivs[0].size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(2), ivs[1].size());

            const SMetricFeatureData::TStrCRefDouble1VecDoublePrPr& ivs1 = ivs[0][0];
            CPPUNIT_ASSERT_EQUAL(inf1, ivs1.first.get());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, ivs1.second.second, 0.0001);
            CPPUNIT_ASSERT_EQUAL(std::size_t(2), ivs1.second.first.size());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ivs1.second.first[0], i1ExpectedVariance, 0.0001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ivs1.second.first[1], i1ExpectedMean, 0.0001);

            // The order of ivs2 and ivs3 seems to be backwards...
            const SMetricFeatureData::TStrCRefDouble1VecDoublePrPr& ivs2 = ivs[1][1];
            CPPUNIT_ASSERT_EQUAL(inf2, ivs2.first.get());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, ivs2.second.second, 0.0001);
            CPPUNIT_ASSERT_EQUAL(std::size_t(2), ivs2.second.first.size());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ivs2.second.first[0], i2ExpectedVariance, 0.0001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ivs2.second.first[1], i2ExpectedMean, 0.0001);

            const SMetricFeatureData::TStrCRefDouble1VecDoublePrPr& ivs3 = ivs[1][0];
            CPPUNIT_ASSERT_EQUAL(inf3, ivs3.first.get());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ivs3.second.second, 0.0001);
            CPPUNIT_ASSERT_EQUAL(std::size_t(2), ivs3.second.first.size());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ivs3.second.first[0], i3ExpectedVariance, 0.0001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(ivs3.second.first[1], i3ExpectedMean, 0.0001);
        }
    }
}

CppUnit::Test* CMetricDataGathererTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CMetricDataGathererTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<
                          CMetricDataGathererTest>("CMetricDataGathererTest::singleSeriesTests",
                                                   &CMetricDataGathererTest::singleSeriesTests));
    suiteOfTests->addTest(new CppUnit::TestCaller<
                          CMetricDataGathererTest>("CMetricDataGathererTest::multipleSeriesTests",
                                                   &CMetricDataGathererTest::multipleSeriesTests));
    suiteOfTests->addTest(new CppUnit::TestCaller<
                          CMetricDataGathererTest>("CMetricDataGathererTest::testSampleCount",
                                                   &CMetricDataGathererTest::testSampleCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<
                          CMetricDataGathererTest>("CMetricDataGathererTest::testRemovePeople",
                                                   &CMetricDataGathererTest::testRemovePeople));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMetricDataGathererTest>("CMetricDataGathererTest::testSum",
                                                         &CMetricDataGathererTest::testSum));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CMetricDataGathererTest>("CMetricDataGathererTest::singleSeriesOutOfOrderTests",
                                     &CMetricDataGathererTest::singleSeriesOutOfOrderTests));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CMetricDataGathererTest>("CMetricDataGathererTest::testResetBucketGivenSingleSeries",
                                     &CMetricDataGathererTest::testResetBucketGivenSingleSeries));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CMetricDataGathererTest>("CMetricDataGathererTest::testResetBucketGivenMultipleSeries",
                                     &CMetricDataGathererTest::testResetBucketGivenMultipleSeries));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CMetricDataGathererTest>("CMetricDataGathererTest::testInfluenceStatistics",
                                     &CMetricDataGathererTest::testInfluenceStatistics));
    suiteOfTests->addTest(new CppUnit::TestCaller<
                          CMetricDataGathererTest>("CMetricDataGathererTest::testMultivariate",
                                                   &CMetricDataGathererTest::testMultivariate));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<
            CMetricDataGathererTest>("CMetricDataGathererTest::testStatisticsPersist",
                                     &CMetricDataGathererTest::testStatisticsPersist));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CMetricDataGathererTest>("CMetricDataGathererTest::testVarp",
                                                         &CMetricDataGathererTest::testVarp));
    return suiteOfTests;
}
