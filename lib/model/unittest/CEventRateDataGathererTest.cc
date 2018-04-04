/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CEventRateDataGathererTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CRegex.h>

#include <maths/COrderings.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CEventRateBucketGatherer.h>
#include <model/CModelParams.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>
#include <model/CStringStore.h>
#include <model/ModelTypes.h>

#include <boost/range.hpp>

#include <utility>
#include <vector>

using namespace ml;
using namespace model;

using TSizeVec = std::vector<std::size_t>;
using TFeatureVec = std::vector<model_t::EFeature>;
using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
using TStrVec = std::vector<std::string>;
using TStrVecCItr = TStrVec::const_iterator;
using TStrVecVec = std::vector<TStrVec>;
using TFeatureData = SEventRateFeatureData;
using TSizeFeatureDataPr = std::pair<std::size_t, TFeatureData>;
using TSizeFeatureDataPrVec = std::vector<TSizeFeatureDataPr>;
using TFeatureSizeFeatureDataPrVecPr = std::pair<model_t::EFeature, TSizeFeatureDataPrVec>;
using TFeatureSizeFeatureDataPrVecPrVec = std::vector<TFeatureSizeFeatureDataPrVecPr>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, TFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TFeatureSizeSizePrFeatureDataPrVecPr = std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;
using TSizeSizePrStoredStringPtrPr = CBucketGatherer::TSizeSizePrStoredStringPtrPr;
using TSizeSizePrStoredStringPtrPrUInt64UMapVec = CBucketGatherer::TSizeSizePrStoredStringPtrPrUInt64UMapVec;
using TTimeVec = std::vector<core_t::TTime>;
using TStrCPtrVec = CBucketGatherer::TStrCPtrVec;

namespace {

const CSearchKey key;
const std::string EMPTY_STRING("");

std::size_t addPerson(CDataGatherer& gatherer,
                      CResourceMonitor& resourceMonitor,
                      const std::string& p,
                      const std::string& v = EMPTY_STRING,
                      std::size_t numInfluencers = 0) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    std::string i("i");
    for (std::size_t j = 0; j < numInfluencers; ++j) {
        person.push_back(&i);
    }
    if (!v.empty()) {
        person.push_back(&v);
    }
    CEventData result;
    gatherer.processFields(person, result, resourceMonitor);
    return *result.personId();
}

void addArrival(CDataGatherer& gatherer, CResourceMonitor& resourceMonitor, core_t::TTime time, const std::string& person) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                const std::string& attribute) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    fieldValues.push_back(&attribute);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                const std::string& value,
                const std::string& influencer) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    fieldValues.push_back(&influencer);
    fieldValues.push_back(&value);

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor,
                core_t::TTime time,
                const std::string& person,
                const TStrVec& influencers,
                const std::string& value) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);

    for (std::size_t i = 0; i < influencers.size(); ++i) {
        fieldValues.push_back(&influencers[i]);
    }

    if (!value.empty()) {
        fieldValues.push_back(&value);
    }

    CEventData eventData;
    eventData.time(time);

    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void testPersistence(const SModelParams& params, const CDataGatherer& gatherer) {
    // Test persistence. (We check for idempotency.)
    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        gatherer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE("model XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CDataGatherer restoredGatherer(model_t::E_EventRate,
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
                                   key,
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
    CPPUNIT_ASSERT_EQUAL(gatherer.checksum(), restoredGatherer.checksum());
}

void testInfluencerPerFeature(model_t::EFeature feature,
                              const TTimeVec& data,
                              const TStrVecVec& influencers,
                              const TStrVec& expected,
                              const std::string& valueField,
                              CResourceMonitor& resourceMonitor) {
    LOG_DEBUG(" *** testing " << model_t::print(feature) << " ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);

    TFeatureVec features;
    features.push_back(feature);
    TStrVec influencerFieldNames;
    influencerFieldNames.push_back("IF1");
    CDataGatherer gatherer(model_t::E_EventRate,
                           model_t::E_None,
                           params,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           EMPTY_STRING,
                           valueField,
                           influencerFieldNames,
                           false,
                           key,
                           features,
                           startTime,
                           0);
    CPPUNIT_ASSERT(!gatherer.isPopulation());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, resourceMonitor, "p", valueField, 1));

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
    for (std::size_t i = 0u; i < features.size(); ++i) {
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

    CPPUNIT_ASSERT_EQUAL(std::size_t(0), gatherer.numberActiveAttributes());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), gatherer.numberOverFieldValues());

    CPPUNIT_ASSERT_EQUAL(startTime, gatherer.currentBucketStartTime());
    gatherer.currentBucketStartTime(200);
    CPPUNIT_ASSERT_EQUAL(static_cast<core_t::TTime>(200), gatherer.currentBucketStartTime());
    gatherer.currentBucketStartTime(startTime);

    CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());

    core_t::TTime time = startTime;
    for (std::size_t i = 0, j = 0u; i < data.size(); ++i) {
        for (/**/; j < 5 && data[i] >= time + bucketLength; time += bucketLength, ++j, gatherer.timeNow(time)) {
            LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());

            CPPUNIT_ASSERT_EQUAL(feature, featureData[0].first);
            CPPUNIT_ASSERT_EQUAL(expected[j], core::CContainerPrinter::print(featureData[0].second));

            testPersistence(params, gatherer);
        }

        if (j < 5) {
            addArrival(gatherer, resourceMonitor, data[i], "p", influencers[i], valueField.empty() ? EMPTY_STRING : "value");
        }
    }
}

void importCsvData(CDataGatherer& gatherer, CResourceMonitor& resourceMonitor, const std::string& filename, const TSizeVec& fields) {
    using TifstreamPtr = boost::shared_ptr<std::ifstream>;
    TifstreamPtr ifs(new std::ifstream(filename.c_str()));
    CPPUNIT_ASSERT(ifs->is_open());

    core::CRegex regex;
    CPPUNIT_ASSERT(regex.init(","));

    std::string line;
    // read the header
    CPPUNIT_ASSERT(std::getline(*ifs, line));

    while (std::getline(*ifs, line)) {
        LOG_TRACE("Got string: " << line);
        core::CRegex::TStrVec tokens;
        regex.split(line, tokens);

        core_t::TTime time;
        CPPUNIT_ASSERT(core::CStringUtils::stringToType(tokens[0], time));

        CDataGatherer::TStrCPtrVec fieldValues;
        CEventData data;
        data.time(time);
        for (std::size_t i = 0; i < fields.size(); i++) {
            fieldValues.push_back(&tokens[fields[i]]);
        }
        gatherer.addArrival(fieldValues, data, resourceMonitor);
    }
    ifs.reset();
}

} // namespace

void CEventRateDataGathererTest::testLatencyPersist() {
    LOG_DEBUG("*** testLatencyPersist ***");

    core_t::TTime bucketLength = 3600;
    core_t::TTime latency = 5 * bucketLength;
    core_t::TTime startTime = 1420192800;
    SModelParams params(bucketLength);
    params.configureLatency(latency, bucketLength);

    {
        // Create a gatherer, no influences
        TFeatureVec features;
        features.push_back(model_t::E_IndividualUniqueCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "program",
                               EMPTY_STRING,
                               "file",
                               TStrVec(),
                               false,
                               key,
                               features,
                               startTime,
                               0);
        TSizeVec fields;
        fields.push_back(2);
        fields.push_back(1);

        importCsvData(gatherer, m_ResourceMonitor, "testfiles/files_users_programs.csv", fields);

        testPersistence(params, gatherer);
    }
    {
        // Create a gatherer, with influences
        TFeatureVec features;
        TStrVec influencers;
        influencers.push_back("user");
        features.push_back(model_t::E_IndividualUniqueCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "program",
                               EMPTY_STRING,
                               "file",
                               influencers,
                               false,
                               key,
                               features,
                               startTime,
                               0);
        TSizeVec fields;
        fields.push_back(2);
        fields.push_back(3);
        fields.push_back(1);

        importCsvData(gatherer, m_ResourceMonitor, "testfiles/files_users_programs.csv", fields);

        testPersistence(params, gatherer);
    }
    {
        // Create a gatherer, no influences
        TFeatureVec features;
        features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "program",
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               key,
                               features,
                               startTime,
                               0);
        TSizeVec fields;
        fields.push_back(2);

        importCsvData(gatherer, m_ResourceMonitor, "testfiles/files_users_programs.csv", fields);

        testPersistence(params, gatherer);
    }
    {
        // Create a gatherer, with influences
        TFeatureVec features;
        TStrVec influencers;
        influencers.push_back("user");
        features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "program",
                               EMPTY_STRING,
                               EMPTY_STRING,
                               influencers,
                               false,
                               key,
                               features,
                               startTime,
                               0);
        TSizeVec fields;
        fields.push_back(2);
        fields.push_back(3);

        importCsvData(gatherer, m_ResourceMonitor, "testfiles/files_users_programs.csv", fields);

        testPersistence(params, gatherer);
    }
}

void CEventRateDataGathererTest::singleSeriesTests() {
    LOG_DEBUG("*** singleSeriesTests ***");

    // Test that the various statistics come back as we expect.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);

    core_t::TTime data[] = {
        1,
        15,
        180,
        190,
        400,
        550, // bucket 1
        600,
        799,
        1199, // bucket 2
        1200,
        1250, // bucket 3
              // bucket 4
        2420,
        2480,
        2490, // bucket 5
        10000 // sentinel
    };

    std::string expectedPersonCounts[] = {
        std::string("[(0, 6)]"), std::string("[(0, 3)]"), std::string("[(0, 2)]"), std::string("[(0, 0)]"), std::string("[(0, 3)]")};

    std::string expectedPersonNonZeroCounts[] = {
        std::string("[(0, 6)]"), std::string("[(0, 3)]"), std::string("[(0, 2)]"), std::string("[]"), std::string("[(0, 3)]")};

    std::string expectedPersonIndicator[] = {
        std::string("[(0, 1)]"), std::string("[(0, 1)]"), std::string("[(0, 1)]"), std::string("[]"), std::string("[(0, 1)]")};

    // Test the count by bucket and person and bad feature
    // (which should be ignored).
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p"));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 1; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }
        CPPUNIT_ASSERT(gatherer.hasFeature(model_t::E_IndividualCountByBucketAndPerson));
        CPPUNIT_ASSERT(!gatherer.hasFeature(model_t::E_IndividualMinByPerson));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), pid);
        CPPUNIT_ASSERT(!gatherer.personId("a.n.other p", pid));

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), gatherer.numberActiveAttributes());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), gatherer.numberOverFieldValues());

        CPPUNIT_ASSERT_EQUAL(startTime, gatherer.currentBucketStartTime());
        gatherer.currentBucketStartTime(200);
        CPPUNIT_ASSERT_EQUAL(static_cast<core_t::TTime>(200), gatherer.currentBucketStartTime());
        gatherer.currentBucketStartTime(startTime);

        CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());

        core_t::TTime time = startTime;
        for (std::size_t i = 0, j = 0u; i < boost::size(data); ++i) {
            for (/**/; j < 5 && data[i] >= time + bucketLength; time += bucketLength, ++j, gatherer.timeNow(time)) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonCounts[j], core::CContainerPrinter::print(featureData[0].second));

                testPersistence(params, gatherer);
            }

            if (j < 5) {
                addArrival(gatherer, m_ResourceMonitor, data[i], "p");
            }
        }
    }

    // Test non-zero count and person bucket count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
        features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p"));

        core_t::TTime time = startTime;
        for (std::size_t i = 0, j = 0u; i < boost::size(data); ++i) {
            for (/**/; j < 5 && data[i] >= time + bucketLength; time += bucketLength, ++j, gatherer.timeNow(time)) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(2), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualNonZeroCountByBucketAndPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonNonZeroCounts[j], core::CContainerPrinter::print(featureData[0].second));
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualTotalBucketCountByPerson, featureData[1].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonNonZeroCounts[j], core::CContainerPrinter::print(featureData[1].second));

                testPersistence(params, gatherer);
            }

            if (j < 5) {
                addArrival(gatherer, m_ResourceMonitor, data[i], "p");
            }
        }
    }

    // Test test person indicator by bucket.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualIndicatorOfBucketPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p"));

        core_t::TTime time = startTime;
        for (std::size_t i = 0, j = 0u; i < boost::size(data); ++i) {
            for (/**/; j < 5 && data[i] >= time + bucketLength; time += bucketLength, ++j, gatherer.timeNow(time)) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonIndicator[j], core::CContainerPrinter::print(featureData[0].second));

                testPersistence(params, gatherer);
            }

            if (j < 5) {
                addArrival(gatherer, m_ResourceMonitor, data[i], "p");
            }
        }
    }
}

void CEventRateDataGathererTest::multipleSeriesTests() {
    LOG_DEBUG("*** multipleSeriesTests ***");

    // Test that the various statistics come back as we expect
    // for multiple people.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;

    core_t::TTime data1[] = {
        1,
        15,
        180,
        190,
        400,
        550, // bucket 1
        600,
        799,
        1199, // bucket 2
        1200,
        1250, // bucket 3
        1900, // bucket 4
        2420,
        2480,
        2490, // bucket 5
        10000 // sentinel
    };
    core_t::TTime data2[] = {
        1,    5,    15,   25,   180,  190,  400, 550, // bucket 1
        600,  605,  609,  799,  1199,                 // bucket 2
        1200, 1250, 1255, 1256, 1300, 1400,           // bucket 3
        1900, 1950,                                   // bucket 4
        2420, 2480, 2490, 2500, 2550, 2600,           // bucket 5
        10000                                         // sentinel
    };

    std::string expectedPersonCounts[] = {std::string("[(0, 6), (1, 8)]"),
                                          std::string("[(0, 3), (1, 5)]"),
                                          std::string("[(0, 2), (1, 6)]"),
                                          std::string("[(0, 1), (1, 2)]"),
                                          std::string("[(0, 3), (1, 6)]")};

    SModelParams params(bucketLength);

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p1"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(gatherer, m_ResourceMonitor, "p2"));

        core_t::TTime time = startTime;
        std::size_t i1 = 0u, i2 = 0u, j = 0u;
        for (;;) {
            for (/**/; j < 5 && std::min(data1[i1], data2[i2]) >= time + bucketLength; time += bucketLength, ++j) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonCounts[j], core::CContainerPrinter::print(featureData[0].second));

                testPersistence(params, gatherer);
            }

            if (j >= 5) {
                break;
            }

            if (data1[i1] < data2[i2]) {
                LOG_DEBUG("Adding arrival for p1 at " << data1[i1]);
                addArrival(gatherer, m_ResourceMonitor, data1[i1], "p1");
                ++i1;
            } else {
                LOG_DEBUG("Adding arrival for p2 at " << data2[i2]);
                addArrival(gatherer, m_ResourceMonitor, data2[i2], "p2");
                ++i2;
            }
        }

        TSizeVec peopleToRemove;
        peopleToRemove.push_back(1);
        gatherer.recyclePeople(peopleToRemove);

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p1"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p1", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), pid);
        CPPUNIT_ASSERT(!gatherer.personId("p2", pid));

        TFeatureSizeFeatureDataPrVecPrVec featureData;
        gatherer.featureData(startTime + 4 * bucketLength, bucketLength, featureData);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 3)]"), core::CContainerPrinter::print(featureData[0].second));
    }

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p1"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(gatherer, m_ResourceMonitor, "p2"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson(gatherer, m_ResourceMonitor, "p3"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), addPerson(gatherer, m_ResourceMonitor, "p4"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), addPerson(gatherer, m_ResourceMonitor, "p5"));

        for (std::size_t i = 0u; i < 5; ++i) {
            addArrival(gatherer, m_ResourceMonitor, startTime, gatherer.personName(i));
        }
        addArrival(gatherer, m_ResourceMonitor, startTime + 1, gatherer.personName(2));
        addArrival(gatherer, m_ResourceMonitor, startTime + 2, gatherer.personName(4));
        addArrival(gatherer, m_ResourceMonitor, startTime + 3, gatherer.personName(4));

        TSizeUInt64PrVec personCounts;

        TFeatureSizeFeatureDataPrVecPrVec featureData;
        gatherer.featureData(startTime, bucketLength, featureData);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (1, 1), (2, 2), (3, 1), (4, 3)]"),
                             core::CContainerPrinter::print(featureData[0].second));

        TSizeVec peopleToRemove;
        peopleToRemove.push_back(0);
        peopleToRemove.push_back(1);
        peopleToRemove.push_back(3);
        gatherer.recyclePeople(peopleToRemove);

        CPPUNIT_ASSERT_EQUAL(std::size_t(2), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p3"), gatherer.personName(2));
        CPPUNIT_ASSERT_EQUAL(std::string("p5"), gatherer.personName(4));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(3));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p3", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), pid);
        CPPUNIT_ASSERT(gatherer.personId("p5", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), pid);

        gatherer.featureData(startTime, bucketLength, featureData);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("[(2, 2), (4, 3)]"), core::CContainerPrinter::print(featureData[0].second));
    }
}

void CEventRateDataGathererTest::testRemovePeople() {
    LOG_DEBUG("*** testRemovePeople ***");

    // Test various combinations of removed people.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
    features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
    features.push_back(model_t::E_IndividualIndicatorOfBucketPerson);
    features.push_back(model_t::E_IndividualLowCountsByBucketAndPerson);
    features.push_back(model_t::E_IndividualHighCountsByBucketAndPerson);
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_EventRate,
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
                           key,
                           features,
                           startTime,
                           0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p1"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(gatherer, m_ResourceMonitor, "p2"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson(gatherer, m_ResourceMonitor, "p3"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), addPerson(gatherer, m_ResourceMonitor, "p4"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(4), addPerson(gatherer, m_ResourceMonitor, "p5"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), addPerson(gatherer, m_ResourceMonitor, "p6"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(6), addPerson(gatherer, m_ResourceMonitor, "p7"));
    CPPUNIT_ASSERT_EQUAL(std::size_t(7), addPerson(gatherer, m_ResourceMonitor, "p8"));

    core_t::TTime counts[] = {0, 3, 5, 2, 0, 5, 7, 10};
    for (std::size_t i = 0u; i < boost::size(counts); ++i) {
        for (core_t::TTime time = 0; time < counts[i]; ++time) {
            addArrival(gatherer, m_ResourceMonitor, startTime + time, gatherer.personName(i));
        }
    }

    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(0);
        peopleToRemove.push_back(1);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer(model_t::E_EventRate,
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
                                       key,
                                       features,
                                       startTime,
                                       0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(expectedGatherer, m_ResourceMonitor, "p3"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(expectedGatherer, m_ResourceMonitor, "p4"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson(expectedGatherer, m_ResourceMonitor, "p5"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), addPerson(expectedGatherer, m_ResourceMonitor, "p6"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), addPerson(expectedGatherer, m_ResourceMonitor, "p7"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(5), addPerson(expectedGatherer, m_ResourceMonitor, "p8"));

        core_t::TTime expectedCounts[] = {5, 2, 0, 5, 7, 10};
        for (std::size_t i = 0u; i < boost::size(expectedCounts); ++i) {
            for (core_t::TTime time = 0; time < expectedCounts[i]; ++time) {
                addArrival(expectedGatherer, m_ResourceMonitor, startTime + time, expectedGatherer.personName(i));
            }
        }

        LOG_DEBUG("checksum          = " << gatherer.checksum());
        LOG_DEBUG("expected checksum = " << expectedGatherer.checksum());
        CPPUNIT_ASSERT_EQUAL(gatherer.checksum(), expectedGatherer.checksum());
    }
    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(3);
        peopleToRemove.push_back(4);
        peopleToRemove.push_back(7);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer(model_t::E_EventRate,
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
                                       key,
                                       features,
                                       startTime,
                                       0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(expectedGatherer, m_ResourceMonitor, "p3"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(expectedGatherer, m_ResourceMonitor, "p6"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson(expectedGatherer, m_ResourceMonitor, "p7"));

        core_t::TTime expectedCounts[] = {5, 5, 7};
        for (std::size_t i = 0u; i < boost::size(expectedCounts); ++i) {
            for (core_t::TTime time = 0; time < expectedCounts[i]; ++time) {
                addArrival(expectedGatherer, m_ResourceMonitor, startTime + time, expectedGatherer.personName(i));
            }
        }

        LOG_DEBUG("checksum          = " << gatherer.checksum());
        LOG_DEBUG("expected checksum = " << expectedGatherer.checksum());
        CPPUNIT_ASSERT_EQUAL(gatherer.checksum(), expectedGatherer.checksum());
    }
    {
        TSizeVec peopleToRemove;
        peopleToRemove.push_back(2);
        peopleToRemove.push_back(5);
        peopleToRemove.push_back(6);
        gatherer.recyclePeople(peopleToRemove);

        CDataGatherer expectedGatherer(model_t::E_EventRate,
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
                                       key,
                                       features,
                                       startTime,
                                       0);

        LOG_DEBUG("checksum          = " << gatherer.checksum());
        LOG_DEBUG("expected checksum = " << expectedGatherer.checksum());
        CPPUNIT_ASSERT_EQUAL(gatherer.checksum(), expectedGatherer.checksum());
    }

    TSizeVec expectedRecycled;
    expectedRecycled.push_back(addPerson(gatherer, m_ResourceMonitor, "p1"));
    expectedRecycled.push_back(addPerson(gatherer, m_ResourceMonitor, "p7"));

    LOG_DEBUG("recycled          = " << core::CContainerPrinter::print(gatherer.recycledPersonIds()));
    LOG_DEBUG("expected recycled = " << core::CContainerPrinter::print(expectedRecycled));
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedRecycled), core::CContainerPrinter::print(gatherer.recycledPersonIds()));
}

void CEventRateDataGathererTest::singleSeriesOutOfOrderFinalResultTests() {
    LOG_DEBUG("*** singleSeriesOutOfOrderFinalResultTests ***");

    // Test that the various statistics come back as we expect.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(3);
    core_t::TTime latencyTime = bucketLength * static_cast<core_t::TTime>(latencyBuckets);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    core_t::TTime data[] = {
        1,
        180,
        1200,
        190,
        400,
        600, // bucket 1, 2 & 3
        550,
        799,
        1199,
        15,   // bucket 1 & 2
        2490, // bucket 5
              // bucket 4 is empty
        2420,
        2480,
        1250, // bucket 3 & 5
        10000 // sentinel
    };

    std::string expectedPersonCounts[] = {
        std::string("[(0, 6)]"), std::string("[(0, 3)]"), std::string("[(0, 2)]"), std::string("[(0, 0)]"), std::string("[(0, 3)]")};

    std::string expectedPersonNonZeroCounts[] = {
        std::string("[(0, 6)]"), std::string("[(0, 3)]"), std::string("[(0, 2)]"), std::string("[]"), std::string("[(0, 3)]")};

    std::string expectedPersonIndicator[] = {
        std::string("[(0, 1)]"), std::string("[(0, 1)]"), std::string("[(0, 1)]"), std::string("[]"), std::string("[(0, 1)]")};

    // Test the count by bucket and person and bad feature
    // (which should be ignored).
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        addPerson(gatherer, m_ResourceMonitor, "p");

        core_t::TTime time = startTime;
        for (std::size_t i = 0, j = 0u; i < boost::size(data); ++i) {
            for (/**/; j < 5 && data[i] >= time + latencyTime; time += bucketLength, ++j, gatherer.timeNow(time)) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonCounts[j], core::CContainerPrinter::print(featureData[0].second));

                testPersistence(params, gatherer);
            }

            if (j < 5) {
                LOG_DEBUG("Arriving = " << data[i]);
                addArrival(gatherer, m_ResourceMonitor, data[i], "p");
            }
        }
    }

    // Test non-zero count and person bucket count.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualNonZeroCountByBucketAndPerson);
        features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p"));

        core_t::TTime time = startTime;
        for (std::size_t i = 0, j = 0u; i < boost::size(data); ++i) {
            for (/**/; j < 5 && data[i] >= time + latencyTime; time += bucketLength, ++j, gatherer.timeNow(time)) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(2), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualNonZeroCountByBucketAndPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonNonZeroCounts[j], core::CContainerPrinter::print(featureData[0].second));
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualTotalBucketCountByPerson, featureData[1].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonNonZeroCounts[j], core::CContainerPrinter::print(featureData[1].second));

                testPersistence(params, gatherer);
            }

            if (j < 5) {
                addArrival(gatherer, m_ResourceMonitor, data[i], "p");
            }
        }
    }

    // Test test person indicator by bucket.
    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualIndicatorOfBucketPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p"));

        core_t::TTime time = startTime;
        for (std::size_t i = 0, j = 0u; i < boost::size(data); ++i) {
            for (/**/; j < 5 && data[i] >= time + latencyTime; time += bucketLength, ++j, gatherer.timeNow(time)) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonIndicator[j], core::CContainerPrinter::print(featureData[0].second));

                testPersistence(params, gatherer);
            }

            if (j < 5) {
                addArrival(gatherer, m_ResourceMonitor, data[i], "p");
            }
        }
    }
}

void CEventRateDataGathererTest::singleSeriesOutOfOrderInterimResultTests() {
    LOG_DEBUG("*** singleSeriesOutOfOrderInterimResultTests ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(3);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    core_t::TTime data[] = {
        1,
        1200,
        600, // bucket 1, 3 & 2
        1199,
        15,   // bucket 2 & 1
        2490, // bucket 5
              // bucket 4 is empty
        2420,
        1250 // bucket 5 & 3
    };

    TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_EventRate,
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
                           key,
                           features,
                           startTime,
                           0);
    addPerson(gatherer, m_ResourceMonitor, "p");
    TFeatureSizeFeatureDataPrVecPrVec featureData;

    // Bucket 1 only
    addArrival(gatherer, m_ResourceMonitor, data[0], "p");

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 1, 2 & 3
    addArrival(gatherer, m_ResourceMonitor, data[1], "p");

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 1, 2 & 3
    addArrival(gatherer, m_ResourceMonitor, data[2], "p");

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 1, 2 & 3
    addArrival(gatherer, m_ResourceMonitor, data[3], "p");

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 1, 2 & 3
    addArrival(gatherer, m_ResourceMonitor, data[4], "p");

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 3, 4 & 5
    addArrival(gatherer, m_ResourceMonitor, data[5], "p");

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1800, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(2400, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 3, 4 & 5
    addArrival(gatherer, m_ResourceMonitor, data[6], "p");

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1800, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(2400, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));

    // Bucket 3, 4 & 5
    addArrival(gatherer, m_ResourceMonitor, data[7], "p");

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(1800, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));
    gatherer.featureData(2400, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));
}

void CEventRateDataGathererTest::multipleSeriesOutOfOrderFinalResultTests() {
    LOG_DEBUG("*** multipleSeriesOutOfOrderFinalResultTests ***");

    // Test that the various statistics come back as we expect
    // for multiple people.

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(3);
    core_t::TTime latencyTime = bucketLength * static_cast<core_t::TTime>(latencyBuckets);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    core_t::TTime data1[] = {
        1,
        15,
        1200,
        190,
        400,
        550, // bucket 1, 2 & 3
        600,
        1250,
        1199, // bucket 2 & 3
        180,
        799,  // bucket 1 & 2
        2480, // bucket 5
        2420,
        1900,
        2490, // bucket 4 & 5
        10000 // sentinel
    };
    core_t::TTime data2[] = {
        1250, 5,    15,   600,  180,  190,  400, 550, // bucket 1, 2 & 3
        25,   605,  609,  799,  1199,                 // bucket 1 & 2
        1200, 1,    1255, 1950, 1400,                 // bucket 1, 3 & 4
        2550, 1300, 2500,                             // bucket 3 & 5
        2420, 2480, 2490, 1256, 1900, 2600,           // bucket 3, 4 & 5
        10000                                         // sentinel
    };

    std::string expectedPersonCounts[] = {std::string("[(0, 6), (1, 8)]"),
                                          std::string("[(0, 3), (1, 5)]"),
                                          std::string("[(0, 2), (1, 6)]"),
                                          std::string("[(0, 1), (1, 2)]"),
                                          std::string("[(0, 3), (1, 6)]")};

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p1"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(gatherer, m_ResourceMonitor, "p2"));

        core_t::TTime time = startTime;
        std::size_t i1 = 0u, i2 = 0u, j = 0u;
        for (;;) {
            for (/**/; j < 5 && std::min(data1[i1], data2[i2]) >= time + latencyTime; time += bucketLength, ++j) {
                LOG_DEBUG("Processing bucket [" << time << ", " << time + bucketLength << ")");

                TFeatureSizeFeatureDataPrVecPrVec featureData;
                gatherer.featureData(time, bucketLength, featureData);
                LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
                CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
                CPPUNIT_ASSERT_EQUAL(expectedPersonCounts[j], core::CContainerPrinter::print(featureData[0].second));

                testPersistence(params, gatherer);
            }

            if (j >= 5) {
                break;
            }

            if (data1[i1] < data2[i2]) {
                LOG_DEBUG("Adding arrival for p1 at " << data1[i1]);
                addArrival(gatherer, m_ResourceMonitor, data1[i1], "p1");
                ++i1;
            } else {
                LOG_DEBUG("Adding arrival for p2 at " << data2[i2]);
                addArrival(gatherer, m_ResourceMonitor, data2[i2], "p2");
                ++i2;
            }
        }

        TSizeVec peopleToRemove;
        peopleToRemove.push_back(1);
        gatherer.recyclePeople(peopleToRemove);

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::string("p1"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p1", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), pid);
        CPPUNIT_ASSERT(!gatherer.personId("p2", pid));

        TFeatureSizeFeatureDataPrVecPrVec featureData;
        gatherer.featureData(startTime + 4 * bucketLength, bucketLength, featureData);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 3)]"), core::CContainerPrinter::print(featureData[0].second));
    }

    {
        TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
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
                               key,
                               features,
                               startTime,
                               0);
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p1"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), addPerson(gatherer, m_ResourceMonitor, "p2"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), addPerson(gatherer, m_ResourceMonitor, "p3"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(3), addPerson(gatherer, m_ResourceMonitor, "p4"));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), addPerson(gatherer, m_ResourceMonitor, "p5"));

        for (std::size_t i = 0u; i < 5; ++i) {
            addArrival(gatherer, m_ResourceMonitor, startTime, gatherer.personName(i));
        }
        addArrival(gatherer, m_ResourceMonitor, startTime + 1, gatherer.personName(2));
        addArrival(gatherer, m_ResourceMonitor, startTime + 2, gatherer.personName(4));
        addArrival(gatherer, m_ResourceMonitor, startTime + 3, gatherer.personName(4));

        TSizeUInt64PrVec personCounts;

        TFeatureSizeFeatureDataPrVecPrVec featureData;
        gatherer.featureData(startTime, bucketLength, featureData);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (1, 1), (2, 2), (3, 1), (4, 3)]"),
                             core::CContainerPrinter::print(featureData[0].second));

        TSizeVec peopleToRemove;
        peopleToRemove.push_back(0);
        peopleToRemove.push_back(1);
        peopleToRemove.push_back(3);
        gatherer.recyclePeople(peopleToRemove);

        CPPUNIT_ASSERT_EQUAL(std::size_t(2), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::string("p3"), gatherer.personName(2));
        CPPUNIT_ASSERT_EQUAL(std::string("p5"), gatherer.personName(4));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(0));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(1));
        CPPUNIT_ASSERT_EQUAL(std::string("-"), gatherer.personName(3));
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId("p3", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(2), pid);
        CPPUNIT_ASSERT(gatherer.personId("p5", pid));
        CPPUNIT_ASSERT_EQUAL(std::size_t(4), pid);

        gatherer.featureData(startTime, bucketLength, featureData);
        LOG_DEBUG("featureData = " << core::CContainerPrinter::print(featureData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
        CPPUNIT_ASSERT_EQUAL(model_t::E_IndividualCountByBucketAndPerson, featureData[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("[(2, 2), (4, 3)]"), core::CContainerPrinter::print(featureData[0].second));
    }
}

void CEventRateDataGathererTest::testArrivalBeforeLatencyWindowIsIgnored() {
    LOG_DEBUG("*** testArrivalBeforeLatencyWindowIsIgnored ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(2);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    core_t::TTime data[] = {
        1800, // Bucket 4, thus bucket 1 values are already out of latency window
        1     // Bucket 1
    };

    TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_EventRate,
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
                           key,
                           features,
                           startTime,
                           0);
    addPerson(gatherer, m_ResourceMonitor, "p");

    addArrival(gatherer, m_ResourceMonitor, data[0], "p");
    addArrival(gatherer, m_ResourceMonitor, data[1], "p");

    TFeatureSizeFeatureDataPrVecPrVec featureData;

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), featureData.size());

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(1800, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
}

void CEventRateDataGathererTest::testResetBucketGivenSingleSeries() {
    LOG_DEBUG("*** testResetBucketGivenSingleSeries ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(2);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    core_t::TTime data[] = {
        100,
        300, // Bucket 1
        600,
        800,
        850, // Bucket 2
        1200 // Bucket 3
    };

    TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_EventRate,
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
                           key,
                           features,
                           startTime,
                           0);
    addPerson(gatherer, m_ResourceMonitor, "p");

    for (std::size_t i = 0; i < boost::size(data); ++i) {
        addArrival(gatherer, m_ResourceMonitor, data[i], "p");
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 3)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.resetBucket(600);

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1)]"), core::CContainerPrinter::print(featureData[0].second));
}

void CEventRateDataGathererTest::testResetBucketGivenMultipleSeries() {
    LOG_DEBUG("*** testResetBucketGivenMultipleSeries ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(2);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    core_t::TTime data[] = {
        100,
        300, // Bucket 1
        600,
        800,
        850, // Bucket 2
        1200 // Bucket 3
    };

    TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_EventRate,
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
                           key,
                           features,
                           startTime,
                           0);
    addPerson(gatherer, m_ResourceMonitor, "p1");
    addPerson(gatherer, m_ResourceMonitor, "p2");
    addPerson(gatherer, m_ResourceMonitor, "p3");

    for (std::size_t i = 0; i < boost::size(data); ++i) {
        addArrival(gatherer, m_ResourceMonitor, data[i], "p1");
        addArrival(gatherer, m_ResourceMonitor, data[i], "p2");
        addArrival(gatherer, m_ResourceMonitor, data[i], "p3");
    }

    TFeatureSizeFeatureDataPrVecPrVec featureData;

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2), (1, 2), (2, 2)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 3), (1, 3), (2, 3)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (1, 1), (2, 1)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.resetBucket(600);

    gatherer.featureData(0, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 2), (1, 2), (2, 2)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(600, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0), (1, 0), (2, 0)]"), core::CContainerPrinter::print(featureData[0].second));

    gatherer.featureData(1200, bucketLength, featureData);
    CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (1, 1), (2, 1)]"), core::CContainerPrinter::print(featureData[0].second));
}

void CEventRateDataGathererTest::testResetBucketGivenBucketNotAvailable() {
    LOG_DEBUG("*** testResetBucketGivenBucketNotAvailable ***");

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    std::size_t latencyBuckets(1);
    SModelParams params(bucketLength);
    params.s_LatencyBuckets = latencyBuckets;

    TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    CDataGatherer gatherer(model_t::E_EventRate,
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
                           key,
                           features,
                           startTime,
                           0);
    addPerson(gatherer, m_ResourceMonitor, "p");

    addArrival(gatherer, m_ResourceMonitor, 1200, "p");

    CPPUNIT_ASSERT(gatherer.resetBucket(0) == false);
    CPPUNIT_ASSERT(gatherer.resetBucket(600));
    CPPUNIT_ASSERT(gatherer.resetBucket(1200));
    CPPUNIT_ASSERT(gatherer.resetBucket(1800) == false);
}

void CEventRateDataGathererTest::testInfluencerBucketStatistics() {
    core_t::TTime data[] = {
        1,
        15,
        180,
        190,
        400,
        550, // bucket 1
        600,
        799,
        1199, // bucket 2
        1200,
        1250, // bucket 3
              // bucket 4
        2420,
        2480,
        2490, // bucket 5
        10000 // sentinel
    };
    TTimeVec dataVec(data, &data[15]);

    TStrVecVec influencers(14, TStrVec(1, "i"));

    std::string expectedPersonCounts[] = {std::string("[(0, 6, [[(i, ([6], 1))]])]"),
                                          std::string("[(0, 3, [[(i, ([3], 1))]])]"),
                                          std::string("[(0, 2, [[(i, ([2], 1))]])]"),
                                          std::string("[(0, 0)]"),
                                          std::string("[(0, 3, [[(i, ([3], 1))]])]")};
    TStrVec expectedPersonCountsVec(&expectedPersonCounts[0], &expectedPersonCounts[5]);

    std::string expectedPersonNonZeroCounts[] = {std::string("[(0, 6, [[(i, ([6], 1))]])]"),
                                                 std::string("[(0, 3, [[(i, ([3], 1))]])]"),
                                                 std::string("[(0, 2, [[(i, ([2], 1))]])]"),
                                                 std::string("[]"),
                                                 std::string("[(0, 3, [[(i, ([3], 1))]])]")};
    TStrVec expectedPersonNonZeroCountsVec(&expectedPersonNonZeroCounts[0], &expectedPersonNonZeroCounts[5]);

    std::string expectedPersonIndicator[] = {std::string("[(0, 1, [[(i, ([1], 1))]])]"),
                                             std::string("[(0, 1, [[(i, ([1], 1))]])]"),
                                             std::string("[(0, 1, [[(i, ([1], 1))]])]"),
                                             std::string("[]"),
                                             std::string("[(0, 1, [[(i, ([1], 1))]])]")};
    TStrVec expectedPersonIndicatorVec(&expectedPersonIndicator[0], &expectedPersonIndicator[5]);

    TStrVec expectedArrivalTimeVec(6, std::string("[]"));

    std::string expectedInfoContent[] = {std::string("[(0, 13, [[(i, ([13], 1))]])]"),
                                         std::string("[(0, 13, [[(i, ([13], 1))]])]"),
                                         std::string("[(0, 13, [[(i, ([13], 1))]])]"),
                                         std::string("[]"),
                                         std::string("[(0, 13, [[(i, ([13], 1))]])]")};
    TStrVec expectedInfoContentVec(&expectedInfoContent[0], &expectedInfoContent[5]);

    testInfluencerPerFeature(
        model_t::E_IndividualCountByBucketAndPerson, dataVec, influencers, expectedPersonCountsVec, "", m_ResourceMonitor);

    testInfluencerPerFeature(
        model_t::E_IndividualNonZeroCountByBucketAndPerson, dataVec, influencers, expectedPersonNonZeroCountsVec, "", m_ResourceMonitor);

    testInfluencerPerFeature(
        model_t::E_IndividualLowCountsByBucketAndPerson, dataVec, influencers, expectedPersonCountsVec, "", m_ResourceMonitor);

    testInfluencerPerFeature(
        model_t::E_IndividualArrivalTimesByPerson, dataVec, influencers, expectedArrivalTimeVec, "", m_ResourceMonitor);

    testInfluencerPerFeature(
        model_t::E_IndividualLowNonZeroCountByBucketAndPerson, dataVec, influencers, expectedPersonNonZeroCountsVec, "", m_ResourceMonitor);

    testInfluencerPerFeature(
        model_t::E_IndividualUniqueCountByBucketAndPerson, dataVec, influencers, expectedPersonIndicatorVec, "value", m_ResourceMonitor);

    testInfluencerPerFeature(
        model_t::E_IndividualInfoContentByBucketAndPerson, dataVec, influencers, expectedInfoContentVec, "value", m_ResourceMonitor);
}

void CEventRateDataGathererTest::testDistinctStrings() {
    using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;
    TSizeSizePr pair(0, 0);

    // Test the SUniqueStringFeatureData struct
    {
        // Check adding values with no influences gives the correct count
        // for distinct_count
        CUniqueStringFeatureData data;
        TStoredStringPtrVec influencers;

        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("0"), featureData.print());
        }

        for (std::size_t i = 0; i < 100; ++i) {
            data.insert("str1", influencers);
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("1"), featureData.print());
        }
        for (std::size_t i = 0; i < 100; ++i) {
            data.insert("str2", influencers);
            data.insert("str3", influencers);
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("3"), featureData.print());
        }
        for (std::size_t i = 1; i < 100; ++i) {
            std::stringstream ss;
            ss << "str" << i;
            data.insert(ss.str(), influencers);
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::max(uint64_t(3), uint64_t(i)), featureData.s_Count);
        }
    }
    {
        // Check we can add influencers now - just 1
        // First with a non-present value
        CUniqueStringFeatureData data;
        TStoredStringPtrVec influencers;
        influencers.push_back(core::CStoredStringPtr());

        data.insert("str1", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("1, [[]]"), featureData.print());
        }

        influencers.back() = CStringStore::influencers().get("inf1");
        data.insert("str1", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("1, [[(inf1, ([1], 1))]]"), featureData.print());
        }

        data.insert("str2", influencers);
        data.insert("str2", influencers);
        data.insert("str2", influencers);
        data.insert("str2", influencers);
        influencers.back() = CStringStore::influencers().get("inf2");
        data.insert("str1", influencers);
        data.insert("str3", influencers);
        influencers.back() = CStringStore::influencers().get("inf3");
        data.insert("str3", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);

            std::sort(featureData.s_InfluenceValues[0].begin(), featureData.s_InfluenceValues[0].end(), maths::COrderings::SFirstLess());

            CPPUNIT_ASSERT_EQUAL(std::string("3, [[(inf1, ([2], 1)), (inf2, ([2], 1)), (inf3, ([1], 1))]]"), featureData.print());
        }
    }

    {
        // Check we can add more than one influencer
        CUniqueStringFeatureData data;
        TStoredStringPtrVec influencers;
        influencers.push_back(core::CStoredStringPtr());
        influencers.push_back(core::CStoredStringPtr());

        data.insert("str1", influencers);
        data.insert("str2", influencers);
        data.insert("str1", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("2, [[], []]"), featureData.print());
        }

        influencers[0] = CStringStore::influencers().get("inf1");
        data.insert("str1", influencers);
        data.insert("str2", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("2, [[(inf1, ([2], 1))], []]"), featureData.print());
        }
        influencers[1] = CStringStore::influencers().get("inf_v2");

        data.insert("str2", influencers);
        influencers[0] = CStringStore::influencers().get("inf2");
        influencers[1] = CStringStore::influencers().get("inf_v3");
        data.insert("str3", influencers);
        data.insert("str1", influencers);
        data.insert("str3", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateDistinctCountFeatureData(featureData);
            for (std::size_t i = 0; i < 2; i++) {
                std::sort(
                    featureData.s_InfluenceValues[i].begin(), featureData.s_InfluenceValues[i].end(), maths::COrderings::SFirstLess());
            }
            CPPUNIT_ASSERT_EQUAL(std::string("3, [[(inf1, ([2], 1)), (inf2, ([2], 1))], [(inf_v2, ([1], 1)), (inf_v3, ([2], 1))]]"),
                                 featureData.print());
        }
    }
    {
        // Now test info_content - compressed strings
        // Check adding values with no influences gives the correct count
        CUniqueStringFeatureData data;
        TStoredStringPtrVec influencers;

        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("0"), featureData.print());
        }

        {
            data.insert("str1", influencers);
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("12"), featureData.print());
        }

        {
            data.insert("str2", influencers);
            data.insert("str3", influencers);
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("18"), featureData.print());
        }

        for (std::size_t i = 1; i < 100; ++i) {
            std::stringstream ss;
            ss << "str" << i;
            data.insert(ss.str(), influencers);
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT((featureData.s_Count - 12) >= std::max(uint64_t(3), uint64_t(i)));
            CPPUNIT_ASSERT((featureData.s_Count - 12) <= std::max(uint64_t(3), uint64_t(i)) * 3);
        }
    }
    {
        // Check we can add influencers now - just 1
        // First with a non-present value
        CUniqueStringFeatureData data;
        TStoredStringPtrVec influencers;
        influencers.push_back(core::CStoredStringPtr());

        data.insert("str1", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("12, [[]]"), featureData.print());
        }

        influencers.back() = CStringStore::influencers().get("inf1");
        data.insert("str1", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("12, [[(inf1, ([12], 1))]]"), featureData.print());
        }

        data.insert("str2", influencers);
        data.insert("str2", influencers);
        data.insert("str2", influencers);
        data.insert("str2", influencers);
        influencers.back() = CStringStore::influencers().get("inf2");
        data.insert("str1", influencers);
        data.insert("str3", influencers);
        influencers.back() = CStringStore::influencers().get("inf3");
        data.insert("str3", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);

            std::sort(featureData.s_InfluenceValues[0].begin(), featureData.s_InfluenceValues[0].end(), maths::COrderings::SFirstLess());

            CPPUNIT_ASSERT_EQUAL(std::string("18, [[(inf1, ([16], 1)), (inf2, ([16], 1)), (inf3, ([12], 1))]]"), featureData.print());
        }
    }
    {
        // Check we can add more than one influencer
        CUniqueStringFeatureData data;
        TStoredStringPtrVec influencers;
        influencers.push_back(core::CStoredStringPtr());
        influencers.push_back(core::CStoredStringPtr());

        data.insert("str1", influencers);
        data.insert("str2", influencers);
        data.insert("str1", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("16, [[], []]"), featureData.print());
        }

        influencers[0] = CStringStore::influencers().get("inf1");
        data.insert("str1", influencers);
        data.insert("str2", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            CPPUNIT_ASSERT_EQUAL(std::string("16, [[(inf1, ([16], 1))], []]"), featureData.print());
        }
        influencers[1] = CStringStore::influencers().get("inf_v2");

        data.insert("str2", influencers);
        influencers[0] = CStringStore::influencers().get("inf2");
        influencers[1] = CStringStore::influencers().get("inf_v3");
        data.insert("str3", influencers);
        data.insert("str1", influencers);
        data.insert("str3", influencers);
        {
            SEventRateFeatureData featureData(0);
            data.populateInfoContentFeatureData(featureData);
            for (std::size_t i = 0; i < 2; i++) {
                std::sort(
                    featureData.s_InfluenceValues[i].begin(), featureData.s_InfluenceValues[i].end(), maths::COrderings::SFirstLess());
            }
            CPPUNIT_ASSERT_EQUAL(std::string("18, [[(inf1, ([16], 1)), (inf2, ([16], 1))], [(inf_v2, ([12], 1)), (inf_v3, ([16], 1))]]"),
                                 featureData.print());
        }
    }
    {
        // Check that we can add distinct strings in latency buckets and restore them
        core_t::TTime bucketLength(1800);
        core_t::TTime startTime(1432733400);
        std::size_t latencyBuckets(3);
        SModelParams params(bucketLength);
        params.s_LatencyBuckets = latencyBuckets;

        TFeatureVec features;
        features.push_back(model_t::E_IndividualUniqueCountByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               "P",
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "V",
                               TStrVec(1, "INF"),
                               false,
                               key,
                               features,
                               startTime,
                               0);

        CPPUNIT_ASSERT(!gatherer.isPopulation());
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson(gatherer, m_ResourceMonitor, "p", "v", 1));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 1; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }
        CPPUNIT_ASSERT(gatherer.hasFeature(model_t::E_IndividualUniqueCountByBucketAndPerson));

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        core_t::TTime time = startTime;

        CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());
        testPersistence(params, gatherer);

        // Add data, some of which will be out of order
        addArrival(gatherer, m_ResourceMonitor, time - (2 * bucketLength), "p", "stringOne", "inf1");
        addArrival(gatherer, m_ResourceMonitor, time - (2 * bucketLength), "p", "stringTwo", "inf2");
        addArrival(gatherer, m_ResourceMonitor, time - (1 * bucketLength), "p", "stringThree", "inf3");
        addArrival(gatherer, m_ResourceMonitor, time - (1 * bucketLength), "p", "stringFour", "inf1");
        addArrival(gatherer, m_ResourceMonitor, time, "p", "stringFive", "inf2");
        addArrival(gatherer, m_ResourceMonitor, time, "p", "stringSix", "inf3");
        testPersistence(params, gatherer);
    }
}

void CEventRateDataGathererTest::testDiurnalFeatures() {
    LOG_DEBUG("*** testDiurnalFeatures ***");
    const std::string person("p");
    const std::string attribute("a");
    const std::string emptyString("");
    {
        // Check that we can add time-of-day events and gather them correctly
        LOG_DEBUG("Testing time_of_day by person");

        core_t::TTime bucketLength(3600);
        core_t::TTime startTime(1432731600);
        std::size_t latencyBuckets(3);
        SModelParams params(bucketLength);
        params.s_LatencyBuckets = latencyBuckets;

        TFeatureVec features;
        features.push_back(model_t::E_IndividualTimeOfDayByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "person",
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               key,
                               features,
                               startTime,
                               0);

        CPPUNIT_ASSERT(!gatherer.isPopulation());

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 1; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }
        CPPUNIT_ASSERT(gatherer.hasFeature(model_t::E_IndividualTimeOfDayByBucketAndPerson));

        core_t::TTime time = startTime;

        CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());
        testPersistence(params, gatherer);

        // Add some data, and check that we get the right numbers out of the featureData
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 86400), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 100, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 50), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 86400), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 86400), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 300, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 150), featureData[0].second[0].second.s_Count);
        }

        // Check latency by going backwards in time
        time -= bucketLength * 2;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 400, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 200), featureData[0].second[0].second.s_Count);
        }

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        testPersistence(params, gatherer);
    }
    {
        // Check that we can add time-of-week events and gather them correctly
        LOG_DEBUG("Testing time_of_week by person");

        core_t::TTime bucketLength(3600);
        core_t::TTime startTime(1432731600);
        std::size_t latencyBuckets(3);
        SModelParams params(bucketLength);
        params.s_LatencyBuckets = latencyBuckets;

        TFeatureVec features;
        features.push_back(model_t::E_IndividualTimeOfWeekByBucketAndPerson);
        CDataGatherer gatherer(model_t::E_EventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "person",
                               EMPTY_STRING,
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               key,
                               features,
                               startTime,
                               0);

        CPPUNIT_ASSERT(!gatherer.isPopulation());

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 1; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }
        CPPUNIT_ASSERT(gatherer.hasFeature(model_t::E_IndividualTimeOfWeekByBucketAndPerson));

        core_t::TTime time = startTime;

        CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());
        testPersistence(params, gatherer);

        // Add some data, and check that we get the right numbers out of the featureData
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 604800), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 100, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 50), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 604800), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 604800), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 300, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 150), featureData[0].second[0].second.s_Count);
        }

        // Check latency by going backwards in time
        time -= bucketLength * 2;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 400, person);

            TFeatureSizeFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 200), featureData[0].second[0].second.s_Count);
        }

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("p"), gatherer.personName(0));
        testPersistence(params, gatherer);
    }
    {
        // Check that we can add time-of-week events and gather them correctly
        LOG_DEBUG("Testing time_of_week over person");

        core_t::TTime bucketLength(3600);
        core_t::TTime startTime(1432731600);
        std::size_t latencyBuckets(3);
        SModelParams params(bucketLength);
        params.s_LatencyBuckets = latencyBuckets;

        TFeatureVec features;
        features.push_back(model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute);
        CDataGatherer gatherer(model_t::E_PopulationEventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "att",
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               key,
                               features,
                               startTime,
                               0);

        CPPUNIT_ASSERT(gatherer.isPopulation());

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 1; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }
        CPPUNIT_ASSERT(gatherer.hasFeature(model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute));

        core_t::TTime time = startTime;

        CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());
        testPersistence(params, gatherer);

        // Add some data, and check that we get the right numbers out of the featureData
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 604800), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 100, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 50), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 604800), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 604800), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 300, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 150), featureData[0].second[0].second.s_Count);
        }

        // Check latency by going backwards in time
        time -= bucketLength * 2;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 400, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 604800) + 200), featureData[0].second[0].second.s_Count);
        }

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActiveAttributes());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("a"), gatherer.attributeName(0));
        testPersistence(params, gatherer);
    }
    {
        // Check that we can add time-of-day events and gather them correctly
        LOG_DEBUG("Testing time_of_day over person");

        core_t::TTime bucketLength(3600);
        core_t::TTime startTime(1432731600);
        std::size_t latencyBuckets(3);
        SModelParams params(bucketLength);
        params.s_LatencyBuckets = latencyBuckets;

        TFeatureVec features;
        features.push_back(model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute);
        CDataGatherer gatherer(model_t::E_PopulationEventRate,
                               model_t::E_None,
                               params,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               EMPTY_STRING,
                               "att",
                               EMPTY_STRING,
                               TStrVec(),
                               false,
                               key,
                               features,
                               startTime,
                               0);

        CPPUNIT_ASSERT(gatherer.isPopulation());

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberFeatures());
        for (std::size_t i = 0u; i < 1; ++i) {
            CPPUNIT_ASSERT_EQUAL(features[i], gatherer.feature(i));
        }
        CPPUNIT_ASSERT(gatherer.hasFeature(model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute));

        core_t::TTime time = startTime;

        CPPUNIT_ASSERT_EQUAL(bucketLength, gatherer.bucketLength());
        testPersistence(params, gatherer);

        // Add some data, and check that we get the right numbers out of the featureData
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 86400), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 100, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 50), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 86400), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 0, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t(time % 86400), featureData[0].second[0].second.s_Count);
        }
        {
            addArrival(gatherer, m_ResourceMonitor, time + 300, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 150), featureData[0].second[0].second.s_Count);
        }

        // Check latency by going backwards in time
        time -= bucketLength * 2;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 200, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 100), featureData[0].second[0].second.s_Count);
        }
        time += bucketLength;
        {
            addArrival(gatherer, m_ResourceMonitor, time + 400, person, attribute);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(time, bucketLength, featureData);
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData.size());
            CPPUNIT_ASSERT_EQUAL(std::size_t(1), featureData[0].second.size());
            CPPUNIT_ASSERT_EQUAL(uint64_t((time % 86400) + 200), featureData[0].second[0].second.s_Count);
        }

        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActivePeople());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberActiveAttributes());
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gatherer.numberByFieldValues());
        CPPUNIT_ASSERT_EQUAL(std::string("a"), gatherer.attributeName(0));
        testPersistence(params, gatherer);
    }
}

CppUnit::Test* CEventRateDataGathererTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CEventRateDataGathererTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::singleSeriesTests",
                                                                              &CEventRateDataGathererTest::singleSeriesTests));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::multipleSeriesTests",
                                                                              &CEventRateDataGathererTest::multipleSeriesTests));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testRemovePeople",
                                                                              &CEventRateDataGathererTest::testRemovePeople));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::singleSeriesOutOfOrderFinalResultTests",
                                                            &CEventRateDataGathererTest::singleSeriesOutOfOrderFinalResultTests));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::singleSeriesOutOfOrderInterimResultTests",
                                                            &CEventRateDataGathererTest::singleSeriesOutOfOrderInterimResultTests));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::multipleSeriesOutOfOrderFinalResultTests",
                                                            &CEventRateDataGathererTest::multipleSeriesOutOfOrderFinalResultTests));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testArrivalBeforeLatencyWindowIsIgnored",
                                                            &CEventRateDataGathererTest::testArrivalBeforeLatencyWindowIsIgnored));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>(
        "CEventRateDataGathererTest::testResetBucketGivenSingleSeries", &CEventRateDataGathererTest::testResetBucketGivenSingleSeries));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>(
        "CEventRateDataGathererTest::testResetBucketGivenMultipleSeries", &CEventRateDataGathererTest::testResetBucketGivenMultipleSeries));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testResetBucketGivenBucketNotAvailable",
                                                            &CEventRateDataGathererTest::testResetBucketGivenBucketNotAvailable));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testInfluencerBucketStatistics",
                                                                              &CEventRateDataGathererTest::testInfluencerBucketStatistics));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testDistinctStrings",
                                                                              &CEventRateDataGathererTest::testDistinctStrings));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testLatencyPersist",
                                                                              &CEventRateDataGathererTest::testLatencyPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateDataGathererTest>("CEventRateDataGathererTest::testDiurnalFeatures",
                                                                              &CEventRateDataGathererTest::testDiurnalFeatures));
    return suiteOfTests;
}
