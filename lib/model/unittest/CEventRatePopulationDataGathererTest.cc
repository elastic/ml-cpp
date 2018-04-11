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

#include "CEventRatePopulationDataGathererTest.h"

#include <core/CCompressUtils.h>
#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CEventRateBucketGatherer.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CModelParams.h>
#include <model/CResourceMonitor.h>

#include <test/CRandomNumbers.h>

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range.hpp>
#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <vector>

using namespace ml;
using namespace model;

namespace {

struct SMessage {
    SMessage(core_t::TTime time, const std::string& attribute, const std::string& person)
        : s_Time(time), s_Attribute(attribute), s_Person(person) {}

    bool operator<(const SMessage& other) const {
        return s_Time < other.s_Time;
    }

    core_t::TTime s_Time;
    std::string s_Attribute;
    std::string s_Person;
};

using TMessageVec = std::vector<SMessage>;
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;
using TEventRateDataGathererPtr = boost::shared_ptr<CDataGatherer>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrUInt64Map = std::map<TSizeSizePr, uint64_t>;
using TSizeSizePrUInt64MapCItr = TSizeSizePrUInt64Map::iterator;
using TSizeSet = std::set<std::size_t>;
using TSizeSizeSetMap = std::map<std::size_t, TSizeSet>;
using TStrSet = std::set<std::string>;
using TSizeStrSetMap = std::map<std::size_t, TStrSet>;
using TSizeStrSetMapItr = std::map<std::size_t, TStrSet>::iterator;
using TFeatureData = SEventRateFeatureData;
using TStrFeatureDataPr = std::pair<std::string, TFeatureData>;
using TStrFeatureDataPrVec = std::vector<TStrFeatureDataPr>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, TFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TFeatureSizeSizePrFeatureDataPrVecPr =
    std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;

TStrVec allCategories() {
    const std::size_t numberCategories = 30u;
    TStrVec categories;
    for (std::size_t i = 0; i < numberCategories; ++i) {
        categories.push_back("c" + boost::lexical_cast<std::string>(i));
    }
    return categories;
}

TStrVec allPeople() {
    const std::size_t numberPeople = 5u;
    TStrVec people;
    for (std::size_t i = 0u; i < numberPeople; ++i) {
        people.push_back("p" + boost::lexical_cast<std::string>(i));
    }
    return people;
}

void generateTestMessages(test::CRandomNumbers& rng,
                          core_t::TTime time,
                          core_t::TTime bucketLength,
                          TMessageVec& messages) {
    using TUIntVec = std::vector<unsigned int>;
    using TDoubleVec = std::vector<double>;

    LOG_DEBUG(<< "bucket = [" << time << ", " << time + bucketLength << ")");

    TStrVec categories = allCategories();
    TStrVec people = allPeople();

    const double rates[] = {1.0, 0.3, 10.1, 25.0, 105.0};

    TSizeVec bucketCounts;
    for (std::size_t j = 0u; j < categories.size(); ++j) {
        double rate = rates[j % boost::size(rates)];
        TUIntVec sample;
        rng.generatePoissonSamples(rate, 1u, sample);
        bucketCounts.push_back(sample[0]);
    }

    TDoubleVec personRange;
    rng.generateUniformSamples(0.0, static_cast<double>(people.size()) - 1e-3, 2u, personRange);
    std::sort(personRange.begin(), personRange.end());
    std::size_t a = static_cast<std::size_t>(personRange[0]);
    std::size_t b = static_cast<std::size_t>(personRange[1]) + 1;
    TSizeVec bucketPeople;
    for (std::size_t i = a; i < b; ++i) {
        bucketPeople.push_back(i);
    }
    LOG_DEBUG(<< "bucketPeople = " << core::CContainerPrinter::print(bucketPeople));

    for (std::size_t i = 0u; i < categories.size(); ++i) {
        TDoubleVec offsets;
        rng.generateUniformSamples(0.0, static_cast<double>(bucketLength) - 1.0,
                                   bucketCounts[i], offsets);

        for (std::size_t j = 0u; j < offsets.size(); ++j) {
            messages.push_back(SMessage(
                time + static_cast<core_t::TTime>(offsets[j]), categories[i],
                people[bucketPeople[j % bucketPeople.size()]]));
        }
    }

    std::sort(messages.begin(), messages.end());
    LOG_DEBUG(<< "Generated " << messages.size() << " messages");
}

const TSizeSizePrFeatureDataPrVec&
extract(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData, model_t::EFeature feature) {
    for (std::size_t i = 0u; i < featureData.size(); ++i) {
        if (featureData[i].first == feature) {
            return featureData[i].second;
        }
    }

    static const TSizeSizePrFeatureDataPrVec EMPTY;
    return EMPTY;
}

const TSizeSizePrFeatureDataPrVec&
extractPeoplePerAttribute(TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationUniquePersonCountByAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractNonZeroAttributeCounts(TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationCountByBucketPersonAndAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractAttributeIndicator(TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationIndicatorOfBucketPersonAndAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractBucketAttributesPerPerson(TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationUniqueCountByBucketPersonAndAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractCompressedLengthPerPerson(TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationInfoContentByBucketPersonAndAttribute);
}

CEventData addArrival(core_t::TTime time,
                      const std::string& p,
                      const std::string& a,
                      CDataGatherer& gatherer,
                      CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&p);
    fields.push_back(&a);

    CEventData result;
    result.time(time);

    gatherer.addArrival(fields, result, resourceMonitor);
    return result;
}

CEventData addArrival(core_t::TTime time,
                      const std::string& p,
                      const std::string& a,
                      const std::string& v,
                      CDataGatherer& gatherer,
                      CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&p);
    fields.push_back(&a);
    fields.push_back(&v);

    CEventData result;
    result.time(time);

    gatherer.addArrival(fields, result, resourceMonitor);
    return result;
}

CSearchKey searchKey;
const std::string EMPTY_STRING;
}

void CEventRatePopulationDataGathererTest::testAttributeCounts() {
    LOG_DEBUG(
        << "*** CEventRatePopulationDataGathererTest::testAttributeCounts ***");

    // We check that we correctly sample the unique people per
    // attribute and (attribute, person) pair counts.

    using TStrSizeMap = std::map<std::string, std::size_t>;

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 20u;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    SModelParams params(bucketLength);
    CDataGatherer dataGatherer(model_t::E_PopulationEventRate, model_t::E_None,
                               params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
                               false, searchKey, features, startTime, 0);
    CPPUNIT_ASSERT(dataGatherer.isPopulation());

    CPPUNIT_ASSERT_EQUAL(startTime, dataGatherer.currentBucketStartTime());
    CPPUNIT_ASSERT_EQUAL(bucketLength, dataGatherer.bucketLength());

    TSizeSizeSetMap expectedAttributePeople;
    std::size_t attributeOrder = 0u;
    TStrSizeMap expectedAttributeOrder;
    std::size_t personOrder = 0u;
    TStrSizeMap expectedPeopleOrder;
    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeSizePrUInt64Map expectedAttributeCounts;
        for (std::size_t j = 0u; j < messages.size(); ++j) {
            addArrival(messages[j].s_Time, messages[j].s_Person,
                       messages[j].s_Attribute, dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(messages[j].s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(messages[j].s_Person, pid);

            ++expectedAttributeCounts[std::make_pair(pid, cid)];
            expectedAttributePeople[cid].insert(pid);
            if (expectedAttributeOrder
                    .insert(TStrSizeMap::value_type(messages[j].s_Attribute, attributeOrder))
                    .second) {
                ++attributeOrder;
            }
            if (expectedPeopleOrder
                    .insert(TStrSizeMap::value_type(messages[j].s_Person, personOrder))
                    .second) {
                ++personOrder;
            }
        }

        CPPUNIT_ASSERT(dataGatherer.dataAvailable(time));
        CPPUNIT_ASSERT(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        CPPUNIT_ASSERT_EQUAL(expectedAttributePeople.size(), peoplePerAttribute.size());
        TSizeSizePrFeatureDataPrVec expectedPeoplePerAttribute;
        for (std::size_t j = 0u; j < peoplePerAttribute.size(); ++j) {
            expectedPeoplePerAttribute.push_back(TSizeSizePrFeatureDataPr(
                std::make_pair(size_t(0), j), expectedAttributePeople[j].size()));
        }
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedPeoplePerAttribute),
                             core::CContainerPrinter::print(peoplePerAttribute));

        const TSizeSizePrFeatureDataPrVec& personAttributeCounts =
            extractNonZeroAttributeCounts(featureData);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedAttributeCounts),
                             core::CContainerPrinter::print(personAttributeCounts));

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        CPPUNIT_ASSERT(attributeIndicator.empty());

        const TSizeSizePrFeatureDataPrVec& bucketAttributesPerPerson =
            extractBucketAttributesPerPerson(featureData);
        CPPUNIT_ASSERT(bucketAttributesPerPerson.empty());

        dataGatherer.timeNow(time + bucketLength);
    }

    TStrVec categories = allCategories();
    TSizeVec attributeIds;
    for (std::size_t i = 0u; i < categories.size(); ++i) {
        std::size_t cid;
        CPPUNIT_ASSERT(dataGatherer.attributeId(categories[i], cid));
        attributeIds.push_back(cid);
        CPPUNIT_ASSERT_EQUAL(expectedAttributeOrder[categories[i]], cid);
    }
    LOG_DEBUG(<< "attribute ids = " << core::CContainerPrinter::print(attributeIds));
    LOG_DEBUG(<< "expected attribute ids = "
              << core::CContainerPrinter::print(expectedAttributeOrder));

    TStrVec people = allPeople();
    TSizeVec peopleIds;
    for (std::size_t i = 0u; i < people.size(); ++i) {
        std::size_t pid;
        CPPUNIT_ASSERT(dataGatherer.personId(people[i], pid));
        peopleIds.push_back(pid);
        CPPUNIT_ASSERT_EQUAL(expectedPeopleOrder[people[i]], pid);
    }
    LOG_DEBUG(<< "people ids = " << core::CContainerPrinter::print(peopleIds));
    LOG_DEBUG(<< "expected people ids = "
              << core::CContainerPrinter::print(expectedPeopleOrder));
}

void CEventRatePopulationDataGathererTest::testAttributeIndicator() {
    LOG_DEBUG(<< "*** "
                 "CEventRatePopulationDataGathererTest::testAttributeIndicator "
                 "***");

    // We check that we correctly sample the (attribute, person)
    // indicator.

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 20u;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationIndicatorOfBucketPersonAndAttribute);
    SModelParams params(bucketLength);
    CDataGatherer dataGatherer(model_t::E_PopulationEventRate, model_t::E_None,
                               params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
                               false, searchKey, features, startTime, 0);

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeSizePrUInt64Map expectedAttributeIndicator;
        for (std::size_t j = 0u; j < messages.size(); ++j) {
            addArrival(messages[j].s_Time, messages[j].s_Person,
                       messages[j].s_Attribute, dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(messages[j].s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(messages[j].s_Person, pid);

            expectedAttributeIndicator[std::make_pair(pid, cid)] = 1;
        }

        CPPUNIT_ASSERT(dataGatherer.dataAvailable(time));
        CPPUNIT_ASSERT(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        CPPUNIT_ASSERT(peoplePerAttribute.empty());

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedAttributeIndicator),
                             core::CContainerPrinter::print(attributeIndicator));

        const TSizeSizePrFeatureDataPrVec& bucketAttributesPerPerson =
            extractBucketAttributesPerPerson(featureData);
        CPPUNIT_ASSERT(bucketAttributesPerPerson.empty());

        dataGatherer.timeNow(time + bucketLength);
    }
}

void CEventRatePopulationDataGathererTest::testUniqueValueCounts() {
    LOG_DEBUG(<< "*** "
                 "CEventRatePopulationDataGathererTest::"
                 "testUniqueAttributeCounts ***");

    // We check that we correctly sample the unique counts
    // of values per person.

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 20u;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationUniqueCountByBucketPersonAndAttribute);
    SModelParams params(bucketLength);
    CDataGatherer dataGatherer(model_t::E_PopulationEventRate, model_t::E_None,
                               params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, "value", TStrVec(),
                               false, searchKey, features, startTime, 0);

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeSizePrUInt64Map expectedUniqueCounts;

        TSizeSizeSetMap bucketPeopleCategories;
        for (std::size_t j = 0u; j < messages.size(); ++j) {
            std::ostringstream ss;
            ss << "thing"
               << "_" << time << "_" << i;
            std::string value(ss.str());
            addArrival(messages[j].s_Time, messages[j].s_Person, messages[j].s_Attribute,
                       value, dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(messages[j].s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(messages[j].s_Person, pid);

            expectedUniqueCounts[std::make_pair(pid, cid)] = 1;

            bucketPeopleCategories[pid].insert(cid);
        }

        CPPUNIT_ASSERT(dataGatherer.dataAvailable(time));
        CPPUNIT_ASSERT(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        CPPUNIT_ASSERT(peoplePerAttribute.empty());

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        CPPUNIT_ASSERT(attributeIndicator.empty());

        const TSizeSizePrFeatureDataPrVec& bucketAttributesPerPerson =
            extractBucketAttributesPerPerson(featureData);

        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedUniqueCounts),
                             core::CContainerPrinter::print(bucketAttributesPerPerson));

        dataGatherer.timeNow(time + bucketLength);
    }
}

void CEventRatePopulationDataGathererTest::testCompressedLength() {
    LOG_DEBUG(<< "*** "
                 "CEventRatePopulationDataGathererTest::testCompressedLength "
                 "***");

    // We check that we correctly sample the compressed length of unique
    // values per person.

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 20u;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationInfoContentByBucketPersonAndAttribute);
    SModelParams params(bucketLength);
    CDataGatherer dataGatherer(model_t::E_PopulationEventRate, model_t::E_None,
                               params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, "value", TStrVec(),
                               false, searchKey, features, startTime, 0);

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeStrSetMap bucketPeopleCategories;
        for (std::size_t j = 0u; j < messages.size(); ++j) {
            addArrival(messages[j].s_Time, messages[j].s_Person, "attribute",
                       messages[j].s_Attribute, dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(messages[j].s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(messages[j].s_Person, pid);

            bucketPeopleCategories[pid].insert(messages[j].s_Attribute);
        }

        CPPUNIT_ASSERT(dataGatherer.dataAvailable(time));
        CPPUNIT_ASSERT(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        CPPUNIT_ASSERT(peoplePerAttribute.empty());

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        CPPUNIT_ASSERT(attributeIndicator.empty());

        const TSizeSizePrFeatureDataPrVec& bucketCompressedLengthPerPerson =
            extractCompressedLengthPerPerson(featureData);
        CPPUNIT_ASSERT_EQUAL(bucketPeopleCategories.size(),
                             bucketCompressedLengthPerPerson.size());

        TSizeSizePrUInt64Map expectedBucketCompressedLengthPerPerson;
        for (TSizeStrSetMapItr iter = bucketPeopleCategories.begin();
             iter != bucketPeopleCategories.end(); ++iter) {
            TSizeSizePr key(iter->first, 0);
            const TStrSet& uniqueValues = iter->second;

            core::CCompressUtils compressor(false);
            CPPUNIT_ASSERT_EQUAL(
                uniqueValues.size(),
                static_cast<size_t>(std::count_if(
                    uniqueValues.begin(), uniqueValues.end(),
                    boost::bind(&core::CCompressUtils::addString, &compressor, _1))));
            size_t length(0);
            CPPUNIT_ASSERT(compressor.compressedLength(true, length));
            expectedBucketCompressedLengthPerPerson[key] = length;
        }
        LOG_DEBUG(<< "Time " << time << " bucketCompressedLengthPerPerson "
                  << core::CContainerPrinter::print(bucketCompressedLengthPerPerson));
        CPPUNIT_ASSERT_EQUAL(expectedBucketCompressedLengthPerPerson.size(),
                             bucketCompressedLengthPerPerson.size());
        for (TSizeSizePrFeatureDataPrVec::const_iterator j =
                 bucketCompressedLengthPerPerson.begin();
             j != bucketCompressedLengthPerPerson.end(); ++j) {
            double expectedLength = expectedBucketCompressedLengthPerPerson[j->first];
            double actual = j->second.s_Count;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedLength, actual, expectedLength * 0.1);
        }

        dataGatherer.timeNow(time + bucketLength);
    }
}

void CEventRatePopulationDataGathererTest::testRemovePeople() {
    LOG_DEBUG(
        << "*** CEventRatePopulationDataGathererTest::testRemovePeople ***");

    using TStrSizeMap = std::map<std::string, std::size_t>;
    using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    const std::size_t numberBuckets = 50u;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_PopulationEventRate, model_t::E_None,
                           params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
                           false, searchKey, features, startTime, 0);
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < numberBuckets; ++i, bucketStart += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, bucketStart, bucketLength, messages);
        for (std::size_t j = 0u; j < messages.size(); ++j) {
            addArrival(messages[j].s_Time, messages[j].s_Person,
                       messages[j].s_Attribute, gatherer, m_ResourceMonitor);
        }
    }

    // Remove people 1 and 4.
    TSizeVec peopleToRemove;
    peopleToRemove.push_back(1u);
    peopleToRemove.push_back(4u);

    std::size_t numberPeople = gatherer.numberActivePeople();
    LOG_DEBUG(<< "numberPeople = " << numberPeople);
    CPPUNIT_ASSERT_EQUAL(numberPeople, gatherer.numberOverFieldValues());
    TStrVec expectedPersonNames;
    TSizeVec expectedPersonIds;
    for (std::size_t i = 0u; i < numberPeople; ++i) {
        if (!std::binary_search(peopleToRemove.begin(), peopleToRemove.end(), i)) {
            expectedPersonNames.push_back(gatherer.personName(i));
            expectedPersonIds.push_back(i);
        } else {
            LOG_DEBUG(<< "Removing " << gatherer.personName(i));
        }
    }

    TStrSizeMap expectedNonZeroCounts;
    {
        TSizeUInt64PrVec nonZeroCounts;
        gatherer.personNonZeroCounts(bucketStart - bucketLength, nonZeroCounts);
        for (std::size_t i = 0u; i < nonZeroCounts.size(); ++i) {
            if (!std::binary_search(peopleToRemove.begin(), peopleToRemove.end(),
                                    nonZeroCounts[i].first)) {
                const std::string& name = gatherer.personName(nonZeroCounts[i].first);
                expectedNonZeroCounts[name] =
                    static_cast<size_t>(nonZeroCounts[i].second);
            }
        }
    }
    LOG_DEBUG(<< "expectedNonZeroCounts = "
              << core::CContainerPrinter::print(expectedNonZeroCounts));

    std::string expectedFeatureData;
    {
        LOG_DEBUG(<< "Expected");
        TStrFeatureDataPrVec expected;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart - bucketLength, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j) {
                if (!std::binary_search(peopleToRemove.begin(),
                                        peopleToRemove.end(), data[j].first.first)) {
                    std::string key = model_t::print(featureData[i].first) + " " +
                                      gatherer.personName(data[j].first.first) + " " +
                                      gatherer.attributeName(data[j].first.second);
                    expected.push_back(TStrFeatureDataPr(key, data[j].second));
                    LOG_DEBUG(<< "  " << key << " = " << data[j].second.s_Count);
                }
            }
        }
        expectedFeatureData = core::CContainerPrinter::print(expected);
    }

    gatherer.recyclePeople(peopleToRemove);

    CPPUNIT_ASSERT_EQUAL(numberPeople - peopleToRemove.size(),
                         gatherer.numberActivePeople());
    for (std::size_t i = 0u; i < expectedPersonNames.size(); ++i) {
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId(expectedPersonNames[i], pid));
        CPPUNIT_ASSERT_EQUAL(expectedPersonIds[i], pid);
    }

    TStrSizeMap actualNonZeroCounts;
    TSizeUInt64PrVec nonZeroCounts;
    gatherer.personNonZeroCounts(bucketStart - bucketLength, nonZeroCounts);
    for (std::size_t i = 0u; i < nonZeroCounts.size(); ++i) {
        const std::string& name = gatherer.personName(nonZeroCounts[i].first);
        actualNonZeroCounts[name] = static_cast<size_t>(nonZeroCounts[i].second);
    }
    LOG_DEBUG(<< "actualNonZeroCounts = "
              << core::CContainerPrinter::print(actualNonZeroCounts));

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedNonZeroCounts),
                         core::CContainerPrinter::print(actualNonZeroCounts));

    std::string actualFeatureData;
    {
        LOG_DEBUG(<< "Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart - bucketLength, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j) {
                std::string key = model_t::print(featureData[i].first) + " " +
                                  gatherer.personName(data[j].first.first) + " " +
                                  gatherer.attributeName(data[j].first.second);
                actual.push_back(TStrFeatureDataPr(key, data[j].second));
                LOG_DEBUG(<< "  " << key << " = " << data[j].second.s_Count);
            }
        }
        actualFeatureData = core::CContainerPrinter::print(actual);
    }

    CPPUNIT_ASSERT_EQUAL(expectedFeatureData, actualFeatureData);
}

void CEventRatePopulationDataGathererTest::testRemoveAttributes() {
    LOG_DEBUG(<< "*** "
                 "CEventRatePopulationDataGathererTest::testRemoveAttributes "
                 "***");

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    SModelParams params(bucketLength);
    CDataGatherer gatherer(model_t::E_PopulationEventRate, model_t::E_None,
                           params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
                           false, searchKey, features, startTime, 0);

    TMessageVec messages;
    generateTestMessages(rng, startTime, bucketLength, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i) {
        addArrival(messages[i].s_Time, messages[i].s_Person,
                   messages[i].s_Attribute, gatherer, m_ResourceMonitor);
    }

    // Remove attributes 1, 2, 3 and 15.
    TSizeVec attributesToRemove;
    attributesToRemove.push_back(1u);
    attributesToRemove.push_back(2u);
    attributesToRemove.push_back(3u);
    attributesToRemove.push_back(15u);

    std::size_t numberAttributes = gatherer.numberActiveAttributes();
    LOG_DEBUG(<< "numberAttributes = " << numberAttributes);
    CPPUNIT_ASSERT_EQUAL(numberAttributes, gatherer.numberByFieldValues());
    TStrVec expectedAttributeNames;
    TSizeVec expectedAttributeIds;
    for (std::size_t i = 0u; i < numberAttributes; ++i) {
        if (!std::binary_search(attributesToRemove.begin(), attributesToRemove.end(), i)) {
            expectedAttributeNames.push_back(gatherer.attributeName(i));
            expectedAttributeIds.push_back(i);
        } else {
            LOG_DEBUG(<< "Removing " << gatherer.attributeName(i));
        }
    }

    std::string expectedFeatureData;
    {
        LOG_DEBUG(<< "Expected");
        TStrFeatureDataPrVec expected;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j) {
                if (!std::binary_search(attributesToRemove.begin(),
                                        attributesToRemove.end(), data[j].first.second)) {
                    std::string key = model_t::print(featureData[i].first) + " " +
                                      gatherer.personName(data[j].first.first) + " " +
                                      gatherer.attributeName(data[j].first.second);
                    expected.push_back(TStrFeatureDataPr(key, data[j].second));
                    LOG_DEBUG(<< "  " << key << " = " << data[j].second.s_Count);
                }
            }
        }
        expectedFeatureData = core::CContainerPrinter::print(expected);
    }

    gatherer.recycleAttributes(attributesToRemove);

    CPPUNIT_ASSERT_EQUAL(numberAttributes - attributesToRemove.size(),
                         gatherer.numberActiveAttributes());
    for (std::size_t i = 0u; i < expectedAttributeNames.size(); ++i) {
        std::size_t cid;
        CPPUNIT_ASSERT(gatherer.attributeId(expectedAttributeNames[i], cid));
        CPPUNIT_ASSERT_EQUAL(expectedAttributeIds[i], cid);
    }

    std::string actualFeatureData;
    {
        LOG_DEBUG(<< "Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j) {
                std::string key = model_t::print(featureData[i].first) + " " +
                                  gatherer.personName(data[j].first.first) + " " +
                                  gatherer.attributeName(data[j].first.second);
                actual.push_back(TStrFeatureDataPr(key, data[j].second));
                LOG_DEBUG(<< "  " << key << " = " << data[j].second.s_Count);
            }
        }
        actualFeatureData = core::CContainerPrinter::print(actual);
    }

    CPPUNIT_ASSERT_EQUAL(expectedFeatureData, actualFeatureData);
}

namespace {
bool isSpace(const char x) {
    return x == ' ' || x == '\t';
}
}

void CEventRatePopulationDataGathererTest::testPersistence() {
    LOG_DEBUG(
        << "*** CEventRatePopulationDataGathererTest::testPersistence ***");

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;

    {
        test::CRandomNumbers rng;

        CDataGatherer::TFeatureVec features;
        features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
        features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
        SModelParams params(bucketLength);
        CDataGatherer origDataGatherer(
            model_t::E_PopulationEventRate, model_t::E_None, params,
            EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
            EMPTY_STRING, TStrVec(), false, searchKey, features, startTime, 0);

        TMessageVec messages;
        generateTestMessages(rng, startTime, bucketLength, messages);

        for (std::size_t i = 0u; i < messages.size(); ++i) {
            addArrival(messages[i].s_Time, messages[i].s_Person,
                       messages[i].s_Attribute, origDataGatherer, m_ResourceMonitor);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origDataGatherer.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "origXml = " << origXml);
        LOG_DEBUG(<< "length = " << origXml.length() << ", # tabs "
                  << std::count_if(origXml.begin(), origXml.end(), isSpace));

        // Restore the XML into a new data gatherer
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        CDataGatherer restoredDataGatherer(
            model_t::E_PopulationEventRate, model_t::E_None, params,
            EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
            EMPTY_STRING, EMPTY_STRING, TStrVec(), false, searchKey, traverser);

        // The XML representation of the new data gatherer should be the same as the
        // original
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredDataGatherer.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
    }
    {
        // Check feature data for model_t::E_UniqueValues
        const std::size_t numberBuckets = 20u;

        test::CRandomNumbers rng;
        CDataGatherer::TFeatureVec features;
        features.push_back(model_t::E_PopulationInfoContentByBucketPersonAndAttribute);
        SModelParams params(bucketLength);
        CDataGatherer dataGatherer(model_t::E_PopulationEventRate, model_t::E_None,
                                   params, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                   EMPTY_STRING, EMPTY_STRING, "value", TStrVec(),
                                   false, searchKey, features, startTime, 0);

        core_t::TTime time = startTime;
        for (std::size_t i = 0u; i < numberBuckets; ++i, time += bucketLength) {
            TMessageVec messages;
            generateTestMessages(rng, time, bucketLength, messages);

            for (std::size_t j = 0u; j < messages.size(); ++j) {
                addArrival(messages[j].s_Time, messages[j].s_Person, "attribute",
                           messages[j].s_Attribute, dataGatherer, m_ResourceMonitor);

                std::size_t cid;
                dataGatherer.attributeId(messages[j].s_Attribute, cid);

                std::size_t pid;
                dataGatherer.personId(messages[j].s_Person, pid);
            }

            CPPUNIT_ASSERT(dataGatherer.dataAvailable(time));
            CPPUNIT_ASSERT(!dataGatherer.dataAvailable(time - 1));

            dataGatherer.timeNow(time + bucketLength);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            dataGatherer.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "origXml = " << origXml);
        LOG_DEBUG(<< "length = " << origXml.length() << ", # tabs "
                  << std::count_if(origXml.begin(), origXml.end(), isSpace));

        // Restore the XML into a new data gatherer
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        CDataGatherer restoredDataGatherer(
            model_t::E_PopulationEventRate, model_t::E_None, params,
            EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
            EMPTY_STRING, EMPTY_STRING, TStrVec(), false, searchKey, traverser);

        // The XML representation of the new data gatherer should be the same as the
        // original
        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredDataGatherer.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        CPPUNIT_ASSERT_EQUAL(origXml, newXml);
        CPPUNIT_ASSERT_EQUAL(dataGatherer.checksum(), restoredDataGatherer.checksum());
    }
}

CppUnit::Test* CEventRatePopulationDataGathererTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CEventRatePopulationDataGathererTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testAttributeCounts",
        &CEventRatePopulationDataGathererTest::testAttributeCounts));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testAttributeIndicator",
        &CEventRatePopulationDataGathererTest::testAttributeIndicator));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testUniqueValueCounts",
        &CEventRatePopulationDataGathererTest::testUniqueValueCounts));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testCompressedLength",
        &CEventRatePopulationDataGathererTest::testCompressedLength));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testRemovePeople",
        &CEventRatePopulationDataGathererTest::testRemovePeople));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testRemoveAttributes",
        &CEventRatePopulationDataGathererTest::testRemoveAttributes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRatePopulationDataGathererTest>(
        "CEventRatePopulationDataGathererTest::testPersistence",
        &CEventRatePopulationDataGathererTest::testPersistence));

    return suiteOfTests;
}
