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
#include <core/CompressUtils.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CEventRateBucketGatherer.h>
#include <model/CResourceMonitor.h>
#include <model/SModelParams.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "ModelTestHelpers.h"

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <set>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CEventRatePopulationDataGathererTest)

using namespace ml;
using namespace model;

namespace {

struct SMessage {
    SMessage(core_t::TTime time, std::string attribute, std::string person)
        : s_Time(time), s_Attribute(std::move(attribute)),
          s_Person(std::move(person)) {}

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
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrUInt64Map = std::map<TSizeSizePr, std::uint64_t>;
using TSizeSet = std::set<std::size_t>;
using TSizeSizeSetMap = std::map<std::size_t, TSizeSet>;
using TStrSet = std::set<std::string>;
using TSizeStrSetMap = std::map<std::size_t, TStrSet>;
using TFeatureData = SEventRateFeatureData;
using TStrFeatureDataPr = std::pair<std::string, TFeatureData>;
using TStrFeatureDataPrVec = std::vector<TStrFeatureDataPr>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, TFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TFeatureSizeSizePrFeatureDataPrVecPr =
    std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;

TStrVec allCategories() {
    constexpr std::size_t numberCategories = 30;
    TStrVec categories;
    for (std::size_t i = 0; i < numberCategories; ++i) {
        categories.push_back("c" + boost::lexical_cast<std::string>(i));
    }
    return categories;
}

TStrVec allPeople() {
    constexpr std::size_t numberPeople = 5;
    TStrVec people;
    for (std::size_t i = 0; i < numberPeople; ++i) {
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

    const TDoubleVec rates{1.0, 0.3, 10.1, 25.0, 105.0};

    TSizeVec bucketCounts;
    for (std::size_t j = 0; j < categories.size(); ++j) {
        double const rate = rates[j % rates.size()];
        TUIntVec sample;
        rng.generatePoissonSamples(rate, 1, sample);
        bucketCounts.push_back(sample[0]);
    }

    TDoubleVec personRange;
    rng.generateUniformSamples(0.0, static_cast<double>(people.size()) - 1e-3, 2U, personRange);
    std::ranges::sort(personRange);
    auto const a = static_cast<std::size_t>(personRange[0]);
    std::size_t const b = static_cast<std::size_t>(personRange[1]) + 1;
    TSizeVec bucketPeople;
    for (std::size_t i = a; i < b; ++i) {
        bucketPeople.push_back(i);
    }
    LOG_DEBUG(<< "bucketPeople = " << bucketPeople);

    for (std::size_t i = 0; i < categories.size(); ++i) {
        TDoubleVec offsets;
        rng.generateUniformSamples(0.0, static_cast<double>(bucketLength) - 1.0,
                                   bucketCounts[i], offsets);

        for (std::size_t j = 0; j < offsets.size(); ++j) {
            messages.emplace_back(time + static_cast<core_t::TTime>(offsets[j]),
                                  categories[i],
                                  people[bucketPeople[j % bucketPeople.size()]]);
        }
    }

    std::ranges::sort(messages, [](const SMessage& lhs, const SMessage& rhs) {
        return lhs.s_Time < rhs.s_Time;
    });
    LOG_DEBUG(<< "Generated " << messages.size() << " messages");
}

const TSizeSizePrFeatureDataPrVec&
extract(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData, model_t::EFeature feature) {
    for (const auto& i : featureData) {
        if (i.first == feature) {
            return i.second;
        }
    }

    static constexpr TSizeSizePrFeatureDataPrVec EMPTY;
    return EMPTY;
}

const TSizeSizePrFeatureDataPrVec&
extractPeoplePerAttribute(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationUniquePersonCountByAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractNonZeroAttributeCounts(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationCountByBucketPersonAndAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractAttributeIndicator(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationIndicatorOfBucketPersonAndAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractBucketAttributesPerPerson(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
    return extract(featureData, model_t::E_PopulationUniqueCountByBucketPersonAndAttribute);
}

const TSizeSizePrFeatureDataPrVec&
extractCompressedLengthPerPerson(const TFeatureSizeSizePrFeatureDataPrVecPrVec& featureData) {
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

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testAttributeCounts, CTestFixture) {
    // We check that we correctly sample the unique people per
    // attribute and (attribute, person) pair counts.

    using TStrSizeMap = std::map<std::string, std::size_t>;

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    constexpr std::size_t numberBuckets = 20;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    SModelParams const params(bucketLength);
    CDataGatherer dataGatherer = CDataGathererBuilder(model_t::E_PopulationEventRate,
                                                      features, params, searchKey, startTime)
                                     .build();
    BOOST_TEST_REQUIRE(dataGatherer.isPopulation());

    BOOST_REQUIRE_EQUAL(startTime, dataGatherer.currentBucketStartTime());
    BOOST_REQUIRE_EQUAL(bucketLength, dataGatherer.bucketLength());

    TSizeSizeSetMap expectedAttributePeople;
    std::size_t attributeOrder = 0;
    TStrSizeMap expectedAttributeOrder;
    std::size_t personOrder = 0;
    TStrSizeMap expectedPeopleOrder;
    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeSizePrUInt64Map expectedAttributeCounts;
        for (auto& message : messages) {
            addArrival(message.s_Time, message.s_Person, message.s_Attribute,
                       dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(message.s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(message.s_Person, pid);

            ++expectedAttributeCounts[std::make_pair(pid, cid)];
            expectedAttributePeople[cid].insert(pid);
            if (expectedAttributeOrder
                    .insert(TStrSizeMap::value_type(message.s_Attribute, attributeOrder))
                    .second) {
                ++attributeOrder;
            }
            if (expectedPeopleOrder
                    .insert(TStrSizeMap::value_type(message.s_Person, personOrder))
                    .second) {
                ++personOrder;
            }
        }

        BOOST_TEST_REQUIRE(dataGatherer.dataAvailable(time));
        BOOST_TEST_REQUIRE(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        BOOST_REQUIRE_EQUAL(expectedAttributePeople.size(), peoplePerAttribute.size());
        TSizeSizePrFeatureDataPrVec expectedPeoplePerAttribute;
        for (std::size_t j = 0; j < peoplePerAttribute.size(); ++j) {
            expectedPeoplePerAttribute.emplace_back(std::make_pair(
                std::make_pair(static_cast<size_t>(0), j), expectedAttributePeople[j].size()));
        }
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedPeoplePerAttribute),
                            core::CContainerPrinter::print(peoplePerAttribute));

        const TSizeSizePrFeatureDataPrVec& personAttributeCounts =
            extractNonZeroAttributeCounts(featureData);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedAttributeCounts),
                            core::CContainerPrinter::print(personAttributeCounts));

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        BOOST_TEST_REQUIRE(attributeIndicator.empty());

        const TSizeSizePrFeatureDataPrVec& bucketAttributesPerPerson =
            extractBucketAttributesPerPerson(featureData);
        BOOST_TEST_REQUIRE(bucketAttributesPerPerson.empty());

        dataGatherer.timeNow(time + bucketLength);
    }

    TStrVec const categories = allCategories();
    TSizeVec attributeIds;
    for (const auto& categorie : categories) {
        std::size_t cid;
        BOOST_TEST_REQUIRE(dataGatherer.attributeId(categorie, cid));
        attributeIds.push_back(cid);
        BOOST_REQUIRE_EQUAL(expectedAttributeOrder[categorie], cid);
    }
    LOG_DEBUG(<< "attribute ids = " << attributeIds);
    LOG_DEBUG(<< "expected attribute ids = " << expectedAttributeOrder);

    TStrVec const people = allPeople();
    TSizeVec peopleIds;
    for (const auto& i : people) {
        std::size_t pid;
        BOOST_TEST_REQUIRE(dataGatherer.personId(i, pid));
        peopleIds.push_back(pid);
        BOOST_REQUIRE_EQUAL(expectedPeopleOrder[i], pid);
    }
    LOG_DEBUG(<< "people ids = " << peopleIds);
    LOG_DEBUG(<< "expected people ids = " << expectedPeopleOrder);
}

BOOST_FIXTURE_TEST_CASE(testAttributeIndicator, CTestFixture) {
    // We check that we correctly sample the (attribute, person)
    // indicator.

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    constexpr std::size_t numberBuckets = 20;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationIndicatorOfBucketPersonAndAttribute);
    SModelParams const params(bucketLength);
    CDataGatherer dataGatherer = CDataGathererBuilder(model_t::E_PopulationEventRate,
                                                      features, params, searchKey, startTime)
                                     .build();
    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeSizePrUInt64Map expectedAttributeIndicator;
        for (auto& message : messages) {
            addArrival(message.s_Time, message.s_Person, message.s_Attribute,
                       dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(message.s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(message.s_Person, pid);

            expectedAttributeIndicator[std::make_pair(pid, cid)] = 1;
        }

        BOOST_TEST_REQUIRE(dataGatherer.dataAvailable(time));
        BOOST_TEST_REQUIRE(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        BOOST_TEST_REQUIRE(peoplePerAttribute.empty());

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedAttributeIndicator),
                            core::CContainerPrinter::print(attributeIndicator));

        const TSizeSizePrFeatureDataPrVec& bucketAttributesPerPerson =
            extractBucketAttributesPerPerson(featureData);
        BOOST_TEST_REQUIRE(bucketAttributesPerPerson.empty());

        dataGatherer.timeNow(time + bucketLength);
    }
}

BOOST_FIXTURE_TEST_CASE(testUniqueValueCounts, CTestFixture) {
    // We check that we correctly sample the unique counts
    // of values per person.

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    constexpr std::size_t numberBuckets = 20;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationUniqueCountByBucketPersonAndAttribute);
    SModelParams const params(bucketLength);
    CDataGatherer dataGatherer = CDataGathererBuilder(model_t::E_PopulationEventRate,
                                                      features, params, searchKey, startTime)
                                     .valueFieldName("value")
                                     .build();
    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeSizePrUInt64Map expectedUniqueCounts;

        TSizeSizeSetMap bucketPeopleCategories;
        for (auto& message : messages) {
            std::ostringstream ss;
            ss << "thing"
               << "_" << time << "_" << i;
            std::string const value(ss.str());
            addArrival(message.s_Time, message.s_Person, message.s_Attribute,
                       value, dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(message.s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(message.s_Person, pid);

            expectedUniqueCounts[std::make_pair(pid, cid)] = 1;

            bucketPeopleCategories[pid].insert(cid);
        }

        BOOST_TEST_REQUIRE(dataGatherer.dataAvailable(time));
        BOOST_TEST_REQUIRE(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        BOOST_TEST_REQUIRE(peoplePerAttribute.empty());

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        BOOST_TEST_REQUIRE(attributeIndicator.empty());

        const TSizeSizePrFeatureDataPrVec& bucketAttributesPerPerson =
            extractBucketAttributesPerPerson(featureData);

        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedUniqueCounts),
                            core::CContainerPrinter::print(bucketAttributesPerPerson));

        dataGatherer.timeNow(time + bucketLength);
    }
}

BOOST_FIXTURE_TEST_CASE(testCompressedLength, CTestFixture) {
    // We check that we correctly sample the compressed length of unique
    // values per person.

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    constexpr std::size_t numberBuckets = 20;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationInfoContentByBucketPersonAndAttribute);
    SModelParams const params(bucketLength);
    CDataGatherer dataGatherer = CDataGathererBuilder(model_t::E_PopulationEventRate,
                                                      features, params, searchKey, startTime)
                                     .valueFieldName("value")
                                     .build();
    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < numberBuckets; ++i, time += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, time, bucketLength, messages);

        TSizeStrSetMap bucketPeopleCategories;
        for (auto& message : messages) {
            addArrival(message.s_Time, message.s_Person, "attribute",
                       message.s_Attribute, dataGatherer, m_ResourceMonitor);

            std::size_t cid;
            dataGatherer.attributeId(message.s_Attribute, cid);

            std::size_t pid;
            dataGatherer.personId(message.s_Person, pid);

            bucketPeopleCategories[pid].insert(message.s_Attribute);
        }

        BOOST_TEST_REQUIRE(dataGatherer.dataAvailable(time));
        BOOST_TEST_REQUIRE(!dataGatherer.dataAvailable(time - 1));

        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        dataGatherer.featureData(time, bucketLength, featureData);

        const TSizeSizePrFeatureDataPrVec& peoplePerAttribute =
            extractPeoplePerAttribute(featureData);
        BOOST_TEST_REQUIRE(peoplePerAttribute.empty());

        const TSizeSizePrFeatureDataPrVec& attributeIndicator =
            extractAttributeIndicator(featureData);
        BOOST_TEST_REQUIRE(attributeIndicator.empty());

        const TSizeSizePrFeatureDataPrVec& bucketCompressedLengthPerPerson =
            extractCompressedLengthPerPerson(featureData);
        BOOST_REQUIRE_EQUAL(bucketPeopleCategories.size(),
                            bucketCompressedLengthPerPerson.size());

        TSizeSizePrUInt64Map expectedBucketCompressedLengthPerPerson;
        for (auto& bucketPeopleCategorie : bucketPeopleCategories) {
            TSizeSizePr const key(bucketPeopleCategorie.first, 0);
            const TStrSet& uniqueValues = bucketPeopleCategorie.second;

            core::CDeflator compressor(false);
BOOST_REQUIRE_EQUAL(
                        uniqueValues.size(),
                        static_cast<size_t>(std::ranges::count_if(
                            uniqueValues, std::bind_front(&core::CCompressUtil::addString, &compressor))));
            std::size_t length(0);
            BOOST_TEST_REQUIRE(compressor.length(true, length));
            expectedBucketCompressedLengthPerPerson[key] = length;
        }
        LOG_DEBUG(<< "Time " << time << " bucketCompressedLengthPerPerson "
                  << bucketCompressedLengthPerPerson);
        BOOST_REQUIRE_EQUAL(expectedBucketCompressedLengthPerPerson.size(),
                            bucketCompressedLengthPerPerson.size());
        for (const auto& j : bucketCompressedLengthPerPerson) {
            auto const expectedLength =
                static_cast<double>(expectedBucketCompressedLengthPerPerson[j.first]);
            auto const actual = static_cast<double>(j.second.s_Count);
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedLength, actual, expectedLength * 0.1);
        }

        dataGatherer.timeNow(time + bucketLength);
    }
}

BOOST_FIXTURE_TEST_CASE(testRemovePeople, CTestFixture) {
    using TStrSizeMap = std::map<std::string, std::size_t>;
    using TSizeUInt64Pr = std::pair<std::size_t, std::uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    constexpr std::size_t numberBuckets = 50;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    SModelParams const params(bucketLength);
    CDataGatherer gatherer = CDataGathererBuilder(model_t::E_PopulationEventRate,
                                                  features, params, searchKey, startTime)
                                 .build();
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < numberBuckets; ++i, bucketStart += bucketLength) {
        TMessageVec messages;
        generateTestMessages(rng, bucketStart, bucketLength, messages);
        for (auto& message : messages) {
            addArrival(message.s_Time, message.s_Person, message.s_Attribute,
                       gatherer, m_ResourceMonitor);
        }
    }

    // Remove people 1 and 4.
    TSizeVec peopleToRemove;
    peopleToRemove.push_back(1U);
    peopleToRemove.push_back(4U);

    std::size_t const numberPeople = gatherer.numberActivePeople();
    LOG_DEBUG(<< "numberPeople = " << numberPeople);
    BOOST_REQUIRE_EQUAL(numberPeople, gatherer.numberOverFieldValues());
    TStrVec expectedPersonNames;
    TSizeVec expectedPersonIds;
    for (std::size_t i = 0; i < numberPeople; ++i) {
        if (!std::ranges::binary_search(peopleToRemove, i)) {
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
        for (auto& nonZeroCount : nonZeroCounts) {
            if (!std::ranges::binary_search(peopleToRemove, nonZeroCount.first)) {
                const std::string& name = gatherer.personName(nonZeroCount.first);
                expectedNonZeroCounts[name] = static_cast<size_t>(nonZeroCount.second);
            }
        }
    }
    LOG_DEBUG(<< "expectedNonZeroCounts = " << expectedNonZeroCounts);

    std::string expectedFeatureData;
    {
        LOG_DEBUG(<< "Expected");
        TStrFeatureDataPrVec expected;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart - bucketLength, bucketLength, featureData);
        for (auto& i : featureData) {
            const TSizeSizePrFeatureDataPrVec& data = i.second;
            for (const auto& j : data) {
                if (!std::ranges::binary_search(peopleToRemove, j.first.first)) {
                    std::string const key = model_t::print(i.first) + " " +
                                            gatherer.personName(j.first.first) + " " +
                                            gatherer.attributeName(j.first.second);
                    expected.emplace_back(key, j.second);
                    LOG_DEBUG(<< "  " << key << " = " << j.second.s_Count);
                }
            }
        }
        expectedFeatureData = core::CContainerPrinter::print(expected);
    }

    gatherer.recyclePeople(peopleToRemove);

    BOOST_REQUIRE_EQUAL(numberPeople - peopleToRemove.size(), gatherer.numberActivePeople());
    for (std::size_t i = 0; i < expectedPersonNames.size(); ++i) {
        std::size_t pid;
        BOOST_TEST_REQUIRE(gatherer.personId(expectedPersonNames[i], pid));
        BOOST_REQUIRE_EQUAL(expectedPersonIds[i], pid);
    }

    TStrSizeMap actualNonZeroCounts;
    TSizeUInt64PrVec nonZeroCounts;
    gatherer.personNonZeroCounts(bucketStart - bucketLength, nonZeroCounts);
    for (auto& nonZeroCount : nonZeroCounts) {
        const std::string& name = gatherer.personName(nonZeroCount.first);
        actualNonZeroCounts[name] = static_cast<size_t>(nonZeroCount.second);
    }
    LOG_DEBUG(<< "actualNonZeroCounts = " << actualNonZeroCounts);

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedNonZeroCounts),
                        core::CContainerPrinter::print(actualNonZeroCounts));

    std::string actualFeatureData;
    {
        LOG_DEBUG(<< "Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart - bucketLength, bucketLength, featureData);
        for (auto& i : featureData) {
            const TSizeSizePrFeatureDataPrVec& data = i.second;
            for (const auto& j : data) {
                std::string const key = model_t::print(i.first) + " " +
                                        gatherer.personName(j.first.first) + " " +
                                        gatherer.attributeName(j.first.second);
                actual.emplace_back(key, j.second);
                LOG_DEBUG(<< "  " << key << " = " << j.second.s_Count);
            }
        }
        actualFeatureData = core::CContainerPrinter::print(actual);
    }

    BOOST_REQUIRE_EQUAL(expectedFeatureData, actualFeatureData);
}

BOOST_FIXTURE_TEST_CASE(testRemoveAttributes, CTestFixture) {
    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;

    test::CRandomNumbers rng;

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
    features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
    SModelParams const params(bucketLength);
    CDataGatherer gatherer = CDataGathererBuilder(model_t::E_PopulationEventRate,
                                                  features, params, searchKey, startTime)
                                 .build();
    TMessageVec messages;
    generateTestMessages(rng, startTime, bucketLength, messages);

    constexpr core_t::TTime bucketStart = startTime;
    for (auto& message : messages) {
        addArrival(message.s_Time, message.s_Person, message.s_Attribute,
                   gatherer, m_ResourceMonitor);
    }

    // Remove attributes 1, 2, 3 and 15.
    TSizeVec attributesToRemove;
    attributesToRemove.push_back(1U);
    attributesToRemove.push_back(2U);
    attributesToRemove.push_back(3U);
    attributesToRemove.push_back(15U);

    std::size_t const numberAttributes = gatherer.numberActiveAttributes();
    LOG_DEBUG(<< "numberAttributes = " << numberAttributes);
    BOOST_REQUIRE_EQUAL(numberAttributes, gatherer.numberByFieldValues());
    TStrVec expectedAttributeNames;
    TSizeVec expectedAttributeIds;
    for (std::size_t i = 0; i < numberAttributes; ++i) {
        if (!std::ranges::binary_search(attributesToRemove, i)) {
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
        for (auto& i : featureData) {
            const TSizeSizePrFeatureDataPrVec& data = i.second;
            for (const auto& j : data) {
                if (!std::ranges::binary_search(attributesToRemove, j.first.second)) {
                    std::string const key = model_t::print(i.first) + " " +
                                            gatherer.personName(j.first.first) + " " +
                                            gatherer.attributeName(j.first.second);
                    expected.emplace_back(key, j.second);
                    LOG_DEBUG(<< "  " << key << " = " << j.second.s_Count);
                }
            }
        }
        expectedFeatureData = core::CContainerPrinter::print(expected);
    }

    gatherer.recycleAttributes(attributesToRemove);

    BOOST_REQUIRE_EQUAL(numberAttributes - attributesToRemove.size(),
                        gatherer.numberActiveAttributes());
    for (std::size_t i = 0; i < expectedAttributeNames.size(); ++i) {
        std::size_t cid;
        BOOST_TEST_REQUIRE(gatherer.attributeId(expectedAttributeNames[i], cid));
        BOOST_REQUIRE_EQUAL(expectedAttributeIds[i], cid);
    }

    std::string actualFeatureData;
    {
        LOG_DEBUG(<< "Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (auto& i : featureData) {
            const TSizeSizePrFeatureDataPrVec& data = i.second;
            for (const auto& j : data) {
                std::string const key = model_t::print(i.first) + " " +
                                        gatherer.personName(j.first.first) + " " +
                                        gatherer.attributeName(j.first.second);
                actual.emplace_back(key, j.second);
                LOG_DEBUG(<< "  " << key << " = " << j.second.s_Count);
            }
        }
        actualFeatureData = core::CContainerPrinter::print(actual);
    }

    BOOST_REQUIRE_EQUAL(expectedFeatureData, actualFeatureData);
}

namespace {
void testPersistDataGatherer(const CDataGatherer& origDataGatherer,
                                    const SModelParams& params) {
    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, std::bind_front(&CDataGatherer::acceptPersistInserter, &origDataGatherer));
    LOG_DEBUG(<< "origJson = " << origJson.str());

    // Restore the Json into a new data gatherer
    // The traverser expects the state json in a embedded document
    std::istringstream origJsonStrm("{\"topLevel\" : " + origJson.str() + "}");
    core::CJsonStateRestoreTraverser traverser(origJsonStrm);

    CDataGatherer restoredDataGatherer(model_t::E_PopulationEventRate,
                                       model_t::E_None, params, EMPTY_STRING,
                                       EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                       EMPTY_STRING, {}, searchKey, traverser);

    // The Json representation of the new data gatherer should be the same as the
    // original
    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, std::bind_front(&CDataGatherer::acceptPersistInserter, &restoredDataGatherer));
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}
}



BOOST_FIXTURE_TEST_CASE(testPersistence, CTestFixture) {
    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;

    {
        test::CRandomNumbers rng;

        CDataGatherer::TFeatureVec features;
        features.push_back(model_t::E_PopulationCountByBucketPersonAndAttribute);
        features.push_back(model_t::E_PopulationUniquePersonCountByAttribute);
        SModelParams const params(bucketLength);
        CDataGatherer origDataGatherer =
            CDataGathererBuilder(model_t::E_PopulationEventRate, features,
                                 params, searchKey, startTime)
                .build();
        TMessageVec messages;
        generateTestMessages(rng, startTime, bucketLength, messages);

        for (auto& message : messages) {
            addArrival(message.s_Time, message.s_Person, message.s_Attribute,
                       origDataGatherer, m_ResourceMonitor);
        }

        testPersistDataGatherer(origDataGatherer, params);
    }
    {
        // Check feature data for model_t::E_UniqueValues
        constexpr std::size_t numberBuckets = 20;

        test::CRandomNumbers rng;
        CDataGatherer::TFeatureVec features;
        features.push_back(model_t::E_PopulationInfoContentByBucketPersonAndAttribute);
        SModelParams const params(bucketLength);
        CDataGatherer dataGatherer =
            CDataGathererBuilder(model_t::E_PopulationEventRate, features,
                                 params, searchKey, startTime)
                .valueFieldName("value")
                .build();
        core_t::TTime time = startTime;
        for (std::size_t i = 0; i < numberBuckets; ++i, time += bucketLength) {
            TMessageVec messages;
            generateTestMessages(rng, time, bucketLength, messages);

            for (auto& message : messages) {
                addArrival(message.s_Time, message.s_Person, "attribute",
                           message.s_Attribute, dataGatherer, m_ResourceMonitor);

                std::size_t cid;
                dataGatherer.attributeId(message.s_Attribute, cid);

                std::size_t pid;
                dataGatherer.personId(message.s_Person, pid);
            }

            BOOST_TEST_REQUIRE(dataGatherer.dataAvailable(time));
            BOOST_TEST_REQUIRE(!dataGatherer.dataAvailable(time - 1));

            dataGatherer.timeNow(time + bucketLength);
        }

        testPersistDataGatherer(dataGatherer, params);
    }
}

BOOST_AUTO_TEST_SUITE_END()
