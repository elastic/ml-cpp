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

#include <core/CContainerPrinter.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CoreTypes.h>
#include <core/UnwrapRef.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CFeatureData.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CMetricBucketGatherer.h>
#include <model/CMetricPopulationModelFactory.h>
#include <model/CResourceMonitor.h>
#include <model/ModelTypes.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <map>
#include <string>
#include <vector>

namespace {

BOOST_AUTO_TEST_SUITE(CMetricPopulationDataGathererTest)

using namespace ml;
using namespace model;

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TStrStrPr = std::pair<std::string, std::string>;
using TStrStrPrDoubleMap = std::map<TStrStrPr, double>;
using TOptionalStr = std::optional<std::string>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, SMetricFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TFeatureSizeSizePrFeatureDataPrVecPr =
    std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;

struct SMessage {
    SMessage(const core_t::TTime& time,
             const std::string& person,
             const std::string& attribute,
             const double& value,
             const TStrVec& influences = TStrVec())
        : s_Time(time), s_Person(person), s_Attribute(attribute),
          s_Value(value), s_Influences(influences) {}

    core_t::TTime s_Time;
    std::string s_Person;
    std::string s_Attribute;
    double s_Value;
    TStrVec s_Influences;
};
using TMessageVec = std::vector<SMessage>;

void generateTestMessages(const core_t::TTime& startTime, TMessageVec& result) {
    const std::size_t numberMessages = 100000;
    const std::size_t numberPeople = 40;
    const std::size_t numberCategories = 10;
    const double locations[] = {1.0, 2.0,  5.0,  15.0, 3.0,
                                0.5, 10.0, 17.0, 8.5,  1.5};
    const double scales[] = {1.0, 1.0, 3.0, 2.0, 0.5, 0.5, 2.0, 3.0, 4.0, 1.0};

    result.clear();
    result.reserve(numberMessages);

    test::CRandomNumbers rng;

    TDoubleVec times;
    rng.generateUniformSamples(0.0, 86399.99, numberMessages, times);
    std::sort(times.begin(), times.end());

    TDoubleVec people;
    rng.generateUniformSamples(0.0, static_cast<double>(numberPeople) - 0.01,
                               numberMessages, people);

    TDoubleVec categories;
    rng.generateUniformSamples(0.0, static_cast<double>(numberCategories) - 0.01,
                               numberMessages, categories);

    for (std::size_t i = 0; i < numberMessages; ++i) {
        core_t::TTime time = startTime + static_cast<core_t::TTime>(times[i]);
        std::size_t person = static_cast<std::size_t>(people[i]);
        std::size_t attribute = static_cast<std::size_t>(categories[i]);
        TDoubleVec value;
        rng.generateNormalSamples(locations[attribute], scales[attribute], 1u, value);
        result.push_back(SMessage(
            time, std::string("p") + boost::lexical_cast<std::string>(person),
            std::string("c") + boost::lexical_cast<std::string>(attribute), value[0]));
    }
}

void addArrival(const SMessage& message, CDataGatherer& gatherer, CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&message.s_Person);
    fields.push_back(&message.s_Attribute);
    for (std::size_t i = 0; i < message.s_Influences.size(); ++i) {
        if (message.s_Influences[i].empty()) {
            fields.push_back(static_cast<std::string*>(nullptr));
        } else {
            fields.push_back(&message.s_Influences[i]);
        }
    }
    std::string value = core::CStringUtils::typeToStringPrecise(
        message.s_Value, core::CIEEE754::E_DoublePrecision);
    fields.push_back(&value);
    CEventData result;
    result.time(message.s_Time);
    gatherer.addArrival(fields, result, resourceMonitor);
}

bool isSpace(const char x) {
    return x == ' ' || x == '\t';
}

const CSearchKey searchKey;
const std::string EMPTY_STRING;

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testMean, CTestFixture) {
    // Test that we correctly sample the bucket means.

    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TStrStrPrMeanAccumulatorMap = std::map<TStrStrPr, TMeanAccumulator>;
    using TStrStrPrMeanAccumulatorMapCItr = TStrStrPrMeanAccumulatorMap::const_iterator;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);
    BOOST_TEST_REQUIRE(gatherer.isPopulation());

    TStrStrPrMeanAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap means;
            for (std::size_t j = 0; j < data.size(); ++j) {
                if (data[j].second.s_BucketValue) {
                    means[TStrStrPr(gatherer.personName(data[j].first.first),
                                    gatherer.attributeName(data[j].first.second))] =
                        data[j].second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMeans;
            for (TStrStrPrMeanAccumulatorMapCItr itr = accumulators.begin();
                 itr != accumulators.end(); ++itr) {
                expectedMeans[itr->first] =
                    maths::common::CBasicStatistics::mean(itr->second);
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMeans),
                                core::CContainerPrinter::print(means));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)].add(
            messages[i].s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testMin, CTestFixture) {
    // Test that we correctly sample the bucket minimums.

    using TMinAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsStack<double, 1>;
    using TStrStrPrMinAccumulatorMap = std::map<TStrStrPr, TMinAccumulator>;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMinByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrMinAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap mins;
            for (std::size_t j = 0; j < data.size(); ++j) {
                if (data[j].second.s_BucketValue) {
                    mins[TStrStrPr(gatherer.personName(data[j].first.first),
                                   gatherer.attributeName(data[j].first.second))] =
                        data[j].second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMins;
            for (auto itr = accumulators.begin(); itr != accumulators.end(); ++itr) {
                expectedMins[itr->first] = itr->second[0];
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMins),
                                core::CContainerPrinter::print(mins));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)].add(
            messages[i].s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testMax, CTestFixture) {
    // Test that we correctly sample the bucket maximums.

    using TMaxAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>>;
    using TStrStrPrMaxAccumulatorMap = std::map<TStrStrPr, TMaxAccumulator>;
    using TStrStrPrMaxAccumulatorMapCItr = TStrStrPrMaxAccumulatorMap::const_iterator;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMaxByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrMaxAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap maxs;
            for (std::size_t j = 0; j < data.size(); ++j) {
                if (data[j].second.s_BucketValue) {
                    maxs[TStrStrPr(gatherer.personName(data[j].first.first),
                                   gatherer.attributeName(data[j].first.second))] =
                        data[j].second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMaxs;
            for (TStrStrPrMaxAccumulatorMapCItr itr = accumulators.begin();
                 itr != accumulators.end(); ++itr) {
                expectedMaxs[itr->first] = itr->second[0];
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMaxs),
                                core::CContainerPrinter::print(maxs));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)].add(
            messages[i].s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testSum, CTestFixture) {
    // Test that we correctly sample the bucket sums.

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationSumByBucketPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrDoubleMap expectedSums;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(expectedSums.size(), data.size());
            TStrStrPrDoubleMap sums;
            for (std::size_t j = 0; j < data.size(); ++j) {
                if (data[j].second.s_BucketValue) {
                    sums[TStrStrPr(gatherer.personName(data[j].first.first),
                                   gatherer.attributeName(data[j].first.second))] =
                        data[j].second.s_BucketValue->value()[0];
                }
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedSums),
                                core::CContainerPrinter::print(sums));

            bucketStart += bucketLength;
            expectedSums.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        expectedSums[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)] +=
            messages[i].s_Value;
    }
}

BOOST_FIXTURE_TEST_CASE(testFeatureData, CTestFixture) {
    // Test we correctly sample the mean, minimum and maximum statistics.

    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMinAccumulator = maths::common::CBasicStatistics::SMin<double>::TAccumulator;
    using TMaxAccumulator = maths::common::CBasicStatistics::SMax<double>::TAccumulator;
    using TStrStrPrMeanAccumulatorMap = std::map<TStrStrPr, TMeanAccumulator>;
    using TStrStrPrMinAccumulatorMap = std::map<TStrStrPr, TMinAccumulator>;
    using TStrStrPrMaxAccumulatorMap = std::map<TStrStrPr, TMaxAccumulator>;
    using TStrStrPrDoubleVecMap = std::map<TStrStrPr, TDoubleVec>;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrMeanAccumulatorMap bucketMeanAccumulators;
    TStrStrPrMinAccumulatorMap bucketMinAccumulators;
    TStrStrPrMaxAccumulatorMap bucketMaxAccumulators;

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            gatherer.sampleNow(bucketStart);

            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(bucketStart, featureData);
            BOOST_REQUIRE_EQUAL(static_cast<std::size_t>(3), featureData.size());
            BOOST_REQUIRE_EQUAL(model_t::E_PopulationMeanByPersonAndAttribute,
                                featureData[0].first);
            BOOST_REQUIRE_EQUAL(model_t::E_PopulationMinByPersonAndAttribute,
                                featureData[1].first);
            BOOST_REQUIRE_EQUAL(model_t::E_PopulationMaxByPersonAndAttribute,
                                featureData[2].first);

            for (std::size_t j = 0; j < 3; ++j) {
                TStrStrPrDoubleMap bucketValues;
                TStrStrPrDoubleVecMap sampleValues;
                for (const auto& data : featureData[j].second) {
                    TStrStrPr key{gatherer.personName(data.first.first),
                                  gatherer.attributeName(data.first.second)};
                    if (data.second.s_BucketValue) {
                        bucketValues[key] = data.second.s_BucketValue->value()[0];
                    }
                    for (const auto& sample : data.second.s_Samples) {
                        sampleValues[key].push_back(sample.value()[0]);
                    }
                }
                TStrStrPrDoubleMap expectedBucketValues;
                TStrStrPrDoubleVecMap expectedSampleValues;
                switch (j) {
                case 0:
                    for (const auto & [ key, value ] : bucketMeanAccumulators) {
                        expectedBucketValues[key] =
                            maths::common::CBasicStatistics::mean(value);
                        expectedSampleValues[key].push_back(
                            maths::common::CBasicStatistics::mean(value));
                    }
                    break;
                case 1:
                    for (const auto & [ key, value ] : bucketMinAccumulators) {
                        expectedBucketValues[key] = value[0];
                        expectedSampleValues[key].push_back(value[0]);
                    }
                    break;
                case 2:
                    for (const auto & [ key, value ] : bucketMaxAccumulators) {
                        expectedBucketValues[key] = value[0];
                        expectedSampleValues[key].push_back(value[0]);
                    }
                    break;
                }
                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedBucketValues),
                                    core::CContainerPrinter::print(bucketValues));
                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedSampleValues),
                                    core::CContainerPrinter::print(sampleValues));
            }

            bucketStart += bucketLength;
            bucketMeanAccumulators.clear();
            bucketMinAccumulators.clear();
            bucketMaxAccumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);

        TStrStrPr key(messages[i].s_Person, messages[i].s_Attribute);
        bucketMeanAccumulators[key].add(messages[i].s_Value);
        bucketMinAccumulators[key].add(messages[i].s_Value);
        bucketMaxAccumulators[key].add(messages[i].s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testRemovePeople, CTestFixture) {
    // Check that all the state is correctly updated when some
    // people are removed.

    using TSizeVec = std::vector<std::size_t>;
    using TSizeUInt64Pr = std::pair<std::size_t, std::uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
    using TStrFeatureDataPr = std::pair<std::string, SMetricFeatureData>;
    using TStrFeatureDataPrVec = std::vector<TStrFeatureDataPr>;
    using TStrSizeMap = std::map<std::string, std::uint64_t>;

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
    CDataGatherer gatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, searchKey, features, startTime);

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");
            gatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    // Remove people 0, 1, 9, 10, 11 and 39 (i.e. last).
    TSizeVec peopleToRemove;
    peopleToRemove.push_back(0u);
    peopleToRemove.push_back(1u);
    peopleToRemove.push_back(9u);
    peopleToRemove.push_back(10u);
    peopleToRemove.push_back(11u);
    peopleToRemove.push_back(39u);

    std::size_t numberPeople = gatherer.numberActivePeople();
    BOOST_REQUIRE_EQUAL(numberPeople, gatherer.numberOverFieldValues());
    TStrVec expectedPersonNames;
    TSizeVec expectedPersonIds;
    for (std::size_t i = 0; i < numberPeople; ++i) {
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
        gatherer.personNonZeroCounts(bucketStart, nonZeroCounts);
        for (std::size_t i = 0; i < nonZeroCounts.size(); ++i) {
            if (!std::binary_search(peopleToRemove.begin(), peopleToRemove.end(),
                                    nonZeroCounts[i].first)) {
                const std::string& name = gatherer.personName(nonZeroCounts[i].first);
                expectedNonZeroCounts[name] = nonZeroCounts[i].second;
            }
        }
    }
    LOG_DEBUG(<< "expectedNonZeroCounts = " << expectedNonZeroCounts);

    LOG_TRACE(<< "Expected");
    TStrFeatureDataPrVec expectedFeatureData;
    {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, featureData);
        for (std::size_t i = 0; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0; j < data.size(); ++j) {
                if (!std::binary_search(peopleToRemove.begin(),
                                        peopleToRemove.end(), data[j].first.first)) {
                    std::string key = model_t::print(featureData[i].first) + " " +
                                      gatherer.personName(data[j].first.first) + " " +
                                      gatherer.attributeName(data[j].first.second);
                    expectedFeatureData.push_back(TStrFeatureDataPr(key, data[j].second));
                    LOG_TRACE(<< "  " << key);
                    LOG_TRACE(<< "  " << data[j].second.print());
                }
            }
        }
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
    gatherer.personNonZeroCounts(bucketStart, nonZeroCounts);
    for (std::size_t i = 0; i < nonZeroCounts.size(); ++i) {
        const std::string& name = gatherer.personName(nonZeroCounts[i].first);
        actualNonZeroCounts[name] = nonZeroCounts[i].second;
    }
    LOG_DEBUG(<< "actualNonZeroCounts = " << actualNonZeroCounts);

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedNonZeroCounts),
                        core::CContainerPrinter::print(actualNonZeroCounts));

    LOG_TRACE(<< "Actual");
    TStrFeatureDataPrVec actualFeatureData;
    {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, featureData);
        for (std::size_t i = 0; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0; j < data.size(); ++j) {
                std::string key = model_t::print(featureData[i].first) + " " +
                                  gatherer.personName(data[j].first.first) + " " +
                                  gatherer.attributeName(data[j].first.second);
                actualFeatureData.emplace_back(key, data[j].second);
                LOG_TRACE(<< "  " << key);
                LOG_TRACE(<< "    " << data[j].second.print());
            }
        }
    }

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedFeatureData),
                        core::CContainerPrinter::print(actualFeatureData));
}

BOOST_FIXTURE_TEST_CASE(testRemoveAttributes, CTestFixture) {
    // Check that all the state is correctly updated when some
    // attributes are removed.

    using TSizeVec = std::vector<std::size_t>;
    using TStrFeatureDataPr = std::pair<std::string, SMetricFeatureData>;
    using TStrFeatureDataPrVec = std::vector<TStrFeatureDataPr>;

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
    CDataGatherer gatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, {}, searchKey, features, startTime);

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");
            gatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    // Remove attributes 0, 2, 3, 9 (i.e. last).
    TSizeVec attributesToRemove;
    attributesToRemove.push_back(0);
    attributesToRemove.push_back(2);
    attributesToRemove.push_back(3);
    attributesToRemove.push_back(9);

    std::size_t numberAttributes = gatherer.numberActiveAttributes();
    BOOST_REQUIRE_EQUAL(numberAttributes, gatherer.numberByFieldValues());
    TStrVec expectedAttributeNames;
    TSizeVec expectedAttributeIds;
    for (std::size_t i = 0; i < numberAttributes; ++i) {
        if (!std::binary_search(attributesToRemove.begin(), attributesToRemove.end(), i)) {
            expectedAttributeNames.push_back(gatherer.attributeName(i));
            expectedAttributeIds.push_back(i);
        } else {
            LOG_DEBUG(<< "Removing " << gatherer.attributeName(i));
        }
    }

    std::string expectedFeatureData;
    {
        LOG_TRACE(<< "Expected");
        TStrFeatureDataPrVec expected;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, featureData);
        for (std::size_t i = 0; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0; j < data.size(); ++j) {
                if (!std::binary_search(attributesToRemove.begin(),
                                        attributesToRemove.end(), data[j].first.second)) {
                    std::string key = model_t::print(featureData[i].first) + " " +
                                      gatherer.personName(data[j].first.first) + " " +
                                      gatherer.attributeName(data[j].first.second);
                    expected.emplace_back(key, data[j].second);
                    LOG_TRACE(<< "  " << key);
                    LOG_TRACE(<< "    " << data[j].second.print());
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

    numberAttributes = gatherer.numberActiveAttributes();

    std::string actualFeatureData;
    {
        LOG_TRACE(<< "Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, featureData);
        for (std::size_t i = 0; i < featureData.size(); ++i) {
            const TSizeSizePrFeatureDataPrVec& data = featureData[i].second;
            for (std::size_t j = 0; j < data.size(); ++j) {
                std::string key = model_t::print(featureData[i].first) + " " +
                                  gatherer.personName(data[j].first.first) + " " +
                                  gatherer.attributeName(data[j].first.second);
                actual.emplace_back(key, data[j].second);
                LOG_TRACE(<< "  " << key);
                LOG_TRACE(<< "    " << data[j].second.print());
            }
        }
        actualFeatureData = core::CContainerPrinter::print(actual);
    }

    BOOST_REQUIRE_EQUAL(expectedFeatureData, actualFeatureData);
}

BOOST_FIXTURE_TEST_CASE(testInfluenceStatistics, CTestFixture) {
    using TDoubleDoublePr = std::pair<double, double>;
    using TStrDoubleDoublePrPr = std::pair<std::string, TDoubleDoublePr>;
    using TStrDoubleDoublePrPrVec = std::vector<TStrDoubleDoublePrPr>;

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    TStrVecVec influencerValues{{"i11", "i12", "i13"}, {"i21", "i22", "i23"}};

    TMessageVec data{
        SMessage(1, "p1", "", 1.0, {influencerValues[0][0], influencerValues[1][0]}), // Bucket 1
        SMessage(150, "p1", "", 5.0, {influencerValues[0][1], influencerValues[1][1]}),
        SMessage(150, "p1", "", 3.0, {influencerValues[0][2], influencerValues[1][2]}),
        SMessage(550, "p2", "", 2.0, {influencerValues[0][0], influencerValues[1][0]}),
        SMessage(551, "p2", "", 2.1, {influencerValues[0][1], influencerValues[1][1]}),
        SMessage(552, "p2", "", 4.0, {influencerValues[0][2], influencerValues[1][2]}),
        SMessage(554, "p2", "", 2.2, {influencerValues[0][2], influencerValues[1][2]}),
        SMessage(600, "p1", "", 3.0, {influencerValues[0][1], influencerValues[1][0]}), // Bucket 2
        SMessage(660, "p2", "", 3.0, {influencerValues[0][0], influencerValues[1][2]}),
        SMessage(690, "p1", "", 7.3, {influencerValues[0][1], ""}),
        SMessage(700, "p2", "", 4.0, {influencerValues[0][0], influencerValues[1][2]}),
        SMessage(800, "p1", "", 2.2, {influencerValues[0][2], influencerValues[1][0]}),
        SMessage(900, "p2", "", 2.5, {influencerValues[0][1], influencerValues[1][0]}),
        SMessage(1000, "p1", "", 5.0, {influencerValues[0][1], influencerValues[1][0]}),
        SMessage(1200, "p2", "", 6.4, {"", influencerValues[1][2]}), // Bucket 3
        SMessage(1210, "p2", "", 6.0, {"", influencerValues[1][2]}),
        SMessage(1240, "p2", "", 7.0, {"", influencerValues[1][1]}),
        SMessage(1600, "p2", "", 11.0, {"", influencerValues[1][0]}),
        SMessage(1800, "p1", "", 11.0, {"", ""}) // Sentinel
    };

    TStrVec expectedStatistics{
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (3.1, 2)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (3.1, 2))]",
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (2.2, 1)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (2.2, 1))]",
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (4, 1))]",
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (6.2, 2)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (6.2, 2))]",
        "[(i12, (5.1, 3)), (i13, (2.2, 1)), (i21, (3.4, 3))]",
        "[(i11, (3.5, 2)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (3.5, 2))]",
        "[(i12, (3, 1)), (i13, (2.2, 1)), (i21, (2.2, 1))]",
        "[(i11, (3, 1)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (3, 1))]",
        "[(i12, (7.3, 1)), (i13, (2.2, 1)), (i21, (5, 1))]",
        "[(i11, (4, 1)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (4, 1))]",
        "[(i12, (15.3, 3)), (i13, (2.2, 1)), (i21, (10.2, 3))]",
        "[(i11, (7, 2)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (7, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.2, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 2))]"};

    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute,
                                  model_t::E_PopulationMinByPersonAndAttribute,
                                  model_t::E_PopulationMaxByPersonAndAttribute,
                                  model_t::E_PopulationHighSumByBucketPersonAndAttribute};
    TStrVec influencerNames{"i1", "i2"};
    CDataGatherer gatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           EMPTY_STRING, influencerNames, searchKey, features, startTime);

    core_t::TTime bucketStart = startTime;
    auto expected = expectedStatistics.begin();
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "*** processing bucket ***");
            TFeatureSizeSizePrFeatureDataPrVecPrVec featuresData;
            gatherer.featureData(bucketStart, featuresData);
            for (const auto & [ feature, featureData ] : featuresData) {
                LOG_DEBUG(<< "feature = " << model_t::print(feature));
                for (const auto& data_ : featureData) {
                    TStrDoubleDoublePrPrVec statistics;
                    for (const auto& influenceValue : data_.second.s_InfluenceValues) {
                        for (const auto & [ influence, value ] : influenceValue) {
                            statistics.emplace_back(
                                influence, std::make_pair(value.first[0], value.second));
                        }
                    }
                    std::sort(statistics.begin(), statistics.end(),
                              maths::common::COrderings::SFirstLess());

                    LOG_DEBUG(<< "statistics = " << statistics);
                    LOG_DEBUG(<< "expected   = " << *expected);
                    BOOST_REQUIRE_EQUAL(*(expected++),
                                        core::CContainerPrinter::print(statistics));
                }
            }

            bucketStart += bucketLength;
        }
        addArrival(data[i], gatherer, m_ResourceMonitor);
    }
}

BOOST_FIXTURE_TEST_CASE(testPersistence, CTestFixture) {
    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationHighSumByBucketPersonAndAttribute);
    CDataGatherer origDataGatherer(model_t::E_PopulationMetric, model_t::E_None,
                                   params, EMPTY_STRING, EMPTY_STRING,
                                   EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, {},
                                   searchKey, features, startTime);

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");
            origDataGatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(messages[i], origDataGatherer, m_ResourceMonitor);
    }
    origDataGatherer.sampleNow(bucketStart);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origDataGatherer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    //LOG_DEBUG(<< "origXml = " << origXml);
    LOG_DEBUG(<< "origXml length = " << origXml.length() << ", # tabs "
              << std::count_if(origXml.begin(), origXml.end(), isSpace));

    std::size_t length = origXml.length() -
                         std::count_if(origXml.begin(), origXml.end(), isSpace);
    BOOST_TEST_REQUIRE(length < 645000);

    // Restore the XML into a new data gatherer
    core::CRapidXmlParser parser;
    BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CDataGatherer restoredDataGatherer(model_t::E_PopulationMetric,
                                       model_t::E_None, params, EMPTY_STRING,
                                       EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                       EMPTY_STRING, {}, searchKey, traverser);

    // The XML representation of the new data gatherer should be the same as the
    // original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredDataGatherer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    //LOG_DEBUG(<< "newXml = " << newXml);
    LOG_DEBUG(<< "newXml length = " << newXml.length() << ", # tabs "
              << std::count_if(newXml.begin(), newXml.end(), isSpace));

    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_SUITE_END()
}
