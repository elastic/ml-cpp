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
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CoreTypes.h>
#include <core/UnwrapRef.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CMetricPopulationModelFactory.h>
#include <model/CResourceMonitor.h>
#include <model/CSampleGatherer.h>
#include <model/ModelTypes.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <map>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "ModelTestHelpers.h"

BOOST_AUTO_TEST_SUITE(CMetricPopulationDataGathererTest)

using namespace ml;
using namespace model;

namespace {

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TStrStrPr = std::pair<std::string, std::string>;
using TStrStrPrDoubleMap = std::map<TStrStrPr, double>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, SMetricFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TFeatureSizeSizePrFeatureDataPrVecPr =
    std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;

struct SMessage {
    SMessage(const core_t::TTime& time,
             std::string person,
             std::string attribute,
             const double& value,
             TStrVec influences = TStrVec())
        : s_Time(time), s_Person(std::move(person)),
          s_Attribute(std::move(attribute)), s_Value(value),
          s_Influences(std::move(influences)) {}

    core_t::TTime s_Time;
    std::string s_Person;
    std::string s_Attribute;
    double s_Value;
    TStrVec s_Influences;
};
using TMessageVec = std::vector<SMessage>;

TStrVec vec(const std::string& s1, const std::string& s2) {
    TStrVec result(1, s1);
    result.push_back(s2);
    return result;
}

void generateTestMessages(const core_t::TTime& startTime, TMessageVec& result) {
    constexpr std::size_t numberMessages = 100000;
    constexpr std::size_t numberPeople = 40;
    constexpr std::size_t numberCategories = 10;
    constexpr std::array locations = {1.0, 2.0,  5.0,  15.0, 3.0,
                                      0.5, 10.0, 17.0, 8.5,  1.5};
    constexpr std::array scales = {1.0, 1.0, 3.0, 2.0, 0.5,
                                   0.5, 2.0, 3.0, 4.0, 1.0};

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
        core_t::TTime const time = startTime + static_cast<core_t::TTime>(times[i]);
        auto const person = static_cast<std::size_t>(people[i]);
        auto const attribute = static_cast<std::size_t>(categories[i]);
        TDoubleVec value;
        rng.generateNormalSamples(locations[attribute], scales[attribute], 1U, value);
        result.emplace_back(
            time, std::string("p") + boost::lexical_cast<std::string>(person),
            std::string("c") + boost::lexical_cast<std::string>(attribute), value[0]);
    }
}

void addArrival(const SMessage& message, CDataGatherer& gatherer, CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&message.s_Person);
    fields.push_back(&message.s_Attribute);
    for (const auto& s_Influence : message.s_Influences) {
        if (s_Influence.empty()) {
            fields.push_back(static_cast<std::string*>(nullptr));
        } else {
            fields.push_back(&s_Influence);
        }
    }
    std::string const value = core::CStringUtils::typeToStringPrecise(
        message.s_Value, core::CIEEE754::E_DoublePrecision);
    fields.push_back(&value);
    CEventData result;
    result.time(message.s_Time);
    gatherer.addArrival(fields, result, resourceMonitor);
}

const CSearchKey searchKey;
const std::string EMPTY_STRING;

} // unnamed::

class CTestFixture {
protected:
    CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testMean, CTestFixture) {
    // Test that we correctly sample the bucket means.

    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TStrStrPrMeanAccumulatorMap = std::map<TStrStrPr, TMeanAccumulator>;

    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMeanByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);
    BOOST_TEST_REQUIRE(gatherer.isPopulation());

    TStrStrPrMeanAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap means;
            for (const auto& j : data) {
                if (j.second.s_BucketValue) {
                    means[TStrStrPr(gatherer.personName(j.first.first),
                                    gatherer.attributeName(j.first.second))] =
                        j.second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMeans;
            for (const auto & [ fst, snd ] : accumulators) {
                expectedMeans[fst] = maths::common::CBasicStatistics::mean(snd);
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMeans),
                                core::CContainerPrinter::print(means));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(message, gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(message.s_Person, message.s_Attribute)].add(
            message.s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testMin, CTestFixture) {
    // Test that we correctly sample the bucket minimums.

    using TMinAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsStack<double, 1U>;
    using TStrStrPrMinAccumulatorMap = std::map<TStrStrPr, TMinAccumulator>;

    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMinByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrMinAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap mins;
            for (const auto& j : data) {
                if (j.second.s_BucketValue) {
                    mins[TStrStrPr(gatherer.personName(j.first.first),
                                   gatherer.attributeName(j.first.second))] =
                        j.second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMins;
            for (auto& accumulator : accumulators) {
                expectedMins[accumulator.first] = accumulator.second[0];
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMins),
                                core::CContainerPrinter::print(mins));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(message, gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(message.s_Person, message.s_Attribute)].add(
            message.s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testMax, CTestFixture) {
    // Test that we correctly sample the bucket maximums.

    using TMaxAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsStack<double, 1U, std::greater<>>;
    using TStrStrPrMaxAccumulatorMap = std::map<TStrStrPr, TMaxAccumulator>;

    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationMaxByPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrMaxAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap maxs;
            for (const auto& j : data) {
                if (j.second.s_BucketValue) {
                    maxs[TStrStrPr(gatherer.personName(j.first.first),
                                   gatherer.attributeName(j.first.second))] =
                        j.second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMaxs;
            for (auto& accumulator : accumulators) {
                expectedMaxs[accumulator.first] = accumulator.second[0];
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMaxs),
                                core::CContainerPrinter::print(maxs));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(message, gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(message.s_Person, message.s_Attribute)].add(
            message.s_Value);
    }
}

BOOST_FIXTURE_TEST_CASE(testSum, CTestFixture) {
    // Test that we correctly sample the bucket sums.

    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_PopulationSumByBucketPersonAndAttribute};
    factory.features(features);
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrDoubleMap expectedSums;
    core_t::TTime bucketStart = startTime;
    for (auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            BOOST_REQUIRE_EQUAL(features.size(), tmp.size());
            BOOST_REQUIRE_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec& data = tmp[0].second;
            BOOST_REQUIRE_EQUAL(expectedSums.size(), data.size());
            TStrStrPrDoubleMap sums;
            for (const auto& j : data) {
                if (j.second.s_BucketValue) {
                    sums[TStrStrPr(gatherer.personName(j.first.first),
                                   gatherer.attributeName(j.first.second))] =
                        j.second.s_BucketValue->value()[0];
                }
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedSums),
                                core::CContainerPrinter::print(sums));

            bucketStart += bucketLength;
            expectedSums.clear();
        }

        addArrival(message, gatherer, m_ResourceMonitor);
        expectedSums[TStrStrPr(message.s_Person, message.s_Attribute)] +=
            message.s_Value;
    }
}

BOOST_FIXTURE_TEST_CASE(testSampleCount, CTestFixture) {
    // Test that we set sensible sample counts for each attribute.

    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;
    SModelParams const params(bucketLength);
    const std::string attribute("c1");
    const std::string person("p1");
    constexpr std::size_t numberBuckets = 40;
    constexpr std::array personMessageCount = {
        11,  11,  11,  11,  110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
        110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
        110, 110, 110, 97,  97,  97,  97,  97,  97,  97,  97,  97};
    constexpr std::array expectedSampleCounts = {
        0.0,     0.0,    0.0,     11.0,   11.0,    11.0,    11.0,    11.0,
        11.0,    11.0,   11.0,    11.0,   11.0,    11.0,    11.0,    11.0,
        11.0,    11.0,   11.0,    11.0,   11.0,    11.0,    11.0,    11.0,
        11.0,    11.0,   11.0,    11.0,   11.0,    11.0,    11.3597, 11.7164,
        12.0701, 12.421, 12.7689, 13.114, 13.4562, 13.7957, 14.1325, 14.4665};
    constexpr double tolerance = 5e-4;

    TMessageVec messages;
    for (std::size_t bucket = 0; bucket < numberBuckets; ++bucket) {
        core_t::TTime const bucketStart =
            startTime + (static_cast<core_t::TTime>(bucket) * bucketLength);

        std::size_t const n = personMessageCount[bucket];
        for (std::size_t i = 0; i < n; ++i) {
            core_t::TTime const time =
                bucketStart + (bucketLength * static_cast<core_t::TTime>(i) /
                               static_cast<core_t::TTime>(n));
            messages.emplace_back(time, person, attribute, 1.0);
        }
    }

    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationMeanByPersonAndAttribute});
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    std::size_t bucket = 0;
    for (const auto& message : messages) {
        if (core_t::TTime const bucketStart =
                startTime + (static_cast<core_t::TTime>(bucket) * bucketLength);
            message.s_Time >= bucketStart + bucketLength) {
            gatherer.sampleNow(bucketStart);
            LOG_DEBUG(<< gatherer.effectiveSampleCount(0));
            BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedSampleCounts[bucket],
                                         gatherer.effectiveSampleCount(0), tolerance);
            ++bucket;
        }

        addArrival(message, gatherer, m_ResourceMonitor);
    }

    core_t::TTime const bucketStart =
        startTime + (static_cast<core_t::TTime>(bucket) * bucketLength);
    gatherer.sampleNow(bucketStart);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(expectedSampleCounts[bucket],
                                 gatherer.effectiveSampleCount(0), tolerance);
}

BOOST_FIXTURE_TEST_CASE(testFeatureData, CTestFixture) {
    // Test we correctly sample the mean, minimum and maximum statistics.

    using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TStrStrPrMeanAccumulatorMap = std::map<TStrStrPr, TMeanAccumulator>;
    using TMinAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsStack<double, 1U>;
    using TStrStrPrMinAccumulatorMap = std::map<TStrStrPr, TMinAccumulator>;
    using TMaxAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsStack<double, 1U, std::greater<>>;
    using TStrStrPrMaxAccumulatorMap = std::map<TStrStrPr, TMaxAccumulator>;
    using TStrStrPrDoubleVecMap = std::map<TStrStrPr, TDoubleVec>;

    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;

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
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);

    TStrStrPrMeanAccumulatorMap bucketMeanAccumulators;
    TStrStrPrMeanAccumulatorMap sampleMeanAccumulators;
    TStrStrPrDoubleVecMap expectedMeanSamples;
    TStrStrPrMinAccumulatorMap bucketMinAccumulators;
    TStrStrPrMinAccumulatorMap sampleMinAccumulators;
    TStrStrPrDoubleVecMap expectedMinSamples;
    TStrStrPrMaxAccumulatorMap bucketMaxAccumulators;
    TStrStrPrMaxAccumulatorMap sampleMaxAccumulators;
    TStrStrPrDoubleVecMap expectedMaxSamples;
    core_t::TTime bucketStart = startTime;
    for (auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");

            gatherer.sampleNow(bucketStart);

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            BOOST_REQUIRE_EQUAL(static_cast<std::size_t>(3), tmp.size());

            BOOST_REQUIRE_EQUAL(model_t::E_PopulationMeanByPersonAndAttribute,
                                tmp[0].first);
            TStrStrPrDoubleMap means;
            TStrStrPrDoubleVecMap meanSamples;
            for (const auto& data : tmp[0].second) {
                TStrStrPr const key(gatherer.personName(data.first.first),
                                    gatherer.attributeName(data.first.second));
                if (data.second.s_BucketValue) {
                    means[key] = data.second.s_BucketValue->value()[0];
                }
                TDoubleVec& samples = meanSamples[key];
                for (const auto& k : core::unwrap_ref(data.second.s_Samples)) {
                    samples.push_back(k.value()[0]);
                }
            }

            BOOST_REQUIRE_EQUAL(model_t::E_PopulationMinByPersonAndAttribute,
                                tmp[1].first);
            TStrStrPrDoubleMap mins;
            TStrStrPrDoubleVecMap minSamples;
            for (const auto& data : tmp[1].second) {
                TStrStrPr const key(gatherer.personName(data.first.first),
                                    gatherer.attributeName(data.first.second));
                if (data.second.s_BucketValue) {
                    mins[key] = data.second.s_BucketValue->value()[0];
                }
                TDoubleVec& samples = minSamples[key];
                for (const auto& k : core::unwrap_ref(data.second.s_Samples)) {
                    samples.push_back(k.value()[0]);
                }
            }

            BOOST_REQUIRE_EQUAL(model_t::E_PopulationMaxByPersonAndAttribute,
                                tmp[2].first);
            TStrStrPrDoubleMap maxs;
            TStrStrPrDoubleVecMap maxSamples;
            for (const auto& data : tmp[2].second) {
                TStrStrPr const key(gatherer.personName(data.first.first),
                                    gatherer.attributeName(data.first.second));
                if (data.second.s_BucketValue) {
                    maxs[key] = data.second.s_BucketValue->value()[0];
                }
                TDoubleVec& samples = maxSamples[key];
                for (const auto& k : core::unwrap_ref(data.second.s_Samples)) {
                    samples.push_back(k.value()[0]);
                }
            }

            TStrStrPrDoubleMap expectedMeans;
            for (const auto & [ key, values ] : bucketMeanAccumulators) {
                expectedMeans[key] = maths::common::CBasicStatistics::mean(values);
            }

            TStrStrPrDoubleMap expectedMins;
            for (const auto & [ key, values ] : bucketMinAccumulators) {
                expectedMins[key] = values[0];
            }

            TStrStrPrDoubleMap expectedMaxs;
            for (const auto & [ key, values ] : bucketMaxAccumulators) {
                expectedMaxs[key] = values[0];
            }

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMeans),
                                core::CContainerPrinter::print(means));
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMins),
                                core::CContainerPrinter::print(mins));
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMaxs),
                                core::CContainerPrinter::print(maxs));

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMeanSamples),
                                core::CContainerPrinter::print(meanSamples));
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMinSamples),
                                core::CContainerPrinter::print(minSamples));
            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedMaxSamples),
                                core::CContainerPrinter::print(maxSamples));

            bucketStart += bucketLength;
            bucketMeanAccumulators.clear();
            expectedMeanSamples.clear();
            bucketMinAccumulators.clear();
            expectedMinSamples.clear();
            bucketMaxAccumulators.clear();
            expectedMaxSamples.clear();
        }

        addArrival(message, gatherer, m_ResourceMonitor);
        TStrStrPr const key(message.s_Person, message.s_Attribute);

        bucketMeanAccumulators[key].add(message.s_Value);
        bucketMinAccumulators[key].add(message.s_Value);
        bucketMaxAccumulators[key].add(message.s_Value);
        expectedMeanSamples.insert(TStrStrPrDoubleVecMap::value_type(key, TDoubleVec()));
        expectedMinSamples.insert(TStrStrPrDoubleVecMap::value_type(key, TDoubleVec()));
        expectedMaxSamples.insert(TStrStrPrDoubleVecMap::value_type(key, TDoubleVec()));

        std::size_t cid;
        BOOST_TEST_REQUIRE(gatherer.attributeId(message.s_Attribute, cid));

        double const sampleCount = gatherer.effectiveSampleCount(cid);
        if (sampleCount > 0.0) {
            sampleMeanAccumulators[key].add(message.s_Value);
            sampleMinAccumulators[key].add(message.s_Value);
            sampleMaxAccumulators[key].add(message.s_Value);
            if (maths::common::CBasicStatistics::count(sampleMeanAccumulators[key]) ==
                std::floor(sampleCount + 0.5)) {
                expectedMeanSamples[key].push_back(
                    maths::common::CBasicStatistics::mean(sampleMeanAccumulators[key]));
                expectedMinSamples[key].push_back(sampleMinAccumulators[key][0]);
                expectedMaxSamples[key].push_back(sampleMaxAccumulators[key][0]);
                sampleMeanAccumulators[key] = TMeanAccumulator();
                sampleMinAccumulators[key] = TMinAccumulator();
                sampleMaxAccumulators[key] = TMaxAccumulator();
            }
        }
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

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
    CDataGatherer gatherer = CDataGathererBuilder(model_t::E_PopulationMetric, features,
                                                  params, searchKey, startTime)
                                 .build();
    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");
            gatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(message, gatherer, m_ResourceMonitor);
    }

    // Remove people 0, 1, 9, 10, 11 and 39 (i.e. last).
    TSizeVec peopleToRemove;
    peopleToRemove.push_back(0U);
    peopleToRemove.push_back(1U);
    peopleToRemove.push_back(9U);
    peopleToRemove.push_back(10U);
    peopleToRemove.push_back(11U);
    peopleToRemove.push_back(39U);

    std::size_t const numberPeople = gatherer.numberActivePeople();
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
        for (const auto& nonZeroCount : nonZeroCounts) {
            if (!std::binary_search(peopleToRemove.begin(),
                                    peopleToRemove.end(), nonZeroCount.first)) {
                const std::string& name = gatherer.personName(nonZeroCount.first);
                expectedNonZeroCounts[name] = nonZeroCount.second;
            }
        }
    }
    LOG_DEBUG(<< "expectedNonZeroCounts = " << expectedNonZeroCounts);

    LOG_TRACE(<< "Expected");
    TStrFeatureDataPrVec expectedFeatureData;
    {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (const auto & [ featureType, data ] : featureData) {
            for (const auto& j : data) {
                if (!std::binary_search(peopleToRemove.begin(),
                                        peopleToRemove.end(), j.first.first)) {
                    std::string const key = model_t::print(featureType) + " " +
                                            gatherer.personName(j.first.first) + " " +
                                            gatherer.attributeName(j.first.second);
                    expectedFeatureData.emplace_back(key, j.second);
                    LOG_TRACE(<< "  " << key);
                    LOG_TRACE(<< "  " << j.second.print());
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
    for (const auto & [ id, count ] : nonZeroCounts) {
        const std::string& name = gatherer.personName(id);
        actualNonZeroCounts[name] = count;
    }
    LOG_DEBUG(<< "actualNonZeroCounts = " << actualNonZeroCounts);

    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedNonZeroCounts),
                        core::CContainerPrinter::print(actualNonZeroCounts));

    LOG_TRACE(<< "Actual");
    TStrFeatureDataPrVec actualFeatureData;
    {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (const auto & [ featureType, data ] : featureData) {
            for (const auto& j : data) {
                std::string const key = model_t::print(featureType) + " " +
                                        gatherer.personName(j.first.first) + " " +
                                        gatherer.attributeName(j.first.second);
                actualFeatureData.emplace_back(key, j.second);
                LOG_TRACE(<< "  " << key);
                LOG_TRACE(<< "    " << j.second.print());
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

    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
    CDataGatherer gatherer = CDataGathererBuilder(model_t::E_PopulationMetric, features,
                                                  params, searchKey, startTime)
                                 .build();

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");
            gatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(message, gatherer, m_ResourceMonitor);
    }

    // Remove attributes 0, 2, 3, 9 (i.e. last).
    TSizeVec attributesToRemove;
    attributesToRemove.push_back(0U);
    attributesToRemove.push_back(2U);
    attributesToRemove.push_back(3U);
    attributesToRemove.push_back(9U);

    std::size_t numberAttributes = gatherer.numberActiveAttributes();
    BOOST_REQUIRE_EQUAL(numberAttributes, gatherer.numberByFieldValues());
    TStrVec expectedAttributeNames;
    TSizeVec expectedAttributeIds;
    TDoubleVec expectedSampleCounts;
    for (std::size_t i = 0; i < numberAttributes; ++i) {
        if (!std::binary_search(attributesToRemove.begin(), attributesToRemove.end(), i)) {
            expectedAttributeNames.push_back(gatherer.attributeName(i));
            expectedAttributeIds.push_back(i);
            expectedSampleCounts.push_back(gatherer.effectiveSampleCount(i));
        } else {
            LOG_DEBUG(<< "Removing " << gatherer.attributeName(i));
        }
    }

    std::string expectedFeatureData;
    {
        LOG_TRACE(<< "Expected");
        TStrFeatureDataPrVec expected;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (const auto & [ featureType, data ] : featureData) {
            for (const auto& j : data) {
                if (!std::binary_search(attributesToRemove.begin(),
                                        attributesToRemove.end(), j.first.second)) {
                    std::string const key = model_t::print(featureType) + " " +
                                            gatherer.personName(j.first.first) + " " +
                                            gatherer.attributeName(j.first.second);
                    expected.emplace_back(key, j.second);
                    LOG_TRACE(<< "  " << key);
                    LOG_TRACE(<< "    " << j.second.print());
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
    TDoubleVec actualSampleCounts;
    for (std::size_t i = 0; i < numberAttributes; ++i) {
        actualSampleCounts.push_back(gatherer.effectiveSampleCount(expectedAttributeIds[i]));
    }
    BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedSampleCounts),
                        core::CContainerPrinter::print(actualSampleCounts));

    std::string actualFeatureData;
    {
        LOG_TRACE(<< "Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (const auto & [ featureType, data ] : featureData) {
            for (const auto& j : data) {
                std::string const key = model_t::print(featureType) + " " +
                                        gatherer.personName(j.first.first) + " " +
                                        gatherer.attributeName(j.first.second);
                actual.emplace_back(key, j.second);
                LOG_TRACE(<< "  " << key);
                LOG_TRACE(<< "    " << j.second.print());
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

    constexpr core_t::TTime startTime = 0;
    constexpr core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    constexpr std::array influencerNames_ = {"i1", "i2"};
    std::array<std::array<std::string, 3>, 2> const influencerValues = {
        {{"i11", "i12", "i13"}, {"i21", "i22", "i23"}}};

    const std::array data = {
        SMessage(1, "p1", "", 1.0, vec(influencerValues[0][0], influencerValues[1][0])), // Bucket 1
        SMessage(150, "p1", "", 5.0, vec(influencerValues[0][1], influencerValues[1][1])),
        SMessage(150, "p1", "", 3.0, vec(influencerValues[0][2], influencerValues[1][2])),
        SMessage(550, "p2", "", 2.0, vec(influencerValues[0][0], influencerValues[1][0])),
        SMessage(551, "p2", "", 2.1, vec(influencerValues[0][1], influencerValues[1][1])),
        SMessage(552, "p2", "", 4.0, vec(influencerValues[0][2], influencerValues[1][2])),
        SMessage(554, "p2", "", 2.2, vec(influencerValues[0][2], influencerValues[1][2])),
        SMessage(600, "p1", "", 3.0, vec(influencerValues[0][1], influencerValues[1][0])), // Bucket 2
        SMessage(660, "p2", "", 3.0, vec(influencerValues[0][0], influencerValues[1][2])),
        SMessage(690, "p1", "", 7.3, vec(influencerValues[0][1], "")),
        SMessage(700, "p2", "", 4.0, vec(influencerValues[0][0], influencerValues[1][2])),
        SMessage(800, "p1", "", 2.2, vec(influencerValues[0][2], influencerValues[1][0])),
        SMessage(900, "p2", "", 2.5, vec(influencerValues[0][1], influencerValues[1][0])),
        SMessage(1000, "p1", "", 5.0, vec(influencerValues[0][1], influencerValues[1][0])),
        SMessage(1200, "p2", "", 6.4, vec("", influencerValues[1][2])), // Bucket 3
        SMessage(1210, "p2", "", 6.0, vec("", influencerValues[1][2])),
        SMessage(1240, "p2", "", 7.0, vec("", influencerValues[1][1])),
        SMessage(1600, "p2", "", 11.0, vec("", influencerValues[1][0])),
        SMessage(1800, "p1", "", 11.0, vec("", "")) // Sentinel
    };

    std::array expectedStatistics = {
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (3.1, 2)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (3.1, 2))]",
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (2.2, 1)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (2.2, 1))]",
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (4, 1)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (4, 1))]",
        "[(i11, (1, 1)), (i12, (5, 1)), (i13, (3, 1)), (i21, (1, 1)), (i22, (5, 1)), (i23, (3, 1))]",
        "[(i11, (2, 1)), (i12, (2.1, 1)), (i13, (6.2, 1)), (i21, (2, 1)), (i22, (2.1, 1)), (i23, (6.2, 1))]",
        "[(i12, (5.1, 3)), (i13, (2.2, 1)), (i21, (3.4, 3))]",
        "[(i11, (3.5, 2)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (3.5, 2))]",
        "[(i12, (3, 1)), (i13, (2.2, 1)), (i21, (2.2, 1))]",
        "[(i11, (3, 1)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (3, 1))]",
        "[(i12, (7.3, 1)), (i13, (2.2, 1)), (i21, (5, 1))]",
        "[(i11, (4, 1)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (4, 1))]",
        "[(i12, (15.3, 1)), (i13, (2.2, 1)), (i21, (10.2, 1))]",
        "[(i11, (7, 1)), (i12, (2.5, 1)), (i21, (2.5, 1)), (i23, (7, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.2, 2))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (6.4, 1))]",
        "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 1))]"};
    const char** expected = expectedStatistics.data();

    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationHighSumByBucketPersonAndAttribute);
    TStrVec const influencerNames(std::begin(influencerNames_), std::end(influencerNames_));
    CDataGatherer gatherer = CDataGathererBuilder(model_t::E_PopulationMetric, features,
                                                  params, searchKey, startTime)
                                 .influenceFieldNames(influencerNames)
                                 .build();

    core_t::TTime bucketStart = startTime;
    for (const auto& i : data) {
        if (i.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "*** processing bucket ***");
            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(bucketStart, bucketLength, featureData);
            for (const auto & [ feature, data_ ] : featureData) {
                LOG_DEBUG(<< "feature = " << model_t::print(feature));
                for (std::size_t k = 0; k < data_.size(); ++k) {
                    TStrDoubleDoublePrPrVec statistics;
                    for (const auto& s_InfluenceValue : data_[k].second.s_InfluenceValues) {
                        for (const auto& n : s_InfluenceValue) {
                            statistics.emplace_back(
                                n.first,
                                TDoubleDoublePr(n.second.first[0], n.second.second));
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
        addArrival(i, gatherer, m_ResourceMonitor);
    }
}

BOOST_FIXTURE_TEST_CASE(testPersistence, CTestFixture) {
    constexpr core_t::TTime startTime = 1367280000;
    constexpr core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    model_t::TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationHighSumByBucketPersonAndAttribute);
    CDataGatherer origDataGatherer =
        CDataGathererBuilder(model_t::E_PopulationMetric, features, params, searchKey, startTime)
            .build();

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (const auto& message : messages) {
        if (message.s_Time >= bucketStart + bucketLength) {
            LOG_DEBUG(<< "Processing bucket [" << bucketStart << ", "
                      << bucketStart + bucketLength << ")");
            origDataGatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(message, origDataGatherer, m_ResourceMonitor);
    }
    origDataGatherer.sampleNow(bucketStart);

    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, [&origDataGatherer](core::CJsonStatePersistInserter& inserter) {
            origDataGatherer.acceptPersistInserter(inserter);
        });
    LOG_DEBUG(<< "origJson length = " << origJson.str().size());
    BOOST_TEST_REQUIRE(origJson.str().size() < 292000);

    // Restore the JSON into a new data gatherer
    // The traverser expects the state json in a embedded document
    std::stringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
    core::CJsonStateRestoreTraverser traverser(origJsonStrm);
    CBucketGatherer::SBucketGathererInitData bucketGathererInitData{
        EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, {}, 0, 0};

    CDataGatherer restoredDataGatherer(model_t::E_PopulationMetric, model_t::E_None,
                                       params, EMPTY_STRING, searchKey,
                                       bucketGathererInitData, traverser);

    // The JSON representation of the new data gatherer should be the same as the
    // original
    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, [&restoredDataGatherer](core::CJsonStatePersistInserter& inserter) {
            restoredDataGatherer.acceptPersistInserter(inserter);
        });
    LOG_DEBUG(<< "newJson length = " << newJson.str().size());

    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

BOOST_FIXTURE_TEST_CASE(testReleaseMemory, CTestFixture) {
    constexpr core_t::TTime startTime = 1373932800;
    constexpr core_t::TTime bucketLength = 3600;

    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 3;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricPopulationModelFactory factory(params, interimBucketCorrector);
    factory.features({model_t::E_PopulationMeanByPersonAndAttribute});
    CModelFactory::SGathererInitializationData const initData(startTime);
    CModelFactory::TDataGathererPtr const gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer& gatherer(*gathererPtr);
    BOOST_TEST_REQUIRE(gatherer.isPopulation());

    core_t::TTime bucketStart = startTime;
    // Add a few buckets with count of 10 so that sample count gets estimated
    for (std::size_t i = 0; i < 10; ++i) {
        // Add 10 events
        for (std::size_t j = 0; j < 10; ++j) {
            addArrival(SMessage(bucketStart, "p1", "", 10.0), gatherer, m_ResourceMonitor);
            addArrival(SMessage(bucketStart, "p2", "", 10.0), gatherer, m_ResourceMonitor);
        }
        gatherer.sampleNow(bucketStart - (params.s_LatencyBuckets * bucketLength));
        bucketStart += bucketLength;
    }

    // Add a bucket with not enough data to sample for p2
    for (std::size_t j = 0; j < 10; ++j) {
        addArrival(SMessage(bucketStart, "p1", "", 10.0), gatherer, m_ResourceMonitor);
    }
    addArrival(SMessage(bucketStart, "p2", "", 10.0), gatherer, m_ResourceMonitor);
    gatherer.sampleNow(bucketStart - (params.s_LatencyBuckets * bucketLength));
    bucketStart += bucketLength;

    std::size_t const mem = gatherer.memoryUsage();

    // Add 48 + 1 buckets ( > 2 days) to force incomplete samples out of consideration for p2
    for (std::size_t i = 0; i < 49 + 1; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            addArrival(SMessage(bucketStart, "p1", "", 10.0), gatherer, m_ResourceMonitor);
        }
        gatherer.sampleNow(bucketStart - (params.s_LatencyBuckets * bucketLength));
        bucketStart += bucketLength;
        gatherer.releaseMemory(bucketStart - params.s_SamplingAgeCutoff);
        if (i <= 40) {
            BOOST_TEST_REQUIRE(gatherer.memoryUsage() >= mem - 1000);
        }
    }
    BOOST_TEST_REQUIRE(gatherer.memoryUsage() < mem - 1000);
}

BOOST_AUTO_TEST_SUITE_END()
