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

#include "CMetricPopulationDataGathererTest.h"

#include <core/CContainerPrinter.h>
#include <core/CoreTypes.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CMetricBucketGatherer.h>
#include <model/CMetricPopulationModelFactory.h>
#include <model/CResourceMonitor.h>
#include <model/CSampleGatherer.h>
#include <model/ModelTypes.h>

#include <test/CRandomNumbers.h>

#include <boost/lexical_cast.hpp>
#include <boost/range.hpp>

#include <string>
#include <vector>

using namespace ml;
using namespace model;

namespace
{

using TDoubleVec = std::vector<double>;
using TFeatureVec = std::vector<model_t::EFeature>;
using TStrVec = std::vector<std::string>;
using TStrStrPr = std::pair<std::string, std::string>;
using TStrStrPrDoubleMap = std::map<TStrStrPr, double>;
using TOptionalStr = boost::optional<std::string>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, SMetricFeatureData>;
using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;
using TFeatureSizeSizePrFeatureDataPrVecPr = std::pair<model_t::EFeature, TSizeSizePrFeatureDataPrVec>;
using TFeatureSizeSizePrFeatureDataPrVecPrVec = std::vector<TFeatureSizeSizePrFeatureDataPrVecPr>;

struct SMessage
{
    SMessage(const core_t::TTime &time,
             const std::string &person,
             const std::string &attribute,
             const double &value,
             const TStrVec &influences = TStrVec()) :
         s_Time(time),
         s_Person(person),
         s_Attribute(attribute),
         s_Value(value),
         s_Influences(influences)
    {
    }

    core_t::TTime s_Time;
    std::string s_Person;
    std::string s_Attribute;
    double s_Value;
    TStrVec s_Influences;
};
using TMessageVec = std::vector<SMessage>;

TStrVec vec(const std::string &s1, const std::string &s2)
{
    TStrVec result(1, s1);
    result.push_back(s2);
    return result;
}

void generateTestMessages(const core_t::TTime &startTime,
                          TMessageVec &result)
{
    const std::size_t numberMessages = 100000;
    const std::size_t numberPeople = 40;
    const std::size_t numberCategories = 10;
    const double locations[] = { 1.0, 2.0, 5.0, 15.0, 3.0, 0.5, 10.0, 17.0, 8.5, 1.5 };
    const double scales[] = { 1.0, 1.0, 3.0, 2.0, 0.5, 0.5, 2.0, 3.0, 4.0, 1.0 };

    result.clear();
    result.reserve(numberMessages);

    test::CRandomNumbers rng;

    TDoubleVec times;
    rng.generateUniformSamples(0.0, 86399.99, numberMessages, times);
    std::sort(times.begin(), times.end());

    TDoubleVec people;
    rng.generateUniformSamples(0.0, static_cast<double>(numberPeople) - 0.01, numberMessages, people);

    TDoubleVec categories;
    rng.generateUniformSamples(0.0, static_cast<double>(numberCategories) - 0.01, numberMessages, categories);

    for (std::size_t i = 0u; i < numberMessages; ++i)
    {
        core_t::TTime time = startTime + static_cast<core_t::TTime>(times[i]);
        std::size_t person = static_cast<std::size_t>(people[i]);
        std::size_t attribute = static_cast<std::size_t>(categories[i]);
        TDoubleVec value;
        rng.generateNormalSamples(locations[attribute], scales[attribute], 1u, value);
        result.push_back(SMessage(time,
                                  std::string("p") + boost::lexical_cast<std::string>(person),
                                  std::string("c") + boost::lexical_cast<std::string>(attribute),
                                  value[0]));
    }
}

void addArrival(const SMessage &message, CDataGatherer &gatherer, CResourceMonitor &resourceMonitor)
{
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&message.s_Person);
    fields.push_back(&message.s_Attribute);
    for (std::size_t i = 0u; i < message.s_Influences.size(); ++i)
    {
        if (message.s_Influences[i].empty())
        {
            fields.push_back(static_cast<std::string *>(0));
        }
        else
        {
            fields.push_back(&message.s_Influences[i]);
        }
    }
    std::string value = core::CStringUtils::typeToStringPrecise(message.s_Value, core::CIEEE754::E_DoublePrecision);
    fields.push_back(&value);
    CEventData result;
    result.time(message.s_Time);
    gatherer.addArrival(fields, result, resourceMonitor);
}

bool isSpace(const char x)
{
    return x == ' ' || x == '\t';
}

const CSearchKey searchKey;
const std::string EMPTY_STRING;

} // unnamed::

void CMetricPopulationDataGathererTest::testMean(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testMean ***");

    // Test that we correctly sample the bucket means.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TStrStrPrMeanAccumulatorMap = std::map<TStrStrPr, TMeanAccumulator>;
    using TStrStrPrMeanAccumulatorMapCItr = TStrStrPrMeanAccumulatorMap::const_iterator;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CMetricPopulationModelFactory factory(params);
    TFeatureVec features(1u, model_t::E_PopulationMeanByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);
    CPPUNIT_ASSERT(gatherer.isPopulation());

    TStrStrPrMeanAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart+bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            CPPUNIT_ASSERT_EQUAL(features.size(), tmp.size());
            CPPUNIT_ASSERT_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec &data = tmp[0].second;
            CPPUNIT_ASSERT_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap means;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                if (data[j].second.s_BucketValue)
                {
                    means[TStrStrPr(gatherer.personName(data[j].first.first),
                                    gatherer.attributeName(data[j].first.second))] =
                                            data[j].second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMeans;
            for (TStrStrPrMeanAccumulatorMapCItr itr = accumulators.begin();
                 itr != accumulators.end();
                 ++itr)
            {
                expectedMeans[itr->first] = maths::CBasicStatistics::mean(itr->second);
            }

            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMeans),
                                 core::CContainerPrinter::print(means));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)].add(messages[i].s_Value);
    }
}

void CMetricPopulationDataGathererTest::testMin(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testMin ***");

    // Test that we correctly sample the bucket minimums.

    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u>;
    using TStrStrPrMinAccumulatorMap = std::map<TStrStrPr, TMinAccumulator>;
    using TStrStrPrMinAccumulatorMapCItr = TStrStrPrMinAccumulatorMap::const_iterator;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CMetricPopulationModelFactory factory(params);
    TFeatureVec features(1u, model_t::E_PopulationMinByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);

    TStrStrPrMinAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart+bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            CPPUNIT_ASSERT_EQUAL(features.size(), tmp.size());
            CPPUNIT_ASSERT_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec &data = tmp[0].second;
            CPPUNIT_ASSERT_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap mins;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                if (data[j].second.s_BucketValue)
                {
                    mins[TStrStrPr(gatherer.personName(data[j].first.first),
                                   gatherer.attributeName(data[j].first.second))] =
                                           data[j].second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMins;
            for (TStrStrPrMinAccumulatorMapCItr itr = accumulators.begin();
                 itr != accumulators.end();
                 ++itr)
            {
                expectedMins[itr->first] = itr->second[0];
            }

            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMins),
                                 core::CContainerPrinter::print(mins));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)].add(messages[i].s_Value);
    }
}

void CMetricPopulationDataGathererTest::testMax(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testMax ***");

    // Test that we correctly sample the bucket maximums.

    using TMaxAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>>;
    using TStrStrPrMaxAccumulatorMap = std::map<TStrStrPr, TMaxAccumulator>;
    using TStrStrPrMaxAccumulatorMapCItr = TStrStrPrMaxAccumulatorMap::const_iterator;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CMetricPopulationModelFactory factory(params);
    TFeatureVec features(1u, model_t::E_PopulationMaxByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);

    TStrStrPrMaxAccumulatorMap accumulators;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart+bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            CPPUNIT_ASSERT_EQUAL(features.size(), tmp.size());
            CPPUNIT_ASSERT_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec &data = tmp[0].second;
            CPPUNIT_ASSERT_EQUAL(accumulators.size(), data.size());
            TStrStrPrDoubleMap maxs;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                if (data[j].second.s_BucketValue)
                {
                    maxs[TStrStrPr(gatherer.personName(data[j].first.first),
                                   gatherer.attributeName(data[j].first.second))] =
                                           data[j].second.s_BucketValue->value()[0];
                }
            }

            TStrStrPrDoubleMap expectedMaxs;
            for (TStrStrPrMaxAccumulatorMapCItr itr = accumulators.begin();
                 itr != accumulators.end();
                 ++itr)
            {
                expectedMaxs[itr->first] = itr->second[0];
            }

            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMaxs),
                                 core::CContainerPrinter::print(maxs));

            bucketStart += bucketLength;
            accumulators.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        accumulators[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)].add(messages[i].s_Value);
    }
}

void CMetricPopulationDataGathererTest::testSum(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testSum ***");

    // Test that we correctly sample the bucket sums.

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CMetricPopulationModelFactory factory(params);
    TFeatureVec features(1u, model_t::E_PopulationSumByBucketPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);

    TStrStrPrDoubleMap expectedSums;
    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart+bucketLength << ")");

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            CPPUNIT_ASSERT_EQUAL(features.size(), tmp.size());
            CPPUNIT_ASSERT_EQUAL(features[0], tmp[0].first);

            const TSizeSizePrFeatureDataPrVec &data = tmp[0].second;
            CPPUNIT_ASSERT_EQUAL(expectedSums.size(), data.size());
            TStrStrPrDoubleMap sums;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                if (data[j].second.s_BucketValue)
                {
                    sums[TStrStrPr(gatherer.personName(data[j].first.first),
                                   gatherer.attributeName(data[j].first.second))] =
                                            data[j].second.s_BucketValue->value()[0];
                }
            }

            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSums),
                                 core::CContainerPrinter::print(sums));

            bucketStart += bucketLength;
            expectedSums.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        expectedSums[TStrStrPr(messages[i].s_Person, messages[i].s_Attribute)] += messages[i].s_Value;
    }
}


void CMetricPopulationDataGathererTest::testSampleCount(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testSampleCount ***");

    // Test that we set sensible sample counts for each attribute.

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    const std::string attribute("c1");
    const std::string person("p1");
    const std::size_t numberBuckets = 40;
    const std::size_t personMessageCount[numberBuckets] =
        {
            11,  11,  11,  11,  110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
            110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
            110, 97,  97,  97,  97,  97,  97,  97,  97,  97
        };
    const double expectedSampleCounts[] =
        {
            0.0,     0.0,     0.0,     11.0,   11.0,    11.0,   11.0,    11.0,    11.0,    11.0,
            11.0,    11.0,    11.0,    11.0,   11.0,    11.0,   11.0,    11.0,    11.0,    11.0,
            11.0,    11.0,    11.0,    11.0,   11.0,    11.0,   11.0,    11.0,    11.0,    11.0,
            11.3597, 11.7164, 12.0701, 12.421, 12.7689, 13.114, 13.4562, 13.7957, 14.1325, 14.4665
        };
    const double tolerance = 5e-4;

    TMessageVec messages;
    for (std::size_t bucket = 0u; bucket < numberBuckets; ++bucket)
    {
        core_t::TTime bucketStart =
                startTime + static_cast<core_t::TTime>(bucket) * bucketLength;

        std::size_t n = personMessageCount[bucket];
        for (std::size_t i = 0u; i < n; ++i)
        {
            core_t::TTime time = bucketStart + bucketLength
                                               * static_cast<core_t::TTime>(i)
                                               / static_cast<core_t::TTime>(n);
            messages.push_back(SMessage(time, person, attribute, 1.0));
        }
    }

    CMetricPopulationModelFactory factory(params);
    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);

    std::size_t bucket = 0u;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        core_t::TTime bucketStart =
                startTime + static_cast<core_t::TTime>(bucket) * bucketLength;

        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            gatherer.sampleNow(bucketStart);
            LOG_DEBUG(gatherer.effectiveSampleCount(0));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedSampleCounts[bucket],
                                         gatherer.effectiveSampleCount(0),
                                         tolerance);
            ++bucket;
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    core_t::TTime bucketStart =
            startTime + static_cast<core_t::TTime>(bucket) * bucketLength;
    gatherer.sampleNow(bucketStart);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedSampleCounts[bucket],
                                 gatherer.effectiveSampleCount(0),
                                 tolerance);
}

void CMetricPopulationDataGathererTest::testFeatureData(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testFeatureData ***");

    // Test we correctly sample the mean, minimum and maximum statistics.

    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TStrStrPrMeanAccumulatorMap = std::map<TStrStrPr, TMeanAccumulator>;
    using TStrStrPrMeanAccumulatorMapCItr = TStrStrPrMeanAccumulatorMap::const_iterator;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u>;
    using TStrStrPrMinAccumulatorMap = std::map<TStrStrPr, TMinAccumulator>;
    using TStrStrPrMinAccumulatorMapCItr = TStrStrPrMinAccumulatorMap::const_iterator;
    using TMaxAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>>;
    using TStrStrPrMaxAccumulatorMap = std::map<TStrStrPr, TMaxAccumulator>;
    using TStrStrPrMaxAccumulatorMapCItr = TStrStrPrMaxAccumulatorMap::const_iterator;
    using TStrStrPrDoubleVecMap = std::map<TStrStrPr, TDoubleVec>;

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CMetricPopulationModelFactory factory(params);
    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);

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
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart+bucketLength << ")");

            gatherer.sampleNow(bucketStart);

            TFeatureSizeSizePrFeatureDataPrVecPrVec tmp;
            gatherer.featureData(bucketStart, bucketLength, tmp);
            CPPUNIT_ASSERT_EQUAL(static_cast<std::size_t>(3), tmp.size());

            CPPUNIT_ASSERT_EQUAL(model_t::E_PopulationMeanByPersonAndAttribute, tmp[0].first);
            TStrStrPrDoubleMap means;
            TStrStrPrDoubleVecMap meanSamples;
            for (std::size_t j = 0u; j < tmp[0].second.size(); ++j)
            {
                const TSizeSizePrFeatureDataPr &data = tmp[0].second[j];
                TStrStrPr key(gatherer.personName(data.first.first),
                              gatherer.attributeName(data.first.second));
                if (data.second.s_BucketValue)
                {
                    means[key] = data.second.s_BucketValue->value()[0];
                }
                TDoubleVec &samples = meanSamples[key];
                for (std::size_t k = 0u;
                     k < boost::unwrap_ref(data.second.s_Samples).size();
                     ++k)
                {
                    samples.push_back(boost::unwrap_ref(data.second.s_Samples)[k].value()[0]);
                }
            }

            CPPUNIT_ASSERT_EQUAL(model_t::E_PopulationMinByPersonAndAttribute, tmp[1].first);
            TStrStrPrDoubleMap mins;
            TStrStrPrDoubleVecMap minSamples;
            for (std::size_t j = 0u; j < tmp[1].second.size(); ++j)
            {
                const TSizeSizePrFeatureDataPr &data = tmp[1].second[j];
                TStrStrPr key(gatherer.personName(data.first.first),
                              gatherer.attributeName(data.first.second));
                if (data.second.s_BucketValue)
                {
                    mins[key] = data.second.s_BucketValue->value()[0];
                }
                TDoubleVec &samples = minSamples[key];
                for (std::size_t k = 0u;
                     k < boost::unwrap_ref(data.second.s_Samples).size();
                     ++k)
                {
                    samples.push_back(boost::unwrap_ref(data.second.s_Samples)[k].value()[0]);
                }
            }

            CPPUNIT_ASSERT_EQUAL(model_t::E_PopulationMaxByPersonAndAttribute, tmp[2].first);
            TStrStrPrDoubleMap maxs;
            TStrStrPrDoubleVecMap maxSamples;
            for (std::size_t j = 0u; j < tmp[2].second.size(); ++j)
            {
                const TSizeSizePrFeatureDataPr &data = tmp[2].second[j];
                TStrStrPr key(gatherer.personName(data.first.first),
                              gatherer.attributeName(data.first.second));
                if (data.second.s_BucketValue)
                {
                    maxs[key] = data.second.s_BucketValue->value()[0];
                }
                TDoubleVec &samples = maxSamples[key];
                for (std::size_t k = 0u;
                     k < boost::unwrap_ref(data.second.s_Samples).size();
                     ++k)
                {
                    samples.push_back(boost::unwrap_ref(data.second.s_Samples)[k].value()[0]);
                }
            }

            TStrStrPrDoubleMap expectedMeans;
            for (TStrStrPrMeanAccumulatorMapCItr itr = bucketMeanAccumulators.begin();
                 itr != bucketMeanAccumulators.end();
                 ++itr)
            {
                expectedMeans[itr->first] = maths::CBasicStatistics::mean(itr->second);
            }

            TStrStrPrDoubleMap expectedMins;
            for (TStrStrPrMinAccumulatorMapCItr itr = bucketMinAccumulators.begin();
                 itr != bucketMinAccumulators.end();
                 ++itr)
            {
                expectedMins[itr->first] = itr->second[0];
            }

            TStrStrPrDoubleMap expectedMaxs;
            for (TStrStrPrMaxAccumulatorMapCItr itr = bucketMaxAccumulators.begin();
                 itr != bucketMaxAccumulators.end();
                 ++itr)
            {
                expectedMaxs[itr->first] = itr->second[0];
            }

            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMeans),
                                 core::CContainerPrinter::print(means));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMins),
                                 core::CContainerPrinter::print(mins));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMaxs),
                                 core::CContainerPrinter::print(maxs));

            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMeanSamples),
                                 core::CContainerPrinter::print(meanSamples));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMinSamples),
                                 core::CContainerPrinter::print(minSamples));
            CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedMaxSamples),
                                 core::CContainerPrinter::print(maxSamples));

            bucketStart += bucketLength;
            bucketMeanAccumulators.clear();
            expectedMeanSamples.clear();
            bucketMinAccumulators.clear();
            expectedMinSamples.clear();
            bucketMaxAccumulators.clear();
            expectedMaxSamples.clear();
        }

        addArrival(messages[i], gatherer, m_ResourceMonitor);
        TStrStrPr key(messages[i].s_Person, messages[i].s_Attribute);

        bucketMeanAccumulators[key].add(messages[i].s_Value);
        bucketMinAccumulators[key].add(messages[i].s_Value);
        bucketMaxAccumulators[key].add(messages[i].s_Value);
        expectedMeanSamples.insert(TStrStrPrDoubleVecMap::value_type(key, TDoubleVec()));
        expectedMinSamples.insert(TStrStrPrDoubleVecMap::value_type(key, TDoubleVec()));
        expectedMaxSamples.insert(TStrStrPrDoubleVecMap::value_type(key, TDoubleVec()));

        std::size_t cid;
        CPPUNIT_ASSERT(gatherer.attributeId(messages[i].s_Attribute, cid));

        double sampleCount = gatherer.effectiveSampleCount(cid);
        if (sampleCount > 0.0)
        {
            sampleMeanAccumulators[key].add(messages[i].s_Value);
            sampleMinAccumulators[key].add(messages[i].s_Value);
            sampleMaxAccumulators[key].add(messages[i].s_Value);
            if (maths::CBasicStatistics::count(sampleMeanAccumulators[key])
                                               == std::floor(sampleCount + 0.5))
            {
                expectedMeanSamples[key].push_back(
                        maths::CBasicStatistics::mean(sampleMeanAccumulators[key]));
                expectedMinSamples[key].push_back(sampleMinAccumulators[key][0]);
                expectedMaxSamples[key].push_back(sampleMaxAccumulators[key][0]);
                sampleMeanAccumulators[key] = TMeanAccumulator();
                sampleMinAccumulators[key] = TMinAccumulator();
                sampleMaxAccumulators[key] = TMaxAccumulator();
            }
        }
    }
}

void CMetricPopulationDataGathererTest::testRemovePeople(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testRemovePeople ***");

    // Check that all the state is correctly updated when some
    // people are removed.

    using TSizeVec = std::vector<std::size_t>;
    using TStrVec = std::vector<std::string>;
    using TSizeUInt64Pr = std::pair<std::size_t, uint64_t>;
    using TSizeUInt64PrVec = std::vector<TSizeUInt64Pr>;
    using TStrFeatureDataPr = std::pair<std::string, SMetricFeatureData>;
    using TStrFeatureDataPrVec = std::vector<TStrFeatureDataPr>;
    using TStrSizeMap = std::map<std::string, uint64_t>;

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
    CDataGatherer gatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           TStrVec(), false, searchKey, features, startTime, 0);

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart + bucketLength << ")");
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
    CPPUNIT_ASSERT_EQUAL(numberPeople, gatherer.numberOverFieldValues());
    TStrVec expectedPersonNames;
    TSizeVec expectedPersonIds;
    for (std::size_t i = 0u; i < numberPeople; ++i)
    {
        if (!std::binary_search(peopleToRemove.begin(), peopleToRemove.end(), i))
        {
            expectedPersonNames.push_back(gatherer.personName(i));
            expectedPersonIds.push_back(i);
        }
        else
        {
            LOG_DEBUG("Removing " << gatherer.personName(i));
        }
    }

    TStrSizeMap expectedNonZeroCounts;
    {
        TSizeUInt64PrVec nonZeroCounts;
        gatherer.personNonZeroCounts(bucketStart, nonZeroCounts);
        for (std::size_t i = 0u; i < nonZeroCounts.size(); ++i)
        {
            if (!std::binary_search(peopleToRemove.begin(),
                                    peopleToRemove.end(),
                                    nonZeroCounts[i].first))
            {
                const std::string &name = gatherer.personName(nonZeroCounts[i].first);
                expectedNonZeroCounts[name] = nonZeroCounts[i].second;
            }
        }
    }
    LOG_DEBUG("expectedNonZeroCounts = "
              << core::CContainerPrinter::print(expectedNonZeroCounts));

    LOG_DEBUG("Expected");
    TStrFeatureDataPrVec expectedFeatureData;
    {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i)
        {
            const TSizeSizePrFeatureDataPrVec &data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                if (!std::binary_search(peopleToRemove.begin(),
                                        peopleToRemove.end(),
                                        data[j].first.first))
                {
                    std::string key = model_t::print(featureData[i].first)
                                      + " " + gatherer.personName(data[j].first.first)
                                      + " " + gatherer.attributeName(data[j].first.second);
                    expectedFeatureData.push_back(TStrFeatureDataPr(key, data[j].second));
                    LOG_DEBUG("  " << key);
                    LOG_DEBUG("  " << data[j].second.print());
                }
            }
        }
    }

    gatherer.recyclePeople(peopleToRemove);

    CPPUNIT_ASSERT_EQUAL(numberPeople - peopleToRemove.size(),
                         gatherer.numberActivePeople());
    for (std::size_t i = 0u; i < expectedPersonNames.size(); ++i)
    {
        std::size_t pid;
        CPPUNIT_ASSERT(gatherer.personId(expectedPersonNames[i], pid));
        CPPUNIT_ASSERT_EQUAL(expectedPersonIds[i], pid);
    }

    TStrSizeMap actualNonZeroCounts;
    TSizeUInt64PrVec nonZeroCounts;
    gatherer.personNonZeroCounts(bucketStart, nonZeroCounts);
    for (std::size_t i = 0u; i < nonZeroCounts.size(); ++i)
    {
        const std::string &name = gatherer.personName(nonZeroCounts[i].first);
        actualNonZeroCounts[name] = nonZeroCounts[i].second;
    }
    LOG_DEBUG("actualNonZeroCounts = "
              << core::CContainerPrinter::print(actualNonZeroCounts));

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedNonZeroCounts),
                         core::CContainerPrinter::print(actualNonZeroCounts));

    LOG_DEBUG("Actual");
    TStrFeatureDataPrVec actualFeatureData;
    {
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i)
        {
            const TSizeSizePrFeatureDataPrVec &data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                std::string key = model_t::print(featureData[i].first)
                                  + " " + gatherer.personName(data[j].first.first)
                                  + " " + gatherer.attributeName(data[j].first.second);
                actualFeatureData.push_back(TStrFeatureDataPr(key, data[j].second));
                LOG_DEBUG("  " << key);
                LOG_DEBUG("    " << data[j].second.print());
            }
        }
    }

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedFeatureData),
                         core::CContainerPrinter::print(actualFeatureData));
}

void CMetricPopulationDataGathererTest::testRemoveAttributes(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testRemoveAttributes ***");

    // Check that all the state is correctly updated when some
    // attributes are removed.

    using TSizeVec = std::vector<std::size_t>;
    using TStrVec = std::vector<std::string>;
    using TStrFeatureDataPr = std::pair<std::string, SMetricFeatureData>;
    using TStrFeatureDataPrVec = std::vector<TStrFeatureDataPr>;

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationSumByBucketPersonAndAttribute);
    CDataGatherer gatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           TStrVec(), false, searchKey, features, startTime, 0);

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart + bucketLength << ")");
            gatherer.sampleNow(bucketStart);
            bucketStart += bucketLength;
        }
        addArrival(messages[i], gatherer, m_ResourceMonitor);
    }

    // Remove attributes 0, 2, 3, 9 (i.e. last).
    TSizeVec attributesToRemove;
    attributesToRemove.push_back(0u);
    attributesToRemove.push_back(2u);
    attributesToRemove.push_back(3u);
    attributesToRemove.push_back(9u);

    std::size_t numberAttributes = gatherer.numberActiveAttributes();
    CPPUNIT_ASSERT_EQUAL(numberAttributes, gatherer.numberByFieldValues());
    TStrVec expectedAttributeNames;
    TSizeVec expectedAttributeIds;
    TDoubleVec expectedSampleCounts;
    for (std::size_t i = 0u; i < numberAttributes; ++i)
    {
        if (!std::binary_search(attributesToRemove.begin(),
                                attributesToRemove.end(), i))
        {
            expectedAttributeNames.push_back(gatherer.attributeName(i));
            expectedAttributeIds.push_back(i);
            expectedSampleCounts.push_back(gatherer.effectiveSampleCount(i));
        }
        else
        {
            LOG_DEBUG("Removing " << gatherer.attributeName(i));
        }
    }

    std::string expectedFeatureData;
    {
        LOG_DEBUG("Expected");
        TStrFeatureDataPrVec expected;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i)
        {
            const TSizeSizePrFeatureDataPrVec &data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                if (!std::binary_search(attributesToRemove.begin(),
                                        attributesToRemove.end(),
                                        data[j].first.second))
                {
                    std::string key = model_t::print(featureData[i].first)
                                      + " " + gatherer.personName(data[j].first.first)
                                      + " " + gatherer.attributeName(data[j].first.second);
                    expected.push_back(TStrFeatureDataPr(key, data[j].second));
                    LOG_DEBUG("  " << key);
                    LOG_DEBUG("    " << data[j].second.print());
                }
            }
        }
        expectedFeatureData = core::CContainerPrinter::print(expected);
    }

    gatherer.recycleAttributes(attributesToRemove);

    CPPUNIT_ASSERT_EQUAL(numberAttributes - attributesToRemove.size(),
                         gatherer.numberActiveAttributes());
    for (std::size_t i = 0u; i < expectedAttributeNames.size(); ++i)
    {
        std::size_t cid;
        CPPUNIT_ASSERT(gatherer.attributeId(expectedAttributeNames[i], cid));
        CPPUNIT_ASSERT_EQUAL(expectedAttributeIds[i], cid);
    }

    numberAttributes = gatherer.numberActiveAttributes();
    TDoubleVec actualSampleCounts;
    for (std::size_t i = 0u; i < numberAttributes; ++i)
    {
        actualSampleCounts.push_back(gatherer.effectiveSampleCount(expectedAttributeIds[i]));
    }
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expectedSampleCounts),
                         core::CContainerPrinter::print(actualSampleCounts));

    std::string actualFeatureData;
    {
        LOG_DEBUG("Actual");
        TStrFeatureDataPrVec actual;
        TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
        gatherer.featureData(bucketStart, bucketLength, featureData);
        for (std::size_t i = 0u; i < featureData.size(); ++i)
        {
            const TSizeSizePrFeatureDataPrVec &data = featureData[i].second;
            for (std::size_t j = 0u; j < data.size(); ++j)
            {
                std::string key = model_t::print(featureData[i].first)
                                  + " " + gatherer.personName(data[j].first.first)
                                  + " " + gatherer.attributeName(data[j].first.second);
                actual.push_back(TStrFeatureDataPr(key, data[j].second));
                LOG_DEBUG("  " << key);
                LOG_DEBUG("    " << data[j].second.print());
            }
        }
        actualFeatureData = core::CContainerPrinter::print(actual);
    }

    CPPUNIT_ASSERT_EQUAL(expectedFeatureData, actualFeatureData);
}

void CMetricPopulationDataGathererTest::testInfluenceStatistics(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testInfluenceStatistics ***");

    using TDoubleDoublePr = std::pair<double, double>;
    using TStrDoubleDoublePrPr = std::pair<std::string, TDoubleDoublePr>;
    using TStrDoubleDoublePrPrVec = std::vector<TStrDoubleDoublePrPr>;

    const core_t::TTime startTime = 0;
    const core_t::TTime bucketLength = 600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    std::string influencerNames_[]    = { "i1", "i2" };
    std::string influencerValues[][3] =
        {
            { "i11", "i12", "i13" },
            { "i21", "i22", "i23" }
        };

    SMessage data[] =
        {
            SMessage(1,    "p1", "",  1.0, vec(influencerValues[0][0], influencerValues[1][0])), // Bucket 1
            SMessage(150,  "p1", "",  5.0, vec(influencerValues[0][1], influencerValues[1][1])),
            SMessage(150,  "p1", "",  3.0, vec(influencerValues[0][2], influencerValues[1][2])),
            SMessage(550,  "p2", "",  2.0, vec(influencerValues[0][0], influencerValues[1][0])),
            SMessage(551,  "p2", "",  2.1, vec(influencerValues[0][1], influencerValues[1][1])),
            SMessage(552,  "p2", "",  4.0, vec(influencerValues[0][2], influencerValues[1][2])),
            SMessage(554,  "p2", "",  2.2, vec(influencerValues[0][2], influencerValues[1][2])),
            SMessage(600,  "p1", "",  3.0, vec(influencerValues[0][1], influencerValues[1][0])), // Bucket 2
            SMessage(660,  "p2", "",  3.0, vec(influencerValues[0][0], influencerValues[1][2])),
            SMessage(690,  "p1", "",  7.3, vec(influencerValues[0][1], "")),
            SMessage(700,  "p2", "",  4.0, vec(influencerValues[0][0], influencerValues[1][2])),
            SMessage(800,  "p1", "",  2.2, vec(influencerValues[0][2], influencerValues[1][0])),
            SMessage(900,  "p2", "",  2.5, vec(influencerValues[0][1], influencerValues[1][0])),
            SMessage(1000, "p1", "",  5.0, vec(influencerValues[0][1], influencerValues[1][0])),
            SMessage(1200, "p2", "",  6.4, vec("",                     influencerValues[1][2])), // Bucket 3
            SMessage(1210, "p2", "",  6.0, vec("",                     influencerValues[1][2])),
            SMessage(1240, "p2", "",  7.0, vec("",                     influencerValues[1][1])),
            SMessage(1600, "p2", "", 11.0, vec("",                     influencerValues[1][0])),
            SMessage(1800, "p1", "", 11.0, vec("",                     ""))                      // Sentinel
        };

    std::string expectedStatistics[] =
        {
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
            "[(i21, (11, 1)), (i22, (7, 1)), (i23, (12.4, 1))]"
        };
    const std::string *expected = expectedStatistics;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationHighSumByBucketPersonAndAttribute);
    TStrVec influencerNames(boost::begin(influencerNames_),
                            boost::end(influencerNames_));
    CDataGatherer gatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                           EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                           influencerNames, false, searchKey, features, startTime, 2u);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u, b = 0u; i < boost::size(data); ++i)
    {
        if (data[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("*** processing bucket ***");
            TFeatureSizeSizePrFeatureDataPrVecPrVec featureData;
            gatherer.featureData(bucketStart, bucketLength, featureData);
            for (std::size_t j = 0u; j < featureData.size(); ++j)
            {
                model_t::EFeature feature = featureData[j].first;
                LOG_DEBUG("feature = " << model_t::print(feature));

                const TSizeSizePrFeatureDataPrVec &data_ = featureData[j].second;
                for (std::size_t k = 0u; k < data_.size(); ++k)
                {
                    TStrDoubleDoublePrPrVec statistics;
                    for (std::size_t m = 0u;
                         m < data_[k].second.s_InfluenceValues.size();
                         ++m)
                    {
                        for (std::size_t n = 0u;
                             n < data_[k].second.s_InfluenceValues[m].size();
                             ++n)
                        {
                            statistics.push_back(TStrDoubleDoublePrPr(
                                    data_[k].second.s_InfluenceValues[m][n].first,
                                    TDoubleDoublePr(data_[k].second.s_InfluenceValues[m][n].second.first[0],
                                                    data_[k].second.s_InfluenceValues[m][n].second.second)));
                        }
                    }
                    std::sort(statistics.begin(),
                              statistics.end(),
                              maths::COrderings::SFirstLess());

                    LOG_DEBUG("statistics = "
                              << core::CContainerPrinter::print(statistics));
                    LOG_DEBUG("expected   = " << *expected);
                    CPPUNIT_ASSERT_EQUAL(*(expected++),
                                         core::CContainerPrinter::print(statistics));
                }
            }

            bucketStart += bucketLength; ++b;
        }
        addArrival(data[i], gatherer, m_ResourceMonitor);
    }
}

void CMetricPopulationDataGathererTest::testPersistence(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testPersistence ***");

    const core_t::TTime startTime = 1367280000;
    const core_t::TTime bucketLength = 3600;
    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    TFeatureVec features;
    features.push_back(model_t::E_PopulationMeanByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMinByPersonAndAttribute);
    features.push_back(model_t::E_PopulationMaxByPersonAndAttribute);
    features.push_back(model_t::E_PopulationHighSumByBucketPersonAndAttribute);
    CDataGatherer origDataGatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                                   EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                   TStrVec(), false, searchKey, features, startTime, 0);

    TMessageVec messages;
    generateTestMessages(startTime, messages);

    core_t::TTime bucketStart = startTime;
    for (std::size_t i = 0u; i < messages.size(); ++i)
    {
        if (messages[i].s_Time >= bucketStart + bucketLength)
        {
            LOG_DEBUG("Processing bucket [" << bucketStart
                      << ", " << bucketStart + bucketLength << ")");
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
    //LOG_DEBUG("origXml = " << origXml);
    LOG_DEBUG("origXml length = " << origXml.length()
              << ", # tabs " << std::count_if(origXml.begin(), origXml.end(), isSpace));

    std::size_t length = origXml.length()
                         - std::count_if(origXml.begin(), origXml.end(), isSpace);
    CPPUNIT_ASSERT(length < 645000);

    // Restore the XML into a new data gatherer
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    CDataGatherer restoredDataGatherer(model_t::E_PopulationMetric, model_t::E_None, params,
                                       EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                       TStrVec(), false, searchKey, traverser);

    // The XML representation of the new data gatherer should be the same as the
    // original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredDataGatherer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    //LOG_DEBUG("newXml = " << newXml);
    LOG_DEBUG("newXml length = " << newXml.length()
              << ", # tabs " << std::count_if(newXml.begin(), newXml.end(), isSpace));

    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CMetricPopulationDataGathererTest::testReleaseMemory(void)
{
    LOG_DEBUG("*** CMetricPopulationDataGathererTest::testReleaseMemory ***");

    const core_t::TTime startTime = 1373932800;
    const core_t::TTime bucketLength = 3600;

    SModelParams params(bucketLength);
    params.s_LatencyBuckets = 3;
    CMetricPopulationModelFactory factory(params);
    TFeatureVec features(1u, model_t::E_PopulationMeanByPersonAndAttribute);
    factory.features(features);
    CModelFactory::SGathererInitializationData initData(startTime);
    CModelFactory::TDataGathererPtr gathererPtr(factory.makeDataGatherer(initData));
    CDataGatherer &gatherer(*gathererPtr);
    CPPUNIT_ASSERT(gatherer.isPopulation());

    core_t::TTime bucketStart = startTime;
    // Add a few buckets with count of 10 so that sample count gets estimated
    for (std::size_t i = 0; i < 10; ++i)
    {
        // Add 10 events
        for (std::size_t j = 0; j < 10; ++j)
        {
            addArrival(SMessage(bucketStart, "p1", "", 10.0), gatherer, m_ResourceMonitor);
            addArrival(SMessage(bucketStart, "p2", "", 10.0), gatherer, m_ResourceMonitor);
        }
        gatherer.sampleNow(bucketStart - params.s_LatencyBuckets * bucketLength);
        bucketStart += bucketLength;
    }

    // Add a bucket with not enough data to sample for p2
    for (std::size_t j = 0; j < 10; ++j)
    {
        addArrival(SMessage(bucketStart, "p1", "", 10.0), gatherer, m_ResourceMonitor);
    }
    addArrival(SMessage(bucketStart, "p2", "", 10.0), gatherer, m_ResourceMonitor);
    gatherer.sampleNow(bucketStart - params.s_LatencyBuckets * bucketLength);
    bucketStart += bucketLength;

    std::size_t mem = gatherer.memoryUsage();

    // Add 48 + 1 buckets ( > 2 days) to force incomplete samples out of consideration for p2
    for (std::size_t i = 0; i < 49 + 1; ++i)
    {
        for (std::size_t j = 0; j < 10; ++j)
        {
            addArrival(SMessage(bucketStart, "p1", "", 10.0), gatherer, m_ResourceMonitor);
        }
        gatherer.sampleNow(bucketStart - params.s_LatencyBuckets * bucketLength);
        bucketStart += bucketLength;
        gatherer.releaseMemory(bucketStart - params.s_SamplingAgeCutoff);
        if (i <= 40)
        {
            CPPUNIT_ASSERT(gatherer.memoryUsage() >= mem - 1000);
        }
    }
    CPPUNIT_ASSERT(gatherer.memoryUsage() < mem - 1000);
}

CppUnit::Test *CMetricPopulationDataGathererTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMetricPopulationDataGathererTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testMean",
                                   &CMetricPopulationDataGathererTest::testMean) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testMin",
                                   &CMetricPopulationDataGathererTest::testMin) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testMax",
                                   &CMetricPopulationDataGathererTest::testMax) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testSum",
                                   &CMetricPopulationDataGathererTest::testSum) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testSampleCount",
                                   &CMetricPopulationDataGathererTest::testSampleCount) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testFeatureData",
                                   &CMetricPopulationDataGathererTest::testFeatureData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testRemovePeople",
                                   &CMetricPopulationDataGathererTest::testRemovePeople) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testRemoveAttributes",
                                   &CMetricPopulationDataGathererTest::testRemoveAttributes) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testInfluenceStatistics",
                                   &CMetricPopulationDataGathererTest::testInfluenceStatistics) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testPersistence",
                                   &CMetricPopulationDataGathererTest::testPersistence) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CMetricPopulationDataGathererTest>(
                                   "CMetricPopulationDataGathererTest::testReleaseMemory",
                                   &CMetricPopulationDataGathererTest::testReleaseMemory) );

    return suiteOfTests;
}
