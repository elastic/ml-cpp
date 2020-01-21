/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMemory.h>

#include <maths/CBasicStatistics.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CEventRateModel.h>
#include <model/CEventRateModelFactory.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CMetricModel.h>
#include <model/CMetricModelFactory.h>
#include <model/CResourceMonitor.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <memory>

BOOST_AUTO_TEST_SUITE(CModelMemoryTest)

using namespace ml;
using namespace model;

namespace {

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

std::size_t addPerson(const std::string& p, const CModelFactory::TDataGathererPtr& gatherer) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    person.resize(gatherer->fieldsOfInterest().size(), nullptr);
    CEventData result;
    CResourceMonitor resourceMonitor;
    gatherer->processFields(person, result, resourceMonitor);
    return *result.personId();
}

void addArrival(CDataGatherer& gatherer, core_t::TTime time, const std::string& person) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    CEventData eventData;
    eventData.time(time);
    CResourceMonitor resourceMonitor;
    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

void addArrival(CDataGatherer& gatherer, core_t::TTime time, const std::string& person, double value) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);
    std::string valueAsString(core::CStringUtils::typeToString(value));
    fieldValues.push_back(&valueAsString);
    CEventData eventData;
    eventData.time(time);
    CResourceMonitor resourceMonitor;
    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

const std::string EMPTY_STRING;
}

BOOST_AUTO_TEST_CASE(testOnlineEventRateModel) {

    // Tests to check that the memory usage of the model goes up
    // as data is fed in and that memoryUsage and debugMemory are
    // consistent.

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CEventRateModelFactory factory(params, interimBucketCorrector);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
    factory.features(features);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
    BOOST_REQUIRE_EQUAL(std::size_t(0), addPerson("p", gatherer));
    CModelFactory::TModelPtr modelPtr(factory.makeModel(gatherer));
    BOOST_TEST_REQUIRE(modelPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_EventRateOnline, modelPtr->category());
    CEventRateModel& model = static_cast<CEventRateModel&>(*modelPtr.get());
    std::size_t startMemoryUsage = model.memoryUsage();
    CResourceMonitor resourceMonitor;

    LOG_DEBUG(<< "Memory used by model: " << model.memoryUsage());

    core_t::TTime time = startTime;
    for (std::size_t bucketCounts : {5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6}) {
        for (std::size_t j = 0; j < bucketCounts; ++j) {
            addArrival(*gatherer, time + static_cast<core_t::TTime>(j), "p");
        }
        model.sample(time, time + bucketLength, resourceMonitor);

        time += bucketLength;
    }
    LOG_INFO(<< "Sizeof model now: " << model.memoryUsage());
    BOOST_TEST_REQUIRE(model.memoryUsage() > startMemoryUsage);

    auto memoryUsage = std::make_shared<core::CMemoryUsage>();
    model.debugMemoryUsage(memoryUsage);
    LOG_DEBUG(<< "Debug sizeof model: " << memoryUsage->usage());
    BOOST_REQUIRE_EQUAL(model.computeMemoryUsage(), memoryUsage->usage());
}

BOOST_AUTO_TEST_CASE(testOnlineMetricModel) {

    // Tests to check that the memory usage of the model goes up
    // as data is fed in and that memoryUsage and debugMemory are
    // consistent.

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CMetricModelFactory factory(params, interimBucketCorrector);

    double mean = 5.0;
    double variance = 2.0;
    std::size_t anomalousBucket = 12u;
    double anomaly = 5 * std::sqrt(variance);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    factory.features(features);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(startTime));
    BOOST_REQUIRE_EQUAL(std::size_t(0), addPerson("p", gatherer));
    CModelFactory::TModelPtr modelPtr(factory.makeModel(gatherer));
    BOOST_TEST_REQUIRE(modelPtr);
    BOOST_REQUIRE_EQUAL(model_t::E_MetricOnline, modelPtr->category());
    CMetricModel& model = static_cast<CMetricModel&>(*modelPtr.get());
    std::size_t startMemoryUsage = model.memoryUsage();
    CResourceMonitor resourceMonitor;

    LOG_DEBUG(<< "Memory used by model: " << model.memoryUsage() << " / "
              << core::CMemory::dynamicSize(model));

    test::CRandomNumbers rng;

    TSizeVec bucketCounts{5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6};

    core_t::TTime time = startTime;
    for (std::size_t i = 0; i < bucketCounts.size(); ++i) {
        TDoubleVec values;
        rng.generateNormalSamples(mean, variance, bucketCounts[i], values);

        for (std::size_t j = 0; j < values.size(); ++j) {
            addArrival(*gatherer, time + static_cast<core_t::TTime>(j), "p",
                       values[j] + (i == anomalousBucket ? anomaly : 0.0));
        }
        model.sample(time, time + bucketLength, resourceMonitor);

        time += bucketLength;
    }
    LOG_INFO(<< "Sizeof model now: " << model.memoryUsage());
    BOOST_TEST_REQUIRE(model.memoryUsage() > startMemoryUsage);

    auto memoryUsage = std::make_shared<core::CMemoryUsage>();
    model.debugMemoryUsage(memoryUsage);
    LOG_DEBUG(<< "Debug sizeof model: " << memoryUsage->usage());
    BOOST_REQUIRE_EQUAL(model.computeMemoryUsage(), memoryUsage->usage());
}

BOOST_AUTO_TEST_SUITE_END()
