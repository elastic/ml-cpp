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

#include "CModelMemoryTest.h"

#include <core/CMemory.h>

#include <maths/CBasicStatistics.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CEventRateModel.h>
#include <model/CEventRateModelFactory.h>
#include <model/CMetricModel.h>
#include <model/CMetricModelFactory.h>
#include <model/CResourceMonitor.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

using namespace ml;
using namespace model;

namespace {

typedef std::vector<double> TDoubleVec;

std::size_t addPerson(const std::string& p, const CModelFactory::TDataGathererPtr& gatherer) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    person.resize(gatherer->fieldsOfInterest().size(), 0);
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

void CModelMemoryTest::testOnlineEventRateModel(void) {
    // Tests to check that the memory usage of the model goes up
    // as data is fed in and that memoryUsage and debugMemory are
    // consistent.

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CEventRateModelFactory factory(params);

    std::size_t bucketCounts[] = {5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6};

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualCountByBucketAndPerson);
    features.push_back(model_t::E_IndividualTotalBucketCountByPerson);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer));
    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelPtr(factory.makeModel(initData));
    CPPUNIT_ASSERT(modelPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_EventRateOnline, modelPtr->category());
    CEventRateModel& model = static_cast<CEventRateModel&>(*modelPtr.get());
    std::size_t startMemoryUsage = model.memoryUsage();
    CResourceMonitor resourceMonitor;

    LOG_DEBUG("Memory used by model: " << model.memoryUsage());

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        for (std::size_t j = 0u; j < bucketCounts[i]; ++j) {
            addArrival(*gatherer, time + static_cast<core_t::TTime>(j), "p");
        }
        model.sample(time, time + bucketLength, resourceMonitor);

        time += bucketLength;
    }
    LOG_INFO("Sizeof model now: " << model.memoryUsage());
    CPPUNIT_ASSERT(model.memoryUsage() > startMemoryUsage);

    core::CMemoryUsage memoryUsage;
    model.debugMemoryUsage(&memoryUsage);
    LOG_DEBUG("Debug sizeof model: " << memoryUsage.usage());
    CPPUNIT_ASSERT_EQUAL(model.computeMemoryUsage(), memoryUsage.usage());
}

void CModelMemoryTest::testOnlineMetricModel(void) {
    // Tests to check that the memory usage of the model goes up
    // as data is fed in and that memoryUsage and debugMemory are
    // consistent.

    core_t::TTime startTime(0);
    core_t::TTime bucketLength(10);
    SModelParams params(bucketLength);
    CMetricModelFactory factory(params);

    std::size_t bucketCounts[] = {5, 6, 3, 5, 0, 7, 8, 5, 4, 3, 5, 5, 6};

    double mean = 5.0;
    double variance = 2.0;
    std::size_t anomalousBucket = 12u;
    double anomaly = 5 * ::sqrt(variance);

    CDataGatherer::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    features.push_back(model_t::E_IndividualMinByPerson);
    features.push_back(model_t::E_IndividualMaxByPerson);
    factory.features(features);
    CModelFactory::SGathererInitializationData gathererInitData(startTime);
    CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererInitData));
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gatherer));
    CModelFactory::SModelInitializationData initData(gatherer);
    CAnomalyDetectorModel::TModelPtr modelPtr(factory.makeModel(initData));
    CPPUNIT_ASSERT(modelPtr);
    CPPUNIT_ASSERT_EQUAL(model_t::E_MetricOnline, modelPtr->category());
    CMetricModel& model = static_cast<CMetricModel&>(*modelPtr.get());
    std::size_t startMemoryUsage = model.memoryUsage();
    CResourceMonitor resourceMonitor;

    LOG_DEBUG("Memory used by model: " << model.memoryUsage() << " / " << core::CMemory::dynamicSize(model));

    test::CRandomNumbers rng;

    core_t::TTime time = startTime;
    for (std::size_t i = 0u; i < boost::size(bucketCounts); ++i) {
        TDoubleVec values;
        rng.generateNormalSamples(mean, variance, bucketCounts[i], values);

        for (std::size_t j = 0u; j < values.size(); ++j) {
            addArrival(*gatherer,
                       time + static_cast<core_t::TTime>(j),
                       "p",
                       values[j] + (i == anomalousBucket ? anomaly : 0.0));
        }
        model.sample(time, time + bucketLength, resourceMonitor);

        time += bucketLength;
    }
    LOG_INFO("Sizeof model now: " << model.memoryUsage());
    CPPUNIT_ASSERT(model.memoryUsage() > startMemoryUsage);

    core::CMemoryUsage memoryUsage;
    model.debugMemoryUsage(&memoryUsage);
    LOG_DEBUG("Debug sizeof model: " << memoryUsage.usage());
    CPPUNIT_ASSERT_EQUAL(model.computeMemoryUsage(), memoryUsage.usage());
}

CppUnit::Test* CModelMemoryTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CModelMemoryTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CModelMemoryTest>("CModelMemoryTest::testOnlineEventRateModel",
                                                                    &CModelMemoryTest::testOnlineEventRateModel));
    suiteOfTests->addTest(new CppUnit::TestCaller<CModelMemoryTest>("CModelMemoryTest::testOnlineMetricModel",
                                                                    &CModelMemoryTest::testOnlineMetricModel));

    return suiteOfTests;
}
