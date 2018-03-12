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

#include "CCountingModelTest.h"

#include <core/CContainerPrinter.h>
#include <core/CoreTypes.h>
#include <core/CLogger.h>

#include <model/CDataGatherer.h>
#include <model/CCountingModelFactory.h>
#include <model/CCountingModel.h>
#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CResourceMonitor.h>
#include <model/ModelTypes.h>

#include <string>
#include <vector>

using namespace ml;
using namespace model;


namespace {
std::size_t addPerson(const std::string &p,
                      const CModelFactory::TDataGathererPtr &gatherer,
                      CResourceMonitor &resourceMonitor) {
    CDataGatherer::TStrCPtrVec person;
    person.push_back(&p);
    CEventData result;
    gatherer->processFields(person, result, resourceMonitor);
    return *result.personId();
}

void addArrival(CDataGatherer &gatherer,
                CResourceMonitor &resourceMonitor,
                core_t::TTime time,
                const std::string &person) {
    CDataGatherer::TStrCPtrVec fieldValues;
    fieldValues.push_back(&person);

    CEventData eventData;
    eventData.time(time);
    gatherer.addArrival(fieldValues, eventData, resourceMonitor);
}

SModelParams::TStrDetectionRulePr makeScheduledEvent(const std::string &description, double start, double end) {
    CRuleCondition conditionGte;
    conditionGte.type(CRuleCondition::E_Time);
    conditionGte.condition().s_Op = CRuleCondition::E_GTE;
    conditionGte.condition().s_Threshold = start;
    CRuleCondition conditionLt;
    conditionLt.type(CRuleCondition::E_Time);
    conditionLt.condition().s_Op = CRuleCondition::E_LT;
    conditionLt.condition().s_Threshold = end;

    CDetectionRule rule;
    rule.action(CDetectionRule::E_SkipSampling);
    rule.conditionsConnective(CDetectionRule::E_And);
    rule.addCondition(conditionGte);
    rule.addCondition(conditionLt);

    SModelParams::TStrDetectionRulePr event = std::make_pair(description, rule);
    return event;
}

const std::string EMPTY_STRING;
}

void CCountingModelTest::testSkipSampling(void) {
    LOG_DEBUG("*** testSkipSampling ***");

    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);
    std::size_t maxAgeBuckets(1);

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    CCountingModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    factory.features(features);

    // Model where gap is not skipped
    {
        CModelFactory::SGathererInitializationData gathererNoGapInitData(startTime);
        CModelFactory::TDataGathererPtr gathererNoGap(factory.makeDataGatherer(gathererNoGapInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gathererNoGap, m_ResourceMonitor));
        CModelFactory::SModelInitializationData modelNoGapInitData(gathererNoGap);
        CAnomalyDetectorModel::TModelPtr modelHolderNoGap(factory.makeModel(modelNoGapInitData));
        CCountingModel *modelNoGap = dynamic_cast<CCountingModel*>(modelHolderNoGap.get());

        // |2|2|0|0|1| -> 1.0 mean count
        addArrival(*gathererNoGap, m_ResourceMonitor, 100, "p");
        addArrival(*gathererNoGap, m_ResourceMonitor, 110, "p");
        modelNoGap->sample(100, 200, m_ResourceMonitor);
        addArrival(*gathererNoGap, m_ResourceMonitor, 250, "p");
        addArrival(*gathererNoGap, m_ResourceMonitor, 280, "p");
        modelNoGap->sample(200, 500, m_ResourceMonitor);
        addArrival(*gathererNoGap, m_ResourceMonitor, 500, "p");
        modelNoGap->sample(500, 600, m_ResourceMonitor);

        CPPUNIT_ASSERT_EQUAL(1.0, *modelNoGap->baselineBucketCount(0));
    }

    // Model where gap is skipped
    {
        CModelFactory::SGathererInitializationData gathererWithGapInitData(startTime);
        CModelFactory::TDataGathererPtr gathererWithGap(factory.makeDataGatherer(gathererWithGapInitData));
        CPPUNIT_ASSERT_EQUAL(std::size_t(0), addPerson("p", gathererWithGap, m_ResourceMonitor));
        CModelFactory::SModelInitializationData modelWithGapInitData(gathererWithGap);
        CAnomalyDetectorModel::TModelPtr modelHolderWithGap(factory.makeModel(modelWithGapInitData));
        CCountingModel *modelWithGap = dynamic_cast<CCountingModel*>(modelHolderWithGap.get());

        // |2|2|0|0|1|
        // |2|X|X|X|1| -> 1.5 mean count where X means skipped bucket
        addArrival(*gathererWithGap, m_ResourceMonitor, 100, "p");
        addArrival(*gathererWithGap, m_ResourceMonitor, 110, "p");
        modelWithGap->sample(100, 200, m_ResourceMonitor);
        addArrival(*gathererWithGap, m_ResourceMonitor, 250, "p");
        addArrival(*gathererWithGap, m_ResourceMonitor, 280, "p");
        modelWithGap->skipSampling(500);
        modelWithGap->prune(maxAgeBuckets);
        CPPUNIT_ASSERT_EQUAL(std::size_t(1), gathererWithGap->numberActivePeople());
        addArrival(*gathererWithGap, m_ResourceMonitor, 500, "p");
        modelWithGap->sample(500, 600, m_ResourceMonitor);

        CPPUNIT_ASSERT_EQUAL(1.5, *modelWithGap->baselineBucketCount(0));
    }
}

void CCountingModelTest::testCheckScheduledEvents(void) {
    LOG_DEBUG("*** testCheckScheduledEvents ***");

    core_t::TTime startTime(100);
    core_t::TTime bucketLength(100);

    SModelParams params(bucketLength);

    SModelParams::TStrDetectionRulePrVec events;
    events.push_back(makeScheduledEvent("first event", 200, 300));
    events.push_back(makeScheduledEvent("long event", 400, 1000));
    events.push_back(makeScheduledEvent("masked event", 600, 800));
    events.push_back(makeScheduledEvent("overlapping event", 900, 1100));
    params.s_ScheduledEvents = boost::cref(events);

    CCountingModelFactory factory(params);
    model_t::TFeatureVec features(1u, model_t::E_IndividualCountByBucketAndPerson);
    factory.features(features);

    {
        CModelFactory::SGathererInitializationData gathererNoGapInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererNoGapInitData));
        CModelFactory::SModelInitializationData modelNoGapInitData(gatherer);
        addArrival(*gatherer, m_ResourceMonitor, 200, "p");

        CAnomalyDetectorModel::TModelPtr modelHolderNoGap(factory.makeModel(modelNoGapInitData));
        CCountingModel *modelNoGap = dynamic_cast<CCountingModel*>(modelHolderNoGap.get());

        SModelParams::TStrDetectionRulePrVec matchedEvents = modelNoGap->checkScheduledEvents(50);
        CPPUNIT_ASSERT_EQUAL(std::size_t{0}, matchedEvents.size());

        matchedEvents = modelNoGap->checkScheduledEvents(200);
        CPPUNIT_ASSERT_EQUAL(std::size_t{1}, matchedEvents.size());
        CPPUNIT_ASSERT_EQUAL(std::string("first event"), matchedEvents[0].first);

        matchedEvents = modelNoGap->checkScheduledEvents(700);
        CPPUNIT_ASSERT_EQUAL(std::size_t{2}, matchedEvents.size());
        CPPUNIT_ASSERT_EQUAL(std::string("long event"), matchedEvents[0].first);
        CPPUNIT_ASSERT_EQUAL(std::string("masked event"), matchedEvents[1].first);

        matchedEvents = modelNoGap->checkScheduledEvents(1000);
        CPPUNIT_ASSERT_EQUAL(std::size_t{1}, matchedEvents.size());
        CPPUNIT_ASSERT_EQUAL(std::string("overlapping event"), matchedEvents[0].first);

        matchedEvents = modelNoGap->checkScheduledEvents(1100);
        CPPUNIT_ASSERT_EQUAL(std::size_t{0}, matchedEvents.size());

        // Test event descriptions are set
        modelNoGap->sample(200, 800, m_ResourceMonitor);
        std::vector<std::string> eventDescriptions = modelNoGap->scheduledEventDescriptions(200);
        CPPUNIT_ASSERT_EQUAL(std::size_t{1}, eventDescriptions.size());
        CPPUNIT_ASSERT_EQUAL(std::string("first event"), eventDescriptions[0]);

        eventDescriptions = modelNoGap->scheduledEventDescriptions(700);
        CPPUNIT_ASSERT_EQUAL(std::size_t{2}, eventDescriptions.size());
        CPPUNIT_ASSERT_EQUAL(std::string("long event"), eventDescriptions[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("masked event"), eventDescriptions[1]);

        // There are no events at this time
        modelNoGap->sample(1500, 1600, m_ResourceMonitor);
        eventDescriptions = modelNoGap->scheduledEventDescriptions(1500);
        CPPUNIT_ASSERT(eventDescriptions.empty());
    }

    // Test sampleBucketStatistics
    {
        CModelFactory::SGathererInitializationData gathererNoGapInitData(startTime);
        CModelFactory::TDataGathererPtr gatherer(factory.makeDataGatherer(gathererNoGapInitData));
        CModelFactory::SModelInitializationData modelNoGapInitData(gatherer);
        addArrival(*gatherer, m_ResourceMonitor, 100, "p");

        CAnomalyDetectorModel::TModelPtr modelHolderNoGap(factory.makeModel(modelNoGapInitData));
        CCountingModel *modelNoGap = dynamic_cast<CCountingModel*>(modelHolderNoGap.get());

        // There are no events at this time
        modelNoGap->sampleBucketStatistics(0, 100, m_ResourceMonitor);
        std::vector<std::string> eventDescriptions = modelNoGap->scheduledEventDescriptions(0);
        CPPUNIT_ASSERT(eventDescriptions.empty());

        // Test event descriptions are set
        modelNoGap->sampleBucketStatistics(200, 800, m_ResourceMonitor);
        eventDescriptions = modelNoGap->scheduledEventDescriptions(200);
        CPPUNIT_ASSERT_EQUAL(std::size_t{1}, eventDescriptions.size());
        CPPUNIT_ASSERT_EQUAL(std::string("first event"), eventDescriptions[0]);

        eventDescriptions = modelNoGap->scheduledEventDescriptions(700);
        CPPUNIT_ASSERT_EQUAL(std::size_t{2}, eventDescriptions.size());
        CPPUNIT_ASSERT_EQUAL(std::string("long event"), eventDescriptions[0]);
        CPPUNIT_ASSERT_EQUAL(std::string("masked event"), eventDescriptions[1]);

        modelNoGap->sampleBucketStatistics(1500, 1600, m_ResourceMonitor);
        eventDescriptions = modelNoGap->scheduledEventDescriptions(1500);
        CPPUNIT_ASSERT(eventDescriptions.empty());
    }
}

CppUnit::Test *CCountingModelTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CCountingModelTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CCountingModelTest>(
                               "CCountingModelTest::testSkipSampling",
                               &CCountingModelTest::testSkipSampling) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CCountingModelTest>(
                               "CCountingModelTest::testCheckScheduledEvents",
                               &CCountingModelTest::testCheckScheduledEvents) );
    return suiteOfTests;
}
