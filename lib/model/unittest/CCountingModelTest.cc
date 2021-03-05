/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CoreTypes.h>

#include <model/CCountingModel.h>
#include <model/CCountingModelFactory.h>
#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CResourceMonitor.h>
#include <model/ModelTypes.h>

#include <test/CRandomNumbers.h>

#include "CModelTestFixtureBase.h"

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CCountingModelTest)

using namespace ml;
using namespace model;

class CTestFixture : public CModelTestFixtureBase {
protected:
    static SModelParams::TStrDetectionRulePr
    makeScheduledEvent(const std::string& description, double start, double end) {
        CRuleCondition conditionGte;
        conditionGte.appliesTo(CRuleCondition::E_Time);
        conditionGte.op(CRuleCondition::E_GTE);
        conditionGte.value(start);
        CRuleCondition conditionLt;
        conditionLt.appliesTo(CRuleCondition::E_Time);
        conditionLt.op(CRuleCondition::E_LT);
        conditionLt.value(end);

        CDetectionRule rule;
        rule.action(CDetectionRule::E_SkipModelUpdate);
        rule.addCondition(conditionGte);
        rule.addCondition(conditionLt);

        SModelParams::TStrDetectionRulePr event = std::make_pair(description, rule);
        return event;
    }

    void makeModel(const SModelParams& params,
                   const model_t::TFeatureVec& features,
                   core_t::TTime startTime) {
        this->makeModelT<CCountingModelFactory>(
            params, features, startTime, model_t::E_Counting, m_Gatherer, m_Model);
    }
};

BOOST_FIXTURE_TEST_CASE(testSkipSampling, CTestFixture) {
    core_t::TTime startTime{100};
    core_t::TTime bucketLength{100};
    std::size_t maxAgeBuckets{1};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
    CCountingModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_IndividualCountByBucketAndPerson};
    factory.features(features);

    // Model where gap is not skipped
    {
        CModelFactory::TDataGathererPtr gathererNoGap;
        CModelFactory::TModelPtr modelNoGap;
        this->makeModelT<CCountingModelFactory>(
            params, features, startTime, model_t::E_Counting, gathererNoGap, modelNoGap);
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gathererNoGap));

        // |2|2|0|0|1| -> 1.0 mean count
        this->addArrival(SMessage(100, "p", TOptionalStr()), gathererNoGap);
        this->addArrival(SMessage(110, "p", TOptionalStr()), gathererNoGap);
        modelNoGap->sample(100, 200, m_ResourceMonitor);
        this->addArrival(SMessage(250, "p", TOptionalStr()), gathererNoGap);
        this->addArrival(SMessage(280, "p", TOptionalStr()), gathererNoGap);
        modelNoGap->sample(200, 500, m_ResourceMonitor);
        this->addArrival(SMessage(500, "p", TOptionalStr()), gathererNoGap);
        modelNoGap->sample(500, 600, m_ResourceMonitor);

        BOOST_REQUIRE_EQUAL(1.0, *modelNoGap->baselineBucketCount(0));
    }

    // Model where gap is skipped
    {
        CModelFactory::TDataGathererPtr gathererWithGap;
        CModelFactory::TModelPtr modelWithGap;
        this->makeModelT<CCountingModelFactory>(params, features, startTime,
                                                model_t::E_Counting,
                                                gathererWithGap, modelWithGap);
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", gathererWithGap));

        // |2|2|0|0|1|
        // |2|X|X|X|1| -> 1.5 mean count where X means skipped bucket
        this->addArrival(SMessage(100, "p", TOptionalStr()), gathererWithGap);
        this->addArrival(SMessage(110, "p", TOptionalStr()), gathererWithGap);
        modelWithGap->sample(100, 200, m_ResourceMonitor);
        this->addArrival(SMessage(250, "p", TOptionalStr()), gathererWithGap);
        this->addArrival(SMessage(280, "p", TOptionalStr()), gathererWithGap);
        modelWithGap->skipSampling(500);
        modelWithGap->prune(maxAgeBuckets);
        BOOST_REQUIRE_EQUAL(1, gathererWithGap->numberActivePeople());
        this->addArrival(SMessage(500, "p", TOptionalStr()), gathererWithGap);
        modelWithGap->sample(500, 600, m_ResourceMonitor);

        BOOST_REQUIRE_EQUAL(1.5, *modelWithGap->baselineBucketCount(0));
    }
}

BOOST_FIXTURE_TEST_CASE(testCheckScheduledEvents, CTestFixture) {
    core_t::TTime startTime{100};
    core_t::TTime bucketLength{100};

    SModelParams params(bucketLength);

    SModelParams::TStrDetectionRulePrVec events{
        makeScheduledEvent("first event", 200, 300),
        makeScheduledEvent("long event", 400, 1000),
        makeScheduledEvent("masked event", 600, 800),
        makeScheduledEvent("overlapping event", 900, 1100)};

    params.s_ScheduledEvents = std::cref(events);
    auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);

    CCountingModelFactory factory(params, interimBucketCorrector);
    model_t::TFeatureVec features{model_t::E_IndividualCountByBucketAndPerson};
    factory.features(features);

    {
        this->makeModel(params, features, startTime);
        CCountingModel* modelNoGap = dynamic_cast<CCountingModel*>(m_Model.get());
        BOOST_TEST_REQUIRE(modelNoGap);
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

        SModelParams::TStrDetectionRulePrVec matchedEvents =
            modelNoGap->checkScheduledEvents(50);
        BOOST_REQUIRE_EQUAL(std::size_t{0}, matchedEvents.size());

        matchedEvents = modelNoGap->checkScheduledEvents(200);
        BOOST_REQUIRE_EQUAL(std::size_t{1}, matchedEvents.size());
        BOOST_REQUIRE_EQUAL(std::string("first event"), matchedEvents[0].first);

        matchedEvents = modelNoGap->checkScheduledEvents(700);
        BOOST_REQUIRE_EQUAL(std::size_t{2}, matchedEvents.size());
        BOOST_REQUIRE_EQUAL(std::string("long event"), matchedEvents[0].first);
        BOOST_REQUIRE_EQUAL(std::string("masked event"), matchedEvents[1].first);

        matchedEvents = modelNoGap->checkScheduledEvents(1000);
        BOOST_REQUIRE_EQUAL(std::size_t{1}, matchedEvents.size());
        BOOST_REQUIRE_EQUAL(std::string("overlapping event"), matchedEvents[0].first);

        matchedEvents = modelNoGap->checkScheduledEvents(1100);
        BOOST_REQUIRE_EQUAL(std::size_t{0}, matchedEvents.size());

        // Test event descriptions are set
        modelNoGap->sample(200, 800, m_ResourceMonitor);
        std::vector<std::string> eventDescriptions =
            modelNoGap->scheduledEventDescriptions(200);
        BOOST_REQUIRE_EQUAL(std::size_t{1}, eventDescriptions.size());
        BOOST_REQUIRE_EQUAL(std::string("first event"), eventDescriptions[0]);

        eventDescriptions = modelNoGap->scheduledEventDescriptions(700);
        BOOST_REQUIRE_EQUAL(std::size_t{2}, eventDescriptions.size());
        BOOST_REQUIRE_EQUAL(std::string("long event"), eventDescriptions[0]);
        BOOST_REQUIRE_EQUAL(std::string("masked event"), eventDescriptions[1]);

        // There are no events at this time
        modelNoGap->sample(1500, 1600, m_ResourceMonitor);
        eventDescriptions = modelNoGap->scheduledEventDescriptions(1500);
        BOOST_TEST_REQUIRE(eventDescriptions.empty());
    }

    // Test sampleBucketStatistics
    {
        this->makeModel(params, features, startTime);
        CCountingModel* modelNoGap = dynamic_cast<CCountingModel*>(m_Model.get());
        BOOST_TEST_REQUIRE(modelNoGap);
        BOOST_REQUIRE_EQUAL(0, this->addPerson("p", m_Gatherer));

        // There are no events at this time
        modelNoGap->sampleBucketStatistics(0, 100, m_ResourceMonitor);
        std::vector<std::string> eventDescriptions =
            modelNoGap->scheduledEventDescriptions(0);
        BOOST_TEST_REQUIRE(eventDescriptions.empty());

        // Test event descriptions are set
        modelNoGap->sampleBucketStatistics(200, 800, m_ResourceMonitor);
        eventDescriptions = modelNoGap->scheduledEventDescriptions(200);
        BOOST_REQUIRE_EQUAL(std::size_t{1}, eventDescriptions.size());
        BOOST_REQUIRE_EQUAL(std::string("first event"), eventDescriptions[0]);

        eventDescriptions = modelNoGap->scheduledEventDescriptions(700);
        BOOST_REQUIRE_EQUAL(std::size_t{2}, eventDescriptions.size());
        BOOST_REQUIRE_EQUAL(std::string("long event"), eventDescriptions[0]);
        BOOST_REQUIRE_EQUAL(std::string("masked event"), eventDescriptions[1]);

        modelNoGap->sampleBucketStatistics(1500, 1600, m_ResourceMonitor);
        eventDescriptions = modelNoGap->scheduledEventDescriptions(1500);
        BOOST_TEST_REQUIRE(eventDescriptions.empty());
    }
}

BOOST_FIXTURE_TEST_CASE(testInterimBucketCorrector, CTestFixture) {
    // Check that we correctly update estimate bucket completeness.

    core_t::TTime time{0};
    core_t::TTime bucketLength{600};

    SModelParams params(bucketLength);
    params.s_DecayRate = 0.001;

    this->makeModel(params, {model_t::E_IndividualCountByBucketAndPerson}, time);
    CCountingModel* model = dynamic_cast<CCountingModel*>(m_Model.get());
    BOOST_TEST_REQUIRE(model);

    BOOST_REQUIRE_EQUAL(0, this->addPerson("p1", m_Gatherer));
    BOOST_REQUIRE_EQUAL(1, this->addPerson("p2", m_Gatherer));

    test::CRandomNumbers rng;

    TDoubleVec uniform01;
    TSizeVec offsets;

    for (std::size_t i = 0; i < 10; ++i, time += bucketLength) {
        rng.generateUniformSamples(0, bucketLength, 10, offsets);
        std::sort(offsets.begin(), offsets.end());
        for (auto offset : offsets) {
            rng.generateUniformSamples(0.0, 1.0, 1, uniform01);
            this->addArrival(SMessage(time + static_cast<core_t::TTime>(offset),
                                      uniform01[0] < 0.5 ? "p1" : "p2", TOptionalStr()),
                             m_Gatherer);
        }
        model->sample(time, time + bucketLength, m_ResourceMonitor);
    }

    rng.generateUniformSamples(0, bucketLength, 10, offsets);
    std::sort(offsets.begin(), offsets.end());

    for (std::size_t i = 0; i < offsets.size(); ++i) {
        rng.generateUniformSamples(0.0, 1.0, 1, uniform01);
        this->addArrival(SMessage(time + static_cast<core_t::TTime>(offsets[i]),
                                  uniform01[0] < 0.5 ? "p1" : "p2", TOptionalStr()),
                         m_Gatherer);
        model->sampleBucketStatistics(time, time + bucketLength, m_ResourceMonitor);
        BOOST_REQUIRE_EQUAL(static_cast<double>(i + 1) / 10.0,
                            m_InterimBucketCorrector->completeness());
    }
}

BOOST_AUTO_TEST_SUITE_END()
