/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEventRateModelTest_h
#define INCLUDED_CEventRateModelTest_h

#include <model/CModelFactory.h>
#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

#include <memory>
#include <string>

namespace ml {
namespace model {
class CEventRateModelFactory;
class CInterimBucketCorrector;
struct SModelParams;
}
}

class CEventRateModelTest : public CppUnit::TestFixture {
public:
    void testOnlineCountSample();
    void testOnlineNonZeroCountSample();
    void testOnlineRare();
    void testOnlineProbabilityCalculation();
    void testOnlineProbabilityCalculationForLowNonZeroCount();
    void testOnlineProbabilityCalculationForHighNonZeroCount();
    void testOnlineCorrelatedNoTrend();
    void testOnlineCorrelatedTrend();
    void testPrune();
    void testKey();
    void testModelsWithValueFields();
    void testCountProbabilityCalculationWithInfluence();
    void testDistinctCountProbabilityCalculationWithInfluence();
    void testOnlineRareWithInfluence();
    void testSkipSampling();
    void testExplicitNulls();
    void testInterimCorrections();
    void testInterimCorrectionsWithCorrelations();
    void testSummaryCountZeroRecordsAreIgnored();
    void testComputeProbabilityGivenDetectionRule();
    void testDecayRateControl();
    void testIgnoreSamplingGivenDetectionRules();

    virtual void setUp();
    static CppUnit::Test* suite();

private:
    using TInterimBucketCorrectorPtr = std::shared_ptr<ml::model::CInterimBucketCorrector>;
    using TEventRateModelFactoryPtr = boost::shared_ptr<ml::model::CEventRateModelFactory>;

private:
    void makeModel(const ml::model::SModelParams& params,
                   const ml::model_t::TFeatureVec& features,
                   ml::core_t::TTime startTime,
                   std::size_t numberPeople,
                   const std::string& summaryCountField = "");

    void makeModel(const ml::model::SModelParams& params,
                   const ml::model_t::TFeatureVec& features,
                   ml::core_t::TTime startTime,
                   std::size_t numberPeople,
                   ml::model::CModelFactory::TDataGathererPtr& gatherer,
                   ml::model::CModelFactory::TModelPtr& model,
                   const std::string& summaryCountField = "");

private:
    TInterimBucketCorrectorPtr m_InterimBucketCorrector;
    TEventRateModelFactoryPtr m_Factory;
    ml::model::CModelFactory::TDataGathererPtr m_Gatherer;
    ml::model::CModelFactory::TModelPtr m_Model;
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRateModelTest_h
