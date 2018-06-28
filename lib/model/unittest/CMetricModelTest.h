/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMetricModelTest_h
#define INCLUDED_CMetricModelTest_h

#include <model/CModelFactory.h>
#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

#include <memory>

namespace ml {
namespace model {
class CMetricModelFactory;
class CInterimBucketCorrector;
struct SModelParams;
}
}

class CMetricModelTest : public CppUnit::TestFixture {
public:
    void testSample();
    void testMultivariateSample();
    void testProbabilityCalculationForMetric();
    void testProbabilityCalculationForMedian();
    void testProbabilityCalculationForLowMedian();
    void testProbabilityCalculationForHighMedian();
    void testProbabilityCalculationForLowMean();
    void testProbabilityCalculationForHighMean();
    void testProbabilityCalculationForLowSum();
    void testProbabilityCalculationForHighSum();
    void testProbabilityCalculationForLatLong();
    void testInfluence();
    void testLatLongInfluence();
    void testPrune();
    void testSkipSampling();
    void testExplicitNulls();
    void testKey();
    void testVarp();
    void testInterimCorrections();
    void testInterimCorrectionsWithCorrelations();
    void testCorrelatePersist();
    void testSummaryCountZeroRecordsAreIgnored();
    void testDecayRateControl();
    void testIgnoreSamplingGivenDetectionRules();

    void setUp();
    static CppUnit::Test* suite();

private:
    using TInterimBucketCorrectorPtr = std::shared_ptr<ml::model::CInterimBucketCorrector>;
    using TMetricModelFactoryPtr = boost::shared_ptr<ml::model::CMetricModelFactory>;

private:
    void makeModel(const ml::model::SModelParams& params,
                   const ml::model_t::TFeatureVec& features,
                   ml::core_t::TTime startTime,
                   unsigned int* sampleCount = nullptr);

    void makeModel(const ml::model::SModelParams& params,
                   const ml::model_t::TFeatureVec& features,
                   ml::core_t::TTime startTime,
                   ml::model::CModelFactory::TDataGathererPtr& gatherer,
                   ml::model::CModelFactory::TModelPtr& model,
                   unsigned int* sampleCount = nullptr);

private:
    TInterimBucketCorrectorPtr m_InterimBucketCorrector;
    TMetricModelFactoryPtr m_Factory;
    ml::model::CModelFactory::TDataGathererPtr m_Gatherer;
    ml::model::CModelFactory::TModelPtr m_Model;
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CMetricModelTest_h
