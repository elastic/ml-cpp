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

#include "CAnomalyDetectorModelConfigTest.h"

#include <core/CContainerPrinter.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CCountingModelFactory.h>
#include <model/CEventRateModelFactory.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CMetricModelFactory.h>

using namespace ml;
using namespace model;

namespace {
typedef std::vector<double> TDoubleVec;

const function_t::EFunction INDIVIDUAL_COUNT = function_t::E_IndividualCount;
const function_t::EFunction INDIVIDUAL_METRIC = function_t::E_IndividualMetricMin;
const function_t::EFunction POPULATION_COUNT = function_t::E_PopulationCount;
const function_t::EFunction POPULATION_METRIC = function_t::E_PopulationMetric;
}

void CAnomalyDetectorModelConfigTest::testNormal(void) {
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig(1800);
        CPPUNIT_ASSERT(config.init("testfiles/mlmodel.conf"));

        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, POPULATION_COUNT)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, POPULATION_METRIC)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(0.01, config.decayRate());
        CPPUNIT_ASSERT_EQUAL(0.01, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(0.01, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(0.01, config.factory(1, POPULATION_COUNT)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(0.01, config.factory(1, POPULATION_METRIC)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(2.0, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(2.0, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(2.0, config.factory(1, POPULATION_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(2.0, config.factory(1, POPULATION_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(0.0, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(0.0, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(0.0, config.factory(1, POPULATION_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(0.0, config.factory(1, POPULATION_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(0.1, config.factory(1, INDIVIDUAL_COUNT)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(0.1, config.factory(1, INDIVIDUAL_METRIC)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(0.01, config.factory(1, POPULATION_COUNT)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(0.01, config.factory(1, POPULATION_METRIC)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(std::size_t(10), config.factory(1, INDIVIDUAL_COUNT)->componentSize());
        CPPUNIT_ASSERT_EQUAL(std::size_t(10), config.factory(1, INDIVIDUAL_METRIC)->componentSize());
        CPPUNIT_ASSERT_EQUAL(std::size_t(10), config.factory(1, POPULATION_COUNT)->componentSize());
        CPPUNIT_ASSERT_EQUAL(std::size_t(10), config.factory(1, POPULATION_METRIC)->componentSize());
        CPPUNIT_ASSERT_EQUAL(std::size_t(20), config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_SampleCountFactor);
        CPPUNIT_ASSERT_EQUAL(std::size_t(20), config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_SampleCountFactor);
        CPPUNIT_ASSERT_EQUAL(std::size_t(20), config.factory(1, POPULATION_COUNT)->modelParams().s_SampleCountFactor);
        CPPUNIT_ASSERT_EQUAL(std::size_t(20), config.factory(1, POPULATION_METRIC)->modelParams().s_SampleCountFactor);
        TDoubleVec params;
        for (std::size_t i = 0u; i < model_t::NUMBER_AGGREGATION_STYLES; ++i) {
            for (std::size_t j = 0u; j < model_t::NUMBER_AGGREGATION_PARAMS; ++j) {
                params.push_back(
                    config.aggregationStyleParam(static_cast<model_t::EAggregationStyle>(i), static_cast<model_t::EAggregationParam>(j)));
            }
        }
        CPPUNIT_ASSERT_EQUAL(std::string("[0.9, 0.1, 2, 4, 0.3, 0.7, 3, 8, 0.6, 0.4, 2, 10]"), core::CContainerPrinter::print(params));
        CPPUNIT_ASSERT_EQUAL(0.01, config.maximumAnomalousProbability());
        CPPUNIT_ASSERT_EQUAL(60.0, config.noisePercentile());
        CPPUNIT_ASSERT_EQUAL(1.2, config.noiseMultiplier());
        CPPUNIT_ASSERT_EQUAL(4.0, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(4.0, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(4.0, config.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(4.0, config.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(0.5, config.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(std::string("[(0, 0), (70, 1.5), (85, 1.6), (90, 1.7), (95, 2), (97, 10), (98, 20), (99.5, 50), (100, 100)]"),
                             core::CContainerPrinter::print(config.normalizedScoreKnotPoints()));
        CPPUNIT_ASSERT_EQUAL(false, config.perPartitionNormalization());
    }
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();
        CPPUNIT_ASSERT(dynamic_cast<const CEventRateModelFactory*>(config.factory(1, function_t::E_IndividualCount).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CEventRateModelFactory*>(config.factory(1, function_t::E_IndividualNonZeroCount).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CEventRateModelFactory*>(config.factory(1, function_t::E_IndividualRareCount).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CMetricModelFactory*>(config.factory(1, function_t::E_IndividualMetricMean).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CMetricModelFactory*>(config.factory(1, function_t::E_IndividualMetricMin).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CMetricModelFactory*>(config.factory(1, function_t::E_IndividualMetricMax).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CMetricModelFactory*>(config.factory(1, function_t::E_IndividualMetric).get()));
        CPPUNIT_ASSERT(
            dynamic_cast<const CEventRatePopulationModelFactory*>(config.factory(1, function_t::E_PopulationDistinctCount).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CEventRatePopulationModelFactory*>(config.factory(1, function_t::E_PopulationRare).get()));
        CPPUNIT_ASSERT(dynamic_cast<const CCountingModelFactory*>(config.factory(CSearchKey::simpleCountKey()).get()));
        CPPUNIT_ASSERT_EQUAL(false, config.perPartitionNormalization());
    }
}

void CAnomalyDetectorModelConfigTest::testErrors(void) {
    {
        CAnomalyDetectorModelConfig config1 = CAnomalyDetectorModelConfig::defaultConfig(1800);
        CPPUNIT_ASSERT(!config1.init("testfiles/invalidmlmodel.conf"));
        CAnomalyDetectorModelConfig config2 = CAnomalyDetectorModelConfig::defaultConfig(1800);

        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_LearnRate,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_LearnRate,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_LearnRate,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_LearnRate,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_LearnRate);
        CPPUNIT_ASSERT_EQUAL(config2.decayRate(), config1.decayRate());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_DecayRate,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_DecayRate,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_DecayRate,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_DecayRate,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_DecayRate);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_InitialDecayRateMultiplier,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_InitialDecayRateMultiplier,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_InitialDecayRateMultiplier,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_InitialDecayRateMultiplier,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_MaximumUpdatesPerBucket,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_MaximumUpdatesPerBucket,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_MaximumUpdatesPerBucket,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_MaximumUpdatesPerBucket,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->minimumModeFraction(),
                             config1.factory(1, INDIVIDUAL_COUNT)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->minimumModeFraction(),
                             config1.factory(1, INDIVIDUAL_METRIC)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->minimumModeFraction(),
                             config1.factory(1, POPULATION_COUNT)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->minimumModeFraction(),
                             config1.factory(1, POPULATION_METRIC)->minimumModeFraction());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->componentSize(), config1.factory(1, INDIVIDUAL_COUNT)->componentSize());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->componentSize(),
                             config1.factory(1, INDIVIDUAL_METRIC)->componentSize());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->componentSize(), config1.factory(1, POPULATION_COUNT)->componentSize());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->componentSize(),
                             config1.factory(1, POPULATION_METRIC)->componentSize());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_SampleCountFactor,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_SampleCountFactor);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_SampleCountFactor,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_SampleCountFactor);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_SampleCountFactor,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_SampleCountFactor);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_SampleCountFactor,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_SampleCountFactor);
        for (std::size_t i = 0u; i < model_t::NUMBER_AGGREGATION_STYLES; ++i) {
            for (std::size_t j = 0u; j < model_t::NUMBER_AGGREGATION_PARAMS; ++j) {
                CPPUNIT_ASSERT_EQUAL(
                    config2.aggregationStyleParam(static_cast<model_t::EAggregationStyle>(i), static_cast<model_t::EAggregationParam>(j)),
                    config1.aggregationStyleParam(static_cast<model_t::EAggregationStyle>(i), static_cast<model_t::EAggregationParam>(j)));
            }
        }
        CPPUNIT_ASSERT_EQUAL(config2.maximumAnomalousProbability(), config1.maximumAnomalousProbability());
        CPPUNIT_ASSERT_EQUAL(config2.noisePercentile(), config1.noisePercentile());
        CPPUNIT_ASSERT_EQUAL(config2.noiseMultiplier(), config1.noiseMultiplier());
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMaximum,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMinimum,
                             config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMaximum,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMinimum,
                             config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMaximum,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMinimum,
                             config1.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMaximum,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        CPPUNIT_ASSERT_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMinimum,
                             config1.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(config2.normalizedScoreKnotPoints()),
                             core::CContainerPrinter::print(config1.normalizedScoreKnotPoints()));
    }
}

CppUnit::Test* CAnomalyDetectorModelConfigTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CAnomalyDetectorModelConfigTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyDetectorModelConfigTest>("CAnomalyDetectorModelConfigTest::testNormal",
                                                                                   &CAnomalyDetectorModelConfigTest::testNormal));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyDetectorModelConfigTest>("CAnomalyDetectorModelConfigTest::testErrors",
                                                                                   &CAnomalyDetectorModelConfigTest::testErrors));

    return suiteOfTests;
}
