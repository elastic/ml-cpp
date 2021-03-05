/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>

#include <maths/CModel.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CCountingModelFactory.h>
#include <model/CEventRateModelFactory.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CMetricModelFactory.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CAnomalyDetectorModelConfigTest)

using namespace ml;
using namespace model;

namespace {
using TDoubleVec = std::vector<double>;

const function_t::EFunction INDIVIDUAL_COUNT = function_t::E_IndividualCount;
const function_t::EFunction INDIVIDUAL_METRIC = function_t::E_IndividualMetricMin;
const function_t::EFunction POPULATION_COUNT = function_t::E_PopulationCount;
const function_t::EFunction POPULATION_METRIC = function_t::E_PopulationMetric;
}

BOOST_AUTO_TEST_CASE(testNormal) {
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig(1800);
        BOOST_TEST_REQUIRE(config.init("testfiles/mlmodel.conf"));

        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, POPULATION_COUNT)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, POPULATION_METRIC)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(0.01, config.decayRate());
        BOOST_REQUIRE_EQUAL(
            0.01, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(
            0.01, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(
            0.01, config.factory(1, POPULATION_COUNT)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(
            0.01, config.factory(1, POPULATION_METRIC)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(
            2.0, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            2.0, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            2.0, config.factory(1, POPULATION_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            2.0, config.factory(1, POPULATION_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            0.0, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(
            0.0, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(
            0.0, config.factory(1, POPULATION_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(
            0.0, config.factory(1, POPULATION_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(0.1, config.factory(1, INDIVIDUAL_COUNT)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(0.1, config.factory(1, INDIVIDUAL_METRIC)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(0.01, config.factory(1, POPULATION_COUNT)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(0.01, config.factory(1, POPULATION_METRIC)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(std::size_t(10),
                            config.factory(1, INDIVIDUAL_COUNT)->componentSize());
        BOOST_REQUIRE_EQUAL(std::size_t(10),
                            config.factory(1, INDIVIDUAL_METRIC)->componentSize());
        BOOST_REQUIRE_EQUAL(std::size_t(10),
                            config.factory(1, POPULATION_COUNT)->componentSize());
        BOOST_REQUIRE_EQUAL(std::size_t(10),
                            config.factory(1, POPULATION_METRIC)->componentSize());
        BOOST_REQUIRE_EQUAL(std::size_t(20),
                            config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_SampleCountFactor);
        BOOST_REQUIRE_EQUAL(std::size_t(20),
                            config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_SampleCountFactor);
        BOOST_REQUIRE_EQUAL(std::size_t(20),
                            config.factory(1, POPULATION_COUNT)->modelParams().s_SampleCountFactor);
        BOOST_REQUIRE_EQUAL(std::size_t(20),
                            config.factory(1, POPULATION_METRIC)->modelParams().s_SampleCountFactor);
        TDoubleVec params;
        for (std::size_t i = 0; i < model_t::NUMBER_AGGREGATION_STYLES; ++i) {
            for (std::size_t j = 0; j < model_t::NUMBER_AGGREGATION_PARAMS; ++j) {
                params.push_back(config.aggregationStyleParam(
                    static_cast<model_t::EAggregationStyle>(i),
                    static_cast<model_t::EAggregationParam>(j)));
            }
        }
        BOOST_REQUIRE_EQUAL(std::string("[0.9, 0.1, 2, 4, 0.3, 0.7, 3, 8, 0.6, 0.4, 2, 10]"),
                            core::CContainerPrinter::print(params));
        BOOST_REQUIRE_EQUAL(0.01, config.maximumAnomalousProbability());
        BOOST_REQUIRE_EQUAL(60.0, config.noisePercentile());
        BOOST_REQUIRE_EQUAL(1.2, config.noiseMultiplier());
        BOOST_REQUIRE_EQUAL(
            4.0, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            4.0, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            4.0, config.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            4.0, config.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            0.5, config.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            std::string("[(0, 0), (70, 1.5), (85, 1.6), (90, 1.7), (95, 2), (97, 10), (98, 20), (99.5, 50), (100, 100)]"),
            core::CContainerPrinter::print(config.normalizedScoreKnotPoints()));
    }
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();
        BOOST_TEST_REQUIRE(dynamic_cast<const CEventRateModelFactory*>(
            config.factory(1, function_t::E_IndividualCount).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CEventRateModelFactory*>(
            config.factory(1, function_t::E_IndividualNonZeroCount).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CEventRateModelFactory*>(
            config.factory(1, function_t::E_IndividualRareCount).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CMetricModelFactory*>(
            config.factory(1, function_t::E_IndividualMetricMean).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CMetricModelFactory*>(
            config.factory(1, function_t::E_IndividualMetricMin).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CMetricModelFactory*>(
            config.factory(1, function_t::E_IndividualMetricMax).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CMetricModelFactory*>(
            config.factory(1, function_t::E_IndividualMetric).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CEventRatePopulationModelFactory*>(
            config.factory(1, function_t::E_PopulationDistinctCount).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CEventRatePopulationModelFactory*>(
            config.factory(1, function_t::E_PopulationRare).get()));
        BOOST_TEST_REQUIRE(dynamic_cast<const CCountingModelFactory*>(
            config.factory(CSearchKey::simpleCountKey()).get()));
    }
}

BOOST_AUTO_TEST_CASE(testErrors) {
    {
        CAnomalyDetectorModelConfig config1 =
            CAnomalyDetectorModelConfig::defaultConfig(1800);
        BOOST_TEST_REQUIRE(!config1.init("testfiles/invalidmlmodel.conf"));
        CAnomalyDetectorModelConfig config2 =
            CAnomalyDetectorModelConfig::defaultConfig(1800);

        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_LearnRate,
                            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_LearnRate,
                            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_LearnRate,
                            config1.factory(1, POPULATION_COUNT)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_LearnRate,
                            config1.factory(1, POPULATION_METRIC)->modelParams().s_LearnRate);
        BOOST_REQUIRE_EQUAL(config2.decayRate(), config1.decayRate());
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_DecayRate,
                            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_DecayRate,
                            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_COUNT)->modelParams().s_DecayRate,
                            config1.factory(1, POPULATION_COUNT)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_METRIC)->modelParams().s_DecayRate,
                            config1.factory(1, POPULATION_METRIC)->modelParams().s_DecayRate);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_InitialDecayRateMultiplier,
            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_InitialDecayRateMultiplier,
            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_COUNT)->modelParams().s_InitialDecayRateMultiplier,
            config1.factory(1, POPULATION_COUNT)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_METRIC)->modelParams().s_InitialDecayRateMultiplier,
            config1.factory(1, POPULATION_METRIC)->modelParams().s_InitialDecayRateMultiplier);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_MaximumUpdatesPerBucket,
            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_MaximumUpdatesPerBucket,
            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_COUNT)->modelParams().s_MaximumUpdatesPerBucket,
            config1.factory(1, POPULATION_COUNT)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_METRIC)->modelParams().s_MaximumUpdatesPerBucket,
            config1.factory(1, POPULATION_METRIC)->modelParams().s_MaximumUpdatesPerBucket);
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->minimumModeFraction(),
                            config1.factory(1, INDIVIDUAL_COUNT)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->minimumModeFraction(),
                            config1.factory(1, INDIVIDUAL_METRIC)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_COUNT)->minimumModeFraction(),
                            config1.factory(1, POPULATION_COUNT)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_METRIC)->minimumModeFraction(),
                            config1.factory(1, POPULATION_METRIC)->minimumModeFraction());
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_COUNT)->componentSize(),
                            config1.factory(1, INDIVIDUAL_COUNT)->componentSize());
        BOOST_REQUIRE_EQUAL(config2.factory(1, INDIVIDUAL_METRIC)->componentSize(),
                            config1.factory(1, INDIVIDUAL_METRIC)->componentSize());
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_COUNT)->componentSize(),
                            config1.factory(1, POPULATION_COUNT)->componentSize());
        BOOST_REQUIRE_EQUAL(config2.factory(1, POPULATION_METRIC)->componentSize(),
                            config1.factory(1, POPULATION_METRIC)->componentSize());
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_SampleCountFactor,
            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_SampleCountFactor);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_SampleCountFactor,
            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_SampleCountFactor);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_COUNT)->modelParams().s_SampleCountFactor,
            config1.factory(1, POPULATION_COUNT)->modelParams().s_SampleCountFactor);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_METRIC)->modelParams().s_SampleCountFactor,
            config1.factory(1, POPULATION_METRIC)->modelParams().s_SampleCountFactor);
        for (std::size_t i = 0; i < model_t::NUMBER_AGGREGATION_STYLES; ++i) {
            for (std::size_t j = 0; j < model_t::NUMBER_AGGREGATION_PARAMS; ++j) {
                BOOST_REQUIRE_EQUAL(config2.aggregationStyleParam(
                                        static_cast<model_t::EAggregationStyle>(i),
                                        static_cast<model_t::EAggregationParam>(j)),
                                    config1.aggregationStyleParam(
                                        static_cast<model_t::EAggregationStyle>(i),
                                        static_cast<model_t::EAggregationParam>(j)));
            }
        }
        BOOST_REQUIRE_EQUAL(config2.maximumAnomalousProbability(),
                            config1.maximumAnomalousProbability());
        BOOST_REQUIRE_EQUAL(config2.noisePercentile(), config1.noisePercentile());
        BOOST_REQUIRE_EQUAL(config2.noiseMultiplier(), config1.noiseMultiplier());
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMaximum,
            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMinimum,
            config1.factory(1, INDIVIDUAL_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMaximum,
            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMinimum,
            config1.factory(1, INDIVIDUAL_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMaximum,
            config1.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMinimum,
            config1.factory(1, POPULATION_COUNT)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMaximum,
            config1.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMaximum);
        BOOST_REQUIRE_EQUAL(
            config2.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMinimum,
            config1.factory(1, POPULATION_METRIC)->modelParams().s_PruneWindowScaleMinimum);
        BOOST_REQUIRE_EQUAL(
            core::CContainerPrinter::print(config2.normalizedScoreKnotPoints()),
            core::CContainerPrinter::print(config1.normalizedScoreKnotPoints()));
    }
}

BOOST_AUTO_TEST_CASE(testConfigureModelPlot) {
    using TStrSet = CAnomalyDetectorModelConfig::TStrSet;
    static const std::string EMPTY_STRING;
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();
        config.configureModelPlot(true, true, EMPTY_STRING);
        BOOST_REQUIRE_EQUAL(ml::maths::CModel::DEFAULT_BOUNDS_PERCENTILE,
                            config.modelPlotBoundsPercentile());
        BOOST_REQUIRE_EQUAL(true, config.modelPlotAnnotationsEnabled());
        BOOST_REQUIRE_EQUAL(0, config.modelPlotTerms().size());
    }
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();
        config.configureModelPlot(false, true, EMPTY_STRING);
        BOOST_REQUIRE_EQUAL(-1.0, config.modelPlotBoundsPercentile());
        BOOST_REQUIRE_EQUAL(true, config.modelPlotAnnotationsEnabled());
        BOOST_REQUIRE_EQUAL(0, config.modelPlotTerms().size());
    }
    {
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();
        config.configureModelPlot(false, false, EMPTY_STRING);
        BOOST_REQUIRE_EQUAL(-1.0, config.modelPlotBoundsPercentile());
        BOOST_REQUIRE_EQUAL(false, config.modelPlotAnnotationsEnabled());
        BOOST_REQUIRE_EQUAL(0, config.modelPlotTerms().size());
    }
    {
        const std::string terms{"foo,bar,baz"};
        const TStrSet termSet{"foo", "bar", "baz"};
        CAnomalyDetectorModelConfig config = CAnomalyDetectorModelConfig::defaultConfig();
        config.configureModelPlot(true, false, terms);
        BOOST_REQUIRE_EQUAL(ml::maths::CModel::DEFAULT_BOUNDS_PERCENTILE,
                            config.modelPlotBoundsPercentile());
        BOOST_REQUIRE_EQUAL(false, config.modelPlotAnnotationsEnabled());
        BOOST_TEST_REQUIRE(termSet == config.modelPlotTerms());
    }
}

BOOST_AUTO_TEST_SUITE_END()
