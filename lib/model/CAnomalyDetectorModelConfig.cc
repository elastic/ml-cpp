/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnomalyDetectorModelConfig.h>

#include <core/CContainerPrinter.h>
#include <core/CStrCaseCmp.h>
#include <core/CStreamUtils.h>
#include <core/Constants.h>

#include <maths/CMultivariatePrior.h>
#include <maths/CTimeSeriesModel.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <core/CRegex.h>
#include <model/CCountingModelFactory.h>
#include <model/CDetectionRule.h>
#include <model/CEventRateModelFactory.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CLimits.h>
#include <model/CMetricModelFactory.h>
#include <model/CMetricPopulationModelFactory.h>
#include <model/CSearchKey.h>
#include <model/FunctionTypes.h>

#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <fstream>

namespace ml {
namespace model {

namespace {

const CAnomalyDetectorModelConfig::TIntDetectionRuleVecUMap EMPTY_RULES_MAP;
const CAnomalyDetectorModelConfig::TStrDetectionRulePrVec EMPTY_EVENTS;

namespace detail {

core_t::TTime validateBucketLength(core_t::TTime length) {
    // A zero or negative length is used by the individual commands to request
    // the default length - this avoids the need for the commands to know the
    // default length
    return length <= 0 ? CAnomalyDetectorModelConfig::DEFAULT_BUCKET_LENGTH : length;
}
}
}

const std::string CAnomalyDetectorModelConfig::DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER(",");
const core_t::TTime CAnomalyDetectorModelConfig::DEFAULT_BUCKET_LENGTH(300);
const std::size_t CAnomalyDetectorModelConfig::DEFAULT_LATENCY_BUCKETS(0);
const std::size_t CAnomalyDetectorModelConfig::DEFAULT_SAMPLE_COUNT_FACTOR_NO_LATENCY(1);
const std::size_t CAnomalyDetectorModelConfig::DEFAULT_SAMPLE_COUNT_FACTOR_WITH_LATENCY(10);
const double CAnomalyDetectorModelConfig::DEFAULT_SAMPLE_QUEUE_GROWTH_FACTOR(0.1);
const core_t::TTime CAnomalyDetectorModelConfig::STANDARD_BUCKET_LENGTH(1800);
const double CAnomalyDetectorModelConfig::DEFAULT_DECAY_RATE(0.0005);
const double CAnomalyDetectorModelConfig::DEFAULT_INITIAL_DECAY_RATE_MULTIPLIER(4.0);
const double CAnomalyDetectorModelConfig::DEFAULT_LEARN_RATE(1.0);
const double CAnomalyDetectorModelConfig::DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION(0.05);
const double CAnomalyDetectorModelConfig::DEFAULT_POPULATION_MINIMUM_MODE_FRACTION(0.05);
const double CAnomalyDetectorModelConfig::DEFAULT_MINIMUM_CLUSTER_SPLIT_COUNT(12.0);
const double CAnomalyDetectorModelConfig::DEFAULT_CATEGORY_DELETE_FRACTION(0.8);
const std::size_t CAnomalyDetectorModelConfig::DEFAULT_COMPONENT_SIZE(36u);
const core_t::TTime
    CAnomalyDetectorModelConfig::DEFAULT_MINIMUM_TIME_TO_DETECT_CHANGE(core::constants::DAY);
const core_t::TTime
    CAnomalyDetectorModelConfig::DEFAULT_MAXIMUM_TIME_TO_TEST_FOR_CHANGE(2 * core::constants::DAY);
const std::size_t CAnomalyDetectorModelConfig::MULTIBUCKET_FEATURES_WINDOW_LENGTH(12);
const double CAnomalyDetectorModelConfig::MAXIMUM_MULTI_BUCKET_IMPACT_MAGNITUDE(5.0);
const double CAnomalyDetectorModelConfig::DEFAULT_MAXIMUM_UPDATES_PER_BUCKET(1.0);
const double CAnomalyDetectorModelConfig::DEFAULT_INFLUENCE_CUTOFF(0.4);
const double CAnomalyDetectorModelConfig::DEFAULT_PRUNE_WINDOW_SCALE_MINIMUM(0.25);
const double CAnomalyDetectorModelConfig::DEFAULT_PRUNE_WINDOW_SCALE_MAXIMUM(4.0);
const double CAnomalyDetectorModelConfig::DEFAULT_CORRELATION_MODELS_OVERHEAD(3.0);
const double CAnomalyDetectorModelConfig::DEFAULT_MINIMUM_SIGNIFICANT_CORRELATION(0.3);
const double CAnomalyDetectorModelConfig::DEFAULT_AGGREGATION_STYLE_PARAMS[][model_t::NUMBER_AGGREGATION_PARAMS] =
    {{0.0, 1.0, 1.0, 1.0}, {0.5, 0.5, 1.0, 5.0}, {0.5, 0.5, 1.0, 1.0}};
// The default for maximumanomalousprobability now matches the default
// for unusualprobabilitythreshold in mllimits.conf - this avoids
// inconsistencies in output
const double CAnomalyDetectorModelConfig::DEFAULT_MAXIMUM_ANOMALOUS_PROBABILITY(0.035);
const double CAnomalyDetectorModelConfig::DEFAULT_NOISE_PERCENTILE(50.0);
const double CAnomalyDetectorModelConfig::DEFAULT_NOISE_MULTIPLIER(1.0);
const CAnomalyDetectorModelConfig::TDoubleDoublePr CAnomalyDetectorModelConfig::DEFAULT_NORMALIZED_SCORE_KNOT_POINTS[9] = {
    CAnomalyDetectorModelConfig::TDoubleDoublePr(0.0, 0.0),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(70.0, 1.0),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(85.0, 1.2),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(90.0, 1.5),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(95.0, 3.0),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(97.0, 20.0),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(99.0, 50.0),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(99.9, 90.0),
    CAnomalyDetectorModelConfig::TDoubleDoublePr(100.0, 100.0)};

CAnomalyDetectorModelConfig
CAnomalyDetectorModelConfig::defaultConfig(core_t::TTime bucketLength,
                                           model_t::ESummaryMode summaryMode,
                                           const std::string& summaryCountFieldName,
                                           core_t::TTime latency,
                                           bool multivariateByFields) {
    bucketLength = detail::validateBucketLength(bucketLength);

    double learnRate = DEFAULT_LEARN_RATE * bucketNormalizationFactor(bucketLength);
    double decayRate = DEFAULT_DECAY_RATE * bucketNormalizationFactor(bucketLength);

    SModelParams params(bucketLength);
    params.s_LearnRate = learnRate;
    params.s_DecayRate = decayRate;
    params.s_ExcludeFrequent = model_t::E_XF_None;
    params.configureLatency(latency, bucketLength);

    TInterimBucketCorrectorPtr interimBucketCorrector =
        std::make_shared<CInterimBucketCorrector>(bucketLength);

    TFactoryTypeFactoryPtrMap factories;
    params.s_MinimumModeFraction = DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;
    factories[E_EventRateFactory] = std::make_shared<CEventRateModelFactory>(
        params, interimBucketCorrector, summaryMode, summaryCountFieldName);
    factories[E_MetricFactory] = std::make_shared<CMetricModelFactory>(
        params, interimBucketCorrector, summaryMode, summaryCountFieldName);
    factories[E_EventRatePopulationFactory] = std::make_shared<CEventRatePopulationModelFactory>(
        params, interimBucketCorrector, summaryMode, summaryCountFieldName);
    params.s_MinimumModeFraction = DEFAULT_POPULATION_MINIMUM_MODE_FRACTION;
    factories[E_MetricPopulationFactory] = std::make_shared<CMetricPopulationModelFactory>(
        params, interimBucketCorrector, summaryMode, summaryCountFieldName);
    params.s_MinimumModeFraction = 1.0;
    factories[E_CountingFactory] = std::make_shared<CCountingModelFactory>(
        params, interimBucketCorrector, summaryMode, summaryCountFieldName);

    CAnomalyDetectorModelConfig result;
    result.bucketLength(bucketLength);
    result.interimBucketCorrector(interimBucketCorrector);
    result.multivariateByFields(multivariateByFields);
    result.factories(factories);
    return result;
}

// De-rates the decay and learn rate to account for differences from the
// standard bucket length.
double CAnomalyDetectorModelConfig::bucketNormalizationFactor(core_t::TTime bucketLength) {
    return std::min(1.0, static_cast<double>(bucketLength) /
                             static_cast<double>(STANDARD_BUCKET_LENGTH));
}

// Standard decay rate for time series decompositions given the specified
// model decay rate and bucket length.
double CAnomalyDetectorModelConfig::trendDecayRate(double modelDecayRate,
                                                   core_t::TTime bucketLength) {
    double scale = static_cast<double>(bucketLength / 24 / STANDARD_BUCKET_LENGTH);
    return std::min(24.0 * modelDecayRate / bucketNormalizationFactor(bucketLength) /
                        std::max(scale, 1.0),
                    0.1);
}

CAnomalyDetectorModelConfig::CAnomalyDetectorModelConfig()
    : m_BucketLength(STANDARD_BUCKET_LENGTH), m_MultivariateByFields(false),
      m_ModelPlotBoundsPercentile(-1.0),
      m_MaximumAnomalousProbability(DEFAULT_MAXIMUM_ANOMALOUS_PROBABILITY),
      m_NoisePercentile(DEFAULT_NOISE_PERCENTILE),
      m_NoiseMultiplier(DEFAULT_NOISE_MULTIPLIER),
      m_NormalizedScoreKnotPoints(std::begin(DEFAULT_NORMALIZED_SCORE_KNOT_POINTS),
                                  std::end(DEFAULT_NORMALIZED_SCORE_KNOT_POINTS)),
      m_DetectionRules(EMPTY_RULES_MAP), m_ScheduledEvents(EMPTY_EVENTS) {
    for (std::size_t i = 0u; i < model_t::NUMBER_AGGREGATION_STYLES; ++i) {
        for (std::size_t j = 0u; j < model_t::NUMBER_AGGREGATION_PARAMS; ++j) {
            m_AggregationStyleParams[i][j] = DEFAULT_AGGREGATION_STYLE_PARAMS[i][j];
        }
    }
}

void CAnomalyDetectorModelConfig::bucketLength(core_t::TTime length) {
    m_BucketLength = length;
    for (auto& factory : m_Factories) {
        factory.second->updateBucketLength(length);
    }
}

void CAnomalyDetectorModelConfig::interimBucketCorrector(const TInterimBucketCorrectorPtr& interimBucketCorrector) {
    m_InterimBucketCorrector = interimBucketCorrector;
    for (auto& factory : m_Factories) {
        factory.second->interimBucketCorrector(m_InterimBucketCorrector);
    }
}

void CAnomalyDetectorModelConfig::useMultibucketFeatures(bool enabled) {
    for (auto& factory : m_Factories) {
        factory.second->multibucketFeaturesWindowLength(
            enabled ? MULTIBUCKET_FEATURES_WINDOW_LENGTH : 0);
    }
}

void CAnomalyDetectorModelConfig::multivariateByFields(bool enabled) {
    m_MultivariateByFields = enabled;
}

void CAnomalyDetectorModelConfig::factories(const TFactoryTypeFactoryPtrMap& factories) {
    m_Factories = factories;
}

bool CAnomalyDetectorModelConfig::aggregationStyleParams(model_t::EAggregationStyle style,
                                                         model_t::EAggregationParam param,
                                                         double value) {
    switch (param) {
    case model_t::E_JointProbabilityWeight:
        if (value < 0.0 || value > 1.0) {
            LOG_ERROR(<< "joint probability weight " << value << " out of in range [0,1]");
            return false;
        }
        m_AggregationStyleParams[style][model_t::E_JointProbabilityWeight] = value;
        break;
    case model_t::E_ExtremeProbabilityWeight:
        if (value < 0.0 || value > 1.0) {
            LOG_ERROR(<< "extreme probability weight " << value << " out of in range [0,1]");
            return false;
        }
        m_AggregationStyleParams[style][model_t::E_ExtremeProbabilityWeight] = value;
        break;
    case model_t::E_MinExtremeSamples:
        if (value < 1.0 || value > 10.0) {
            LOG_ERROR(<< "min extreme samples " << value << " out of in range [0,10]");
            return false;
        }
        m_AggregationStyleParams[style][model_t::E_MinExtremeSamples] = value;
        m_AggregationStyleParams[style][model_t::E_MaxExtremeSamples] = std::max(
            value, m_AggregationStyleParams[style][model_t::E_MaxExtremeSamples]);
        break;
    case model_t::E_MaxExtremeSamples:
        if (value < 1.0 || value > 10.0) {
            LOG_ERROR(<< "max extreme samples " << value << " out of in range [0,10]");
            return false;
        }
        m_AggregationStyleParams[style][model_t::E_MaxExtremeSamples] = value;
        m_AggregationStyleParams[style][model_t::E_MinExtremeSamples] = std::min(
            value, m_AggregationStyleParams[style][model_t::E_MinExtremeSamples]);
        break;
    }
    return true;
}

void CAnomalyDetectorModelConfig::maximumAnomalousProbability(double probability) {
    double minimum = 100 * maths::MINUSCULE_PROBABILITY;
    if (probability < minimum || probability > 1.0) {
        LOG_INFO(<< "Maximum anomalous probability " << probability
                 << " out of range [" << minimum << "," << 1.0 << "] truncating");
    }
    m_MaximumAnomalousProbability = maths::CTools::truncate(probability, minimum, 1.0);
}

bool CAnomalyDetectorModelConfig::noisePercentile(double percentile) {
    if (percentile < 0.0 || percentile > 100.0) {
        LOG_ERROR(<< "Noise percentile " << percentile << " out of range [0, 100]");
        return false;
    }
    m_NoisePercentile = percentile;
    return true;
}

bool CAnomalyDetectorModelConfig::noiseMultiplier(double multiplier) {
    if (multiplier <= 0.0) {
        LOG_ERROR(<< "Noise multiplier must be positive");
        return false;
    }
    m_NoiseMultiplier = multiplier;
    return true;
}

bool CAnomalyDetectorModelConfig::normalizedScoreKnotPoints(const TDoubleDoublePrVec& points) {
    if (points.empty()) {
        LOG_ERROR(<< "Must provide at least two know points");
        return false;
    }
    if (points[0].first != 0.0 && points[0].second != 0.0) {
        LOG_ERROR(<< "First knot point must be (0,0)");
        return false;
    }
    if (points.back().first != 100.0 && points.back().second != 100.0) {
        LOG_ERROR(<< "Last knot point must be (100,100)");
        return false;
    }
    for (std::size_t i = 0u; i < points.size(); i += 2) {
        if (points[i].first < 0.0 || points[i].first > 100.0) {
            LOG_ERROR(<< "Unexpected value " << points[i].first << " for percentile");
            return false;
        }
        if (points[i].second < 0.0 || points[i].second > 100.0) {
            LOG_ERROR(<< "Unexpected value " << points[i].second << " for score");
            return false;
        }
    }
    if (!std::is_sorted(points.begin(), points.end(), maths::COrderings::SFirstLess())) {
        LOG_ERROR(<< "Percentiles must be monotonic increasing "
                  << core::CContainerPrinter::print(points));
        return false;
    }
    if (!std::is_sorted(points.begin(), points.end(), maths::COrderings::SSecondLess())) {
        LOG_ERROR(<< "Scores must be monotonic increasing "
                  << core::CContainerPrinter::print(points));
        return false;
    }

    m_NormalizedScoreKnotPoints = points;
    m_NormalizedScoreKnotPoints.erase(std::unique(m_NormalizedScoreKnotPoints.begin(),
                                                  m_NormalizedScoreKnotPoints.end()),
                                      m_NormalizedScoreKnotPoints.end());
    return true;
}

bool CAnomalyDetectorModelConfig::init(const std::string& configFile) {
    boost::property_tree::ptree propTree;
    return this->init(configFile, propTree);
}

bool CAnomalyDetectorModelConfig::init(const std::string& configFile,
                                       boost::property_tree::ptree& propTree) {
    LOG_DEBUG(<< "Reading config file " << configFile);

    try {
        std::ifstream strm(configFile.c_str());
        if (!strm.is_open()) {
            LOG_ERROR(<< "Error opening config file " << configFile);
            return false;
        }
        core::CStreamUtils::skipUtf8Bom(strm);

        boost::property_tree::ini_parser::read_ini(strm, propTree);
    } catch (boost::property_tree::ptree_error& e) {
        LOG_ERROR(<< "Error reading config file " << configFile << " : " << e.what());
        return false;
    }

    if (this->init(propTree) == false) {
        LOG_ERROR(<< "Error reading config file " << configFile);
        return false;
    }

    return true;
}

bool CAnomalyDetectorModelConfig::init(const boost::property_tree::ptree& propTree) {
    static const std::string MODEL_STANZA("model");
    static const std::string ANOMALY_SCORE_STANZA("anomalyscore");

    bool result = true;

    for (boost::property_tree::ptree::const_iterator i = propTree.begin();
         i != propTree.end(); ++i) {
        const std::string& stanzaName = i->first;
        const boost::property_tree::ptree& propertyTree = i->second;

        if (stanzaName == MODEL_STANZA) {
            if (this->processStanza(propertyTree) == false) {
                LOG_ERROR(<< "Error reading model config stanza: " << MODEL_STANZA);
                result = false;
            }
        } else if (stanzaName == ANOMALY_SCORE_STANZA) {
            if (this->processStanza(propertyTree) == false) {
                LOG_ERROR(<< "Error reading model config stanza: " << ANOMALY_SCORE_STANZA);
                result = false;
            }
        } else {
            LOG_WARN(<< "Ignoring unknown model config stanza: " << stanzaName);
        }
    }

    return result;
}

bool CAnomalyDetectorModelConfig::configureModelPlot(const std::string& modelPlotConfigFile) {
    LOG_DEBUG(<< "Reading model plot config file " << modelPlotConfigFile);

    boost::property_tree::ptree propTree;
    try {
        std::ifstream strm(modelPlotConfigFile.c_str());
        if (!strm.is_open()) {
            LOG_ERROR(<< "Error opening model plot config file " << modelPlotConfigFile);
            return false;
        }
        core::CStreamUtils::skipUtf8Bom(strm);

        boost::property_tree::ini_parser::read_ini(strm, propTree);
    } catch (boost::property_tree::ptree_error& e) {
        LOG_ERROR(<< "Error reading model plot config file "
                  << modelPlotConfigFile << " : " << e.what());
        return false;
    }

    if (this->configureModelPlot(propTree) == false) {
        LOG_ERROR(<< "Error reading model plot config file " << modelPlotConfigFile);
        return false;
    }

    return true;
}

namespace {
// Model debug config properties
const std::string BOUNDS_PERCENTILE_PROPERTY("boundspercentile");
const std::string TERMS_PROPERTY("terms");
}

bool CAnomalyDetectorModelConfig::configureModelPlot(const boost::property_tree::ptree& propTree) {
    try {
        std::string valueStr(propTree.get<std::string>(BOUNDS_PERCENTILE_PROPERTY));
        if (core::CStringUtils::stringToType(valueStr, m_ModelPlotBoundsPercentile) == false) {
            LOG_ERROR(<< "Cannot parse as double: " << valueStr);
            return false;
        }
    } catch (boost::property_tree::ptree_error&) {
        LOG_ERROR(<< "Error reading model debug config. Property '"
                  << BOUNDS_PERCENTILE_PROPERTY << "' is missing");
        return false;
    }

    m_ModelPlotTerms.clear();
    try {
        std::string valueStr(propTree.get<std::string>(TERMS_PROPERTY));

        TStrVec tokens;
        std::string remainder;
        core::CStringUtils::tokenise(",", valueStr, tokens, remainder);
        if (!remainder.empty()) {
            tokens.push_back(remainder);
        }
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            m_ModelPlotTerms.insert(tokens[i]);
        }
    } catch (boost::property_tree::ptree_error&) {
        LOG_ERROR(<< "Error reading model debug config. Property '"
                  << TERMS_PROPERTY << "' is missing");
        return false;
    }

    return true;
}

CAnomalyDetectorModelConfig::TModelFactoryCPtr
CAnomalyDetectorModelConfig::factory(const CSearchKey& key) const {
    TModelFactoryCPtr result = m_FactoryCache[key];
    if (!result) {
        result = key.isSimpleCount()
                     ? this->factory(key.identifier(), key.function(), true,
                                     key.excludeFrequent(), key.partitionFieldName(),
                                     key.overFieldName(), key.byFieldName(),
                                     key.fieldName(), key.influenceFieldNames())
                     : this->factory(key.identifier(), key.function(), key.useNull(),
                                     key.excludeFrequent(), key.partitionFieldName(),
                                     key.overFieldName(), key.byFieldName(),
                                     key.fieldName(), key.influenceFieldNames());
    }
    return result;
}

CAnomalyDetectorModelConfig::TModelFactoryCPtr
CAnomalyDetectorModelConfig::factory(int identifier,
                                     function_t::EFunction function,
                                     bool useNull,
                                     model_t::EExcludeFrequent excludeFrequent,
                                     const std::string& partitionFieldName,
                                     const std::string& overFieldName,
                                     const std::string& byFieldName,
                                     const std::string& valueFieldName,
                                     const CSearchKey::TStoredStringPtrVec& influenceFieldNames) const {
    const TFeatureVec& features = function_t::features(function);

    // Simple state machine to deduce the factory type from
    // a collection of features.
    EFactoryType factory = E_UnknownFactory;
    for (std::size_t i = 0u; i < features.size(); ++i) {
        switch (factory) {
        case E_EventRateFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
                break;
            case model_t::E_Metric:
                factory = E_MetricFactory;
                break;
            case model_t::E_PopulationEventRate:
            case model_t::E_PopulationMetric:
            case model_t::E_PeersEventRate:
            case model_t::E_PeersMetric:
                factory = E_BadFactory;
                break;
            }
            break;

        case E_MetricFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
            case model_t::E_Metric:
                break;
            case model_t::E_PopulationEventRate:
            case model_t::E_PopulationMetric:
            case model_t::E_PeersEventRate:
            case model_t::E_PeersMetric:
                factory = E_BadFactory;
                break;
            }
            break;

        case E_EventRatePopulationFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
            case model_t::E_Metric:
                factory = E_BadFactory;
                break;
            case model_t::E_PopulationEventRate:
                break;
            case model_t::E_PopulationMetric:
            case model_t::E_PeersEventRate:
            case model_t::E_PeersMetric:
                factory = E_BadFactory;
                break;
            }
            break;

        case E_MetricPopulationFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
            case model_t::E_Metric:
            case model_t::E_PopulationEventRate:
                factory = E_BadFactory;
                break;
            case model_t::E_PopulationMetric:
                factory = E_MetricPopulationFactory;
                break;
            case model_t::E_PeersEventRate:
            case model_t::E_PeersMetric:
                factory = E_BadFactory;
                break;
            }
            break;

        case E_EventRatePeersFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
            case model_t::E_Metric:
            case model_t::E_PopulationEventRate:
            case model_t::E_PopulationMetric:
                factory = E_BadFactory;
                break;
            case model_t::E_PeersEventRate:
                break;
            case model_t::E_PeersMetric:
                factory = E_BadFactory;
                break;
            }
            break;

        case E_CountingFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
            case model_t::E_Metric:
            case model_t::E_PopulationEventRate:
            case model_t::E_PopulationMetric:
            case model_t::E_PeersEventRate:
            case model_t::E_PeersMetric:
                factory = E_BadFactory;
                break;
            }
            break;

        case E_UnknownFactory:
            switch (model_t::analysisCategory(features[i])) {
            case model_t::E_EventRate:
                factory = CSearchKey::isSimpleCount(function, byFieldName)
                              ? E_CountingFactory
                              : E_EventRateFactory;
                break;
            case model_t::E_Metric:
                factory = E_MetricFactory;
                break;
            case model_t::E_PopulationEventRate:
                factory = E_EventRatePopulationFactory;
                break;
            case model_t::E_PopulationMetric:
                factory = E_MetricPopulationFactory;
                break;
            case model_t::E_PeersEventRate:
                factory = E_EventRatePeersFactory;
                break;
            case model_t::E_PeersMetric:
                // TODO
                factory = E_BadFactory;
                break;
            }
            break;

        case E_BadFactory:
            break;
        }
    }

    TFactoryTypeFactoryPtrMapCItr prototype = m_Factories.find(factory);
    if (prototype == m_Factories.end()) {
        LOG_ABORT(<< "No factory for features = "
                  << core::CContainerPrinter::print(features));
    }

    TModelFactoryPtr result(prototype->second->clone());
    result->identifier(identifier);
    TStrVec influences;
    influences.reserve(influenceFieldNames.size());
    for (const auto& influenceFieldName : influenceFieldNames) {
        influences.push_back(*influenceFieldName);
    }
    result->fieldNames(partitionFieldName, overFieldName, byFieldName,
                       valueFieldName, influences);
    result->useNull(useNull);
    result->excludeFrequent(excludeFrequent);
    result->features(features);
    result->multivariateByFields(m_MultivariateByFields);
    TIntDetectionRuleVecUMapCItr rulesItr = m_DetectionRules.get().find(identifier);
    if (rulesItr != m_DetectionRules.get().end()) {
        result->detectionRules(TDetectionRuleVecCRef(rulesItr->second));
    }
    result->scheduledEvents(m_ScheduledEvents);

    return result;
}

void CAnomalyDetectorModelConfig::decayRate(double value) {
    for (auto& factory : m_Factories) {
        factory.second->decayRate(value);
    }
}

double CAnomalyDetectorModelConfig::decayRate() const {
    return m_Factories.begin()->second->modelParams().s_DecayRate;
}

core_t::TTime CAnomalyDetectorModelConfig::bucketLength() const {
    return m_BucketLength;
}

core_t::TTime CAnomalyDetectorModelConfig::latency() const {
    return m_BucketLength * m_Factories.begin()->second->modelParams().s_LatencyBuckets;
}

std::size_t CAnomalyDetectorModelConfig::latencyBuckets() const {
    return m_Factories.begin()->second->modelParams().s_LatencyBuckets;
}

const CInterimBucketCorrector& CAnomalyDetectorModelConfig::interimBucketCorrector() const {
    return *m_InterimBucketCorrector;
}

bool CAnomalyDetectorModelConfig::multivariateByFields() const {
    return m_MultivariateByFields;
}

void CAnomalyDetectorModelConfig::modelPlotBoundsPercentile(double percentile) {
    if (percentile < 0.0 || percentile >= 100.0) {
        LOG_ERROR(<< "Bad confidence interval");
        return;
    }
    m_ModelPlotBoundsPercentile = percentile;
}

double CAnomalyDetectorModelConfig::modelPlotBoundsPercentile() const {
    return m_ModelPlotBoundsPercentile;
}

void CAnomalyDetectorModelConfig::modelPlotTerms(TStrSet terms) {
    m_ModelPlotTerms.swap(terms);
}

const CAnomalyDetectorModelConfig::TStrSet& CAnomalyDetectorModelConfig::modelPlotTerms() const {
    return m_ModelPlotTerms;
}

double CAnomalyDetectorModelConfig::aggregationStyleParam(model_t::EAggregationStyle style,
                                                          model_t::EAggregationParam param) const {
    return m_AggregationStyleParams[style][param];
}

double CAnomalyDetectorModelConfig::maximumAnomalousProbability() const {
    return m_MaximumAnomalousProbability;
}

double CAnomalyDetectorModelConfig::noisePercentile() const {
    return m_NoisePercentile;
}

double CAnomalyDetectorModelConfig::noiseMultiplier() const {
    return m_NoiseMultiplier;
}

const CAnomalyDetectorModelConfig::TDoubleDoublePrVec&
CAnomalyDetectorModelConfig::normalizedScoreKnotPoints() const {
    return m_NormalizedScoreKnotPoints;
}

void CAnomalyDetectorModelConfig::detectionRules(TIntDetectionRuleVecUMapCRef detectionRules) {
    m_DetectionRules = detectionRules;
}

void CAnomalyDetectorModelConfig::scheduledEvents(TStrDetectionRulePrVecCRef scheduledEvents) {
    m_ScheduledEvents = scheduledEvents;
}

core_t::TTime CAnomalyDetectorModelConfig::samplingAgeCutoff() const {
    return m_Factories.begin()->second->modelParams().s_SamplingAgeCutoff;
}

namespace {
const std::string ONLINE_LEARN_RATE_PROPERTY("learnrate");
const std::string DECAY_RATE_PROPERTY("decayrate");
const std::string INITIAL_DECAY_RATE_MULTIPLIER_PROPERTY("initialdecayratemultiplier");
const std::string MAXIMUM_UPDATES_PER_BUCKET_PROPERTY("maximumupdatesperbucket");
const std::string INDIVIDUAL_MODE_FRACTION_PROPERTY("individualmodefraction");
const std::string POPULATION_MODE_FRACTION_PROPERTY("populationmodefraction");
const std::string PEERS_MODE_FRACTION_PROPERTY("peersmodefraction");
const std::string COMPONENT_SIZE_PROPERTY("componentsize");
const std::string SAMPLE_COUNT_FACTOR_PROPERTY("samplecountfactor");
const std::string PRUNE_WINDOW_SCALE_MINIMUM("prunewindowscaleminimum");
const std::string PRUNE_WINDOW_SCALE_MAXIMUM("prunewindowscalemaximum");
const std::string AGGREGATION_STYLE_PARAMS("aggregationstyleparams");
const std::string MAXIMUM_ANOMALOUS_PROBABILITY_PROPERTY("maximumanomalousprobability");
const std::string NOISE_PERCENTILE_PROPERTY("noisepercentile");
const std::string NOISE_MULTIPLIER_PROPERTY("noisemultiplier");
const std::string NORMALIZED_SCORE_KNOT_POINTS("normalizedscoreknotpoints");
}

bool CAnomalyDetectorModelConfig::processStanza(const boost::property_tree::ptree& propertyTree) {
    bool result = true;

    for (const auto& property : propertyTree) {
        std::string propName = property.first;
        std::string propValue = property.second.data();
        core::CStringUtils::trimWhitespace(propValue);

        if (propName == ONLINE_LEARN_RATE_PROPERTY) {
            double learnRate = DEFAULT_LEARN_RATE;
            if (core::CStringUtils::stringToType(propValue, learnRate) == false ||
                learnRate <= 0.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            learnRate *= bucketNormalizationFactor(this->bucketLength());
            for (auto& factory : m_Factories) {
                factory.second->learnRate(learnRate);
            }
        } else if (propName == DECAY_RATE_PROPERTY) {
            double decayRate = DEFAULT_DECAY_RATE;
            if (core::CStringUtils::stringToType(propValue, decayRate) == false ||
                decayRate <= 0.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            decayRate *= bucketNormalizationFactor(this->bucketLength());
            for (auto& factory : m_Factories) {
                factory.second->decayRate(decayRate);
            }
        } else if (propName == INITIAL_DECAY_RATE_MULTIPLIER_PROPERTY) {
            double multiplier = DEFAULT_INITIAL_DECAY_RATE_MULTIPLIER;
            if (core::CStringUtils::stringToType(propValue, multiplier) == false ||
                multiplier < 1.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            for (auto& factory : m_Factories) {
                factory.second->initialDecayRateMultiplier(multiplier);
            }
        } else if (propName == MAXIMUM_UPDATES_PER_BUCKET_PROPERTY) {
            double maximumUpdatesPerBucket;
            if (core::CStringUtils::stringToType(propValue, maximumUpdatesPerBucket) == false ||
                maximumUpdatesPerBucket < 0.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            for (auto& factory : m_Factories) {
                factory.second->maximumUpdatesPerBucket(maximumUpdatesPerBucket);
            }
        } else if (propName == INDIVIDUAL_MODE_FRACTION_PROPERTY) {
            double fraction;
            if (core::CStringUtils::stringToType(propValue, fraction) == false ||
                fraction < 0.0 || fraction > 1.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            if (m_Factories.count(E_EventRateFactory) > 0) {
                m_Factories[E_EventRateFactory]->minimumModeFraction(fraction);
            }
            if (m_Factories.count(E_MetricFactory) > 0) {
                m_Factories[E_MetricFactory]->minimumModeFraction(fraction);
            }
        } else if (propName == POPULATION_MODE_FRACTION_PROPERTY) {
            double fraction;
            if (core::CStringUtils::stringToType(propValue, fraction) == false ||
                fraction < 0.0 || fraction > 1.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            if (m_Factories.count(E_EventRatePopulationFactory) > 0) {
                m_Factories[E_EventRatePopulationFactory]->minimumModeFraction(fraction);
            }
            if (m_Factories.count(E_MetricPopulationFactory) > 0) {
                m_Factories[E_MetricPopulationFactory]->minimumModeFraction(fraction);
            }
        } else if (propName == PEERS_MODE_FRACTION_PROPERTY) {
            double fraction;
            if (core::CStringUtils::stringToType(propValue, fraction) == false ||
                fraction < 0.0 || fraction > 1.0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }

            if (m_Factories.count(E_EventRatePeersFactory) > 0) {
                m_Factories[E_EventRatePeersFactory]->minimumModeFraction(fraction);
            }
        } else if (propName == COMPONENT_SIZE_PROPERTY) {
            int componentSize;
            if (core::CStringUtils::stringToType(propValue, componentSize) == false ||
                componentSize < 0) {
                LOG_ERROR(<< "Invalid value of property " << propName << " : " << propValue);
                result = false;
                continue;
            }
            for (auto& factory : m_Factories) {
                factory.second->componentSize(componentSize);
            }
        } else if (propName == SAMPLE_COUNT_FACTOR_PROPERTY) {
            int factor;
            if (core::CStringUtils::stringToType(propValue, factor) == false || factor < 0) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }
            for (auto& factory : m_Factories) {
                factory.second->sampleCountFactor(factor);
            }
        } else if (propName == PRUNE_WINDOW_SCALE_MINIMUM) {
            double factor;
            if (core::CStringUtils::stringToType(propValue, factor) == false) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }
            for (auto& factory : m_Factories) {
                factory.second->pruneWindowScaleMinimum(factor);
            }
        } else if (propName == PRUNE_WINDOW_SCALE_MAXIMUM) {
            double factor;
            if (core::CStringUtils::stringToType(propValue, factor) == false) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }
            for (auto& factory : m_Factories) {
                factory.second->pruneWindowScaleMaximum(factor);
            }
        } else if (propName == AGGREGATION_STYLE_PARAMS) {
            core::CStringUtils::trimWhitespace(propValue);
            propValue = core::CStringUtils::normaliseWhitespace(propValue);

            TStrVec strings;
            std::string remainder;
            core::CStringUtils::tokenise(" ", propValue, strings, remainder);
            if (!remainder.empty()) {
                strings.push_back(remainder);
            }
            std::size_t n = model_t::NUMBER_AGGREGATION_STYLES * model_t::NUMBER_AGGREGATION_PARAMS;
            if (strings.size() != n) {
                LOG_ERROR(<< "Expected " << n << " values for " << propName);
                result = false;
                continue;
            }
            for (std::size_t j = 0u, l = 0u; j < model_t::NUMBER_AGGREGATION_STYLES; ++j) {
                for (std::size_t k = 0u; k < model_t::NUMBER_AGGREGATION_PARAMS; ++k, ++l) {
                    double value;
                    if (core::CStringUtils::stringToType(strings[l], value) == false) {
                        LOG_ERROR(<< "Unexpected value " << strings[l]
                                  << " in property " << propName);
                        result = false;
                        continue;
                    }

                    this->aggregationStyleParams(
                        static_cast<model_t::EAggregationStyle>(j),
                        static_cast<model_t::EAggregationParam>(k), value);
                }
            }
        } else if (propName == MAXIMUM_ANOMALOUS_PROBABILITY_PROPERTY) {
            double probability;
            if (core::CStringUtils::stringToType(propValue, probability) == false) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }
            this->maximumAnomalousProbability(probability);
        } else if (propName == NOISE_PERCENTILE_PROPERTY) {
            double percentile;
            if (core::CStringUtils::stringToType(propValue, percentile) == false ||
                this->noisePercentile(percentile) == false) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }
        } else if (propName == NOISE_MULTIPLIER_PROPERTY) {
            double multiplier;
            if (core::CStringUtils::stringToType(propValue, multiplier) == false ||
                this->noiseMultiplier(multiplier) == false) {
                LOG_ERROR(<< "Invalid value for property " << propName << " : " << propValue);
                result = false;
                continue;
            }
        } else if (propName == NORMALIZED_SCORE_KNOT_POINTS) {
            core::CStringUtils::trimWhitespace(propValue);
            propValue = core::CStringUtils::normaliseWhitespace(propValue);

            TStrVec strings;
            std::string remainder;
            core::CStringUtils::tokenise(" ", propValue, strings, remainder);
            if (!remainder.empty()) {
                strings.push_back(remainder);
            }
            if (strings.empty() || (strings.size() % 2) != 0) {
                LOG_ERROR(<< "Expected even number of values for property " << propName
                          << " " << core::CContainerPrinter::print(strings));
                result = false;
                continue;
            }

            TDoubleDoublePrVec points;
            points.reserve(strings.size() / 2 + 2);
            points.emplace_back(0.0, 0.0);
            for (std::size_t j = 0u; j < strings.size(); j += 2) {
                double rate;
                double score;
                if (core::CStringUtils::stringToType(strings[j], rate) == false) {
                    LOG_ERROR(<< "Unexpected value " << strings[j]
                              << " for rate in property " << propName);
                    result = false;
                    continue;
                }
                if (core::CStringUtils::stringToType(strings[j + 1], score) == false) {
                    LOG_ERROR(<< "Unexpected value " << strings[j + 1]
                              << " for score in property " << propName);
                    result = false;
                    continue;
                }
                points.emplace_back(rate, score);
            }
            points.emplace_back(100.0, 100.0);
            this->normalizedScoreKnotPoints(points);
        } else {
            LOG_WARN(<< "Ignoring unknown property " << propName);
        }
    }

    return result;
}

double CAnomalyDetectorModelConfig::bucketNormalizationFactor() const {
    return bucketNormalizationFactor(m_BucketLength);
}
}
}
