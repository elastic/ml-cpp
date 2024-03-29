/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/SModelParams.h>

#include <core/CMemoryDefMultiIndex.h>
#include <core/Constants.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CRestoreParams.h>

#include <model/CAnomalyDetectorModelConfig.h>

#include <cmath>

namespace ml {
namespace model {

namespace {
const SModelParams::TDetectionRuleVec EMPTY_RULES;
const SModelParams::TStrDetectionRulePrVec EMPTY_SCHEDULED_EVENTS;
const core_t::TTime SAMPLING_AGE_CUTOFF_DEFAULT(2 * core::constants::DAY);
}

SModelParams::SModelParams(core_t::TTime bucketLength)
    : s_LearnRate(1.0), s_DecayRate(0.0),
      s_InitialDecayRateMultiplier(CAnomalyDetectorModelConfig::DEFAULT_INITIAL_DECAY_RATE_MULTIPLIER),
      s_ControlDecayRate(true), s_MinimumModeFraction(0.0),
      s_MinimumModeCount(CAnomalyDetectorModelConfig::DEFAULT_MINIMUM_CLUSTER_SPLIT_COUNT),
      s_ComponentSize(CAnomalyDetectorModelConfig::DEFAULT_COMPONENT_SIZE),
      s_MinimumTimeToDetectChange(CAnomalyDetectorModelConfig::DEFAULT_MINIMUM_TIME_TO_DETECT_CHANGE),
      s_MaximumTimeToTestForChange(CAnomalyDetectorModelConfig::DEFAULT_MAXIMUM_TIME_TO_TEST_FOR_CHANGE),
      s_MultibucketFeaturesWindowLength(CAnomalyDetectorModelConfig::MULTIBUCKET_FEATURES_WINDOW_LENGTH),
      s_MultivariateByFields(false),
      s_CorrelationModelsOverhead(CAnomalyDetectorModelConfig::DEFAULT_CORRELATION_MODELS_OVERHEAD),
      s_MinimumSignificantCorrelation(
          CAnomalyDetectorModelConfig::DEFAULT_MINIMUM_SIGNIFICANT_CORRELATION),
      s_BucketLength(bucketLength),
      s_MultivariateComponentDelimiter(
          CAnomalyDetectorModelConfig::DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER),
      s_ExcludeFrequent(model_t::E_XF_None), s_ExcludePersonFrequency(0.1),
      s_ExcludeAttributeFrequency(0.1),
      s_MaximumUpdatesPerBucket(CAnomalyDetectorModelConfig::DEFAULT_MAXIMUM_UPDATES_PER_BUCKET),
      s_LatencyBuckets(CAnomalyDetectorModelConfig::DEFAULT_LATENCY_BUCKETS),
      s_SampleCountFactor(CAnomalyDetectorModelConfig::DEFAULT_SAMPLE_COUNT_FACTOR_NO_LATENCY),
      s_SampleQueueGrowthFactor(CAnomalyDetectorModelConfig::DEFAULT_SAMPLE_QUEUE_GROWTH_FACTOR),
      s_SamplingAgeCutoff(SAMPLING_AGE_CUTOFF_DEFAULT),
      s_PruneWindowScaleMinimum(CAnomalyDetectorModelConfig::DEFAULT_PRUNE_WINDOW_SCALE_MINIMUM),
      s_PruneWindowScaleMaximum(CAnomalyDetectorModelConfig::DEFAULT_PRUNE_WINDOW_SCALE_MAXIMUM),
      s_DetectionRules(EMPTY_RULES), s_ScheduledEvents(EMPTY_SCHEDULED_EVENTS),
      s_InfluenceCutoff(CAnomalyDetectorModelConfig::DEFAULT_INFLUENCE_CUTOFF),
      s_MinimumToFuzzyDeduplicate(10000), s_CacheProbabilities(true),
      s_AnnotationsEnabled(false) {
}

void SModelParams::configureLatency(core_t::TTime latency, core_t::TTime bucketLength) {
    s_LatencyBuckets = (latency + bucketLength - 1) / bucketLength;
    if (s_LatencyBuckets > 0) {
        s_SampleCountFactor = CAnomalyDetectorModelConfig::DEFAULT_SAMPLE_COUNT_FACTOR_WITH_LATENCY;
        if (s_LatencyBuckets > 50) {
            LOG_WARN(<< "There are a large number of buckets in the latency window. "
                        "Please ensure sufficient resources are available for this job.");
        }
    }
}

double SModelParams::minimumCategoryCount() const {
    return s_LearnRate * CAnomalyDetectorModelConfig::DEFAULT_CATEGORY_DELETE_FRACTION;
}

maths::common::STimeSeriesDecompositionRestoreParams
SModelParams::decompositionRestoreParams(maths_t::EDataType dataType) const {
    double decayRate{CAnomalyDetectorModelConfig::trendDecayRate(s_DecayRate, s_BucketLength)};
    return {decayRate, s_BucketLength, s_ComponentSize,
            this->distributionRestoreParams(dataType)};
}

maths::common::SDistributionRestoreParams
SModelParams::distributionRestoreParams(maths_t::EDataType dataType) const {
    return {dataType, s_DecayRate, s_MinimumModeFraction, s_MinimumModeCount,
            this->minimumCategoryCount()};
}

std::uint64_t SModelParams::checksum(std::uint64_t seed) const {
    seed = maths::common::CChecksum::calculate(seed, s_LearnRate);
    seed = maths::common::CChecksum::calculate(seed, s_DecayRate);
    seed = maths::common::CChecksum::calculate(seed, s_InitialDecayRateMultiplier);
    seed = maths::common::CChecksum::calculate(seed, s_MinimumModeFraction);
    seed = maths::common::CChecksum::calculate(seed, s_MinimumModeCount);
    seed = maths::common::CChecksum::calculate(seed, s_ComponentSize);
    seed = maths::common::CChecksum::calculate(seed, s_MinimumTimeToDetectChange);
    seed = maths::common::CChecksum::calculate(seed, s_MaximumTimeToTestForChange);
    seed = maths::common::CChecksum::calculate(seed, s_ExcludeFrequent);
    seed = maths::common::CChecksum::calculate(seed, s_ExcludePersonFrequency);
    seed = maths::common::CChecksum::calculate(seed, s_ExcludeAttributeFrequency);
    seed = maths::common::CChecksum::calculate(seed, s_MaximumUpdatesPerBucket);
    seed = maths::common::CChecksum::calculate(seed, s_InfluenceCutoff);
    seed = maths::common::CChecksum::calculate(seed, s_LatencyBuckets);
    seed = maths::common::CChecksum::calculate(seed, s_SampleCountFactor);
    seed = maths::common::CChecksum::calculate(seed, s_SampleQueueGrowthFactor);
    seed = maths::common::CChecksum::calculate(seed, s_PruneWindowScaleMinimum);
    seed = maths::common::CChecksum::calculate(seed, s_PruneWindowScaleMaximum);
    seed = maths::common::CChecksum::calculate(seed, s_CorrelationModelsOverhead);
    seed = maths::common::CChecksum::calculate(seed, s_MultivariateByFields);
    seed = maths::common::CChecksum::calculate(seed, s_MinimumSignificantCorrelation);
    //seed = maths::common::CChecksum::calculate(seed, s_DetectionRules);
    //seed = maths::common::CChecksum::calculate(seed, s_ScheduledEvents);
    seed = maths::common::CChecksum::calculate(seed, s_MinimumToFuzzyDeduplicate);
    return maths::common::CChecksum::calculate(seed, s_SamplingAgeCutoff);
}
}
}
