/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CModelParams_h
#define INCLUDED_ml_model_CModelParams_h

#include <core/CLogger.h>
#include <core/CMemoryUsage.h>

#include <maths/MathsTypes.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>
#include <model/CDetectionRule.h>

#include <boost/ref.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace ml
{
namespace maths
{
struct SDistributionRestoreParams;
}
namespace model
{
//! \brief Wraps up model global parameters.
//!
//! DESCIRIPTION:\n
//! The idea of this class is to encapsulate global model configuration
//! to avoid the need of updating the constructor signatures of all the
//! classes in the CModel hierarchy when new parameters added.
//!
//! IMPLEMENTATION:\n
//! This is purposely not implemented as a nested class so that it can
//! be forward declared.
struct MODEL_EXPORT SModelParams
{
    using TDetectionRuleVec = std::vector<CDetectionRule>;
    using TDetectionRuleVecCRef = boost::reference_wrapper<const TDetectionRuleVec>;
    using TStrDetectionRulePr = std::pair<std::string, model::CDetectionRule>;
    using TStrDetectionRulePrVec = std::vector<TStrDetectionRulePr>;
    using TStrDetectionRulePrVecCRef = boost::reference_wrapper<const TStrDetectionRulePrVec>;

    using TTimeVec = std::vector<core_t::TTime>;

    explicit SModelParams(core_t::TTime bucketLength);

    //! Calculates and sets latency in number of buckets.
    void configureLatency(core_t::TTime latency, core_t::TTime bucketLength);

    //! Get the minimum permitted number of points in a sketched point.
    double minimumCategoryCount(void) const;

    //! Get the parameters supplied when restoring distribution models.
    maths::SDistributionRestoreParams distributionRestoreParams(maths_t::EDataType dataType) const;

    //! Get a checksum for an object of this class.
    uint64_t checksum(uint64_t seed) const;

    //! The bucketLength to use for the models
    core_t::TTime s_BucketLength;

    //! The delimiter used for separating components of a multivariate
    //! feature.
    std::string s_MultivariateComponentDelimiter;

    //! The rate at which the model learns per bucket.
    double s_LearnRate;

    //! The rate at which the model returns to non-informative per bucket.
    double s_DecayRate;

    //! The initial rate, as a multiple of s_DecayRate, at which the model
    //! returns to non-informative per bucket.
    double s_InitialDecayRateMultiplier;

    //! If true control the decay rate based on the model characteristics.
    bool s_ControlDecayRate;

    //! The minimum permitted fraction of points in a distribution mode.
    double s_MinimumModeFraction;

    //! The minimum permitted count of points in a distribution mode.
    double s_MinimumModeCount;

    //! The minimum frequency of non-empty buckets at which we model all buckets.
    double s_CutoffToModelEmptyBuckets;

    //! The number of points to use for approximating each seasonal component.
    std::size_t s_ComponentSize;

    //! Controls whether to exclude heavy hitters.
    model_t::EExcludeFrequent s_ExcludeFrequent;

    //! The frequency at which to exclude a person.
    double s_ExcludePersonFrequency;

    //! The frequency at which to exclude an attribute.
    double s_ExcludeAttributeFrequency;

    //! The maximum number of times we'll update a metric model in a bucket.
    double s_MaximumUpdatesPerBucket;

    //! The number of times we sample the people's attribute distributions
    //! to compute raw total probabilities for population models.
    std::size_t s_TotalProbabilityCalcSamplingSize;

    //! The minimum value for the influence for which an influencing field
    //! value is judged to have any influence on a feature value.
    double s_InfluenceCutoff;

    //! The number of buckets that are within the latency window.
    std::size_t s_LatencyBuckets;

    //! The factor to divide sample count in order to determine size of sub-samples.
    std::size_t s_SampleCountFactor;

    //! The factor that determines how much the sample queue grows.
    double s_SampleQueueGrowthFactor;

    //! The scale factor of the decayRate that determines the minimum size
    //! of the sliding prune window for purging older entries from the model
    double s_PruneWindowScaleMinimum;

    //! The scale factor of the decayRate that determines the maximum size
    //! of the sliding prune window for purging older entries from the model
    double s_PruneWindowScaleMaximum;

    //! The maximum overhead as a multiple of the base number of priors for
    //! modeling correlations.
    double s_CorrelationModelsOverhead;

    //! Should multivariate analysis of correlated 'by' fields be performed?
    bool s_MultivariateByFields;

    //! The minimum Pearson correlation coefficient at which a correlate will
    //! be modeled.
    double s_MinimumSignificantCorrelation;

    //! The detection rules for a detector.
    TDetectionRuleVecCRef s_DetectionRules;

    //! Scheduled events
    TStrDetectionRulePrVecCRef s_ScheduledEvents;

    //! The number of buckets to delay finalizing out-of-phase buckets.
    std::size_t s_BucketResultsDelay;

    //! The collection of multiple bucket lengths (if any)
    TTimeVec s_MultipleBucketLengths;

    //! The minimum data size to trigger fuzzy de-duplication of samples to add
    //! to population models.
    std::size_t s_MinimumToDeduplicate;

    //! If true then cache the results of the probability calculation.
    bool s_CacheProbabilities;

    //! The time window during which samples are accepted.
    core_t::TTime s_SamplingAgeCutoff;
};

}
}

#endif // INCLUDED_ml_model_CModelParams_h
