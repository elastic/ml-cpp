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

#include <model/FunctionTypes.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <boost/config.hpp>

#include <ostream>


namespace ml {
namespace model {
namespace function_t {

typedef model_t::TFeatureVec TFeatureVec;

bool isIndividual(EFunction function) {
    switch (function) {
        case E_IndividualCount:
        case E_IndividualNonZeroCount:
        case E_IndividualRareCount:
        case E_IndividualRareNonZeroCount:
        case E_IndividualRare:
        case E_IndividualLowCounts:
        case E_IndividualHighCounts:
        case E_IndividualLowNonZeroCount:
        case E_IndividualHighNonZeroCount:
        case E_IndividualDistinctCount:
        case E_IndividualLowDistinctCount:
        case E_IndividualHighDistinctCount:
        case E_IndividualInfoContent:
        case E_IndividualHighInfoContent:
        case E_IndividualLowInfoContent:
        case E_IndividualTimeOfDay:
        case E_IndividualTimeOfWeek:
        case E_IndividualMetric:
        case E_IndividualMetricMean:
        case E_IndividualMetricLowMean:
        case E_IndividualMetricHighMean:
        case E_IndividualMetricMedian:
        case E_IndividualMetricLowMedian:
        case E_IndividualMetricHighMedian:
        case E_IndividualMetricMin:
        case E_IndividualMetricMax:
        case E_IndividualMetricVariance:
        case E_IndividualMetricLowVariance:
        case E_IndividualMetricHighVariance:
        case E_IndividualMetricSum:
        case E_IndividualMetricLowSum:
        case E_IndividualMetricHighSum:
        case E_IndividualMetricNonNullSum:
        case E_IndividualMetricLowNonNullSum:
        case E_IndividualMetricHighNonNullSum:
        case E_IndividualLatLong:
        case E_IndividualMaxVelocity:
        case E_IndividualMinVelocity:
        case E_IndividualMeanVelocity:
        case E_IndividualSumVelocity:
            return true;

        case E_PopulationCount:
        case E_PopulationDistinctCount:
        case E_PopulationLowDistinctCount:
        case E_PopulationHighDistinctCount:
        case E_PopulationRare:
        case E_PopulationRareCount:
        case E_PopulationFreqRare:
        case E_PopulationFreqRareCount:
        case E_PopulationLowCounts:
        case E_PopulationHighCounts:
        case E_PopulationInfoContent:
        case E_PopulationLowInfoContent:
        case E_PopulationHighInfoContent:
        case E_PopulationTimeOfDay:
        case E_PopulationTimeOfWeek:
        case E_PopulationMetric:
        case E_PopulationMetricMean:
        case E_PopulationMetricLowMean:
        case E_PopulationMetricHighMean:
        case E_PopulationMetricMedian:
        case E_PopulationMetricLowMedian:
        case E_PopulationMetricHighMedian:
        case E_PopulationMetricMin:
        case E_PopulationMetricMax:
        case E_PopulationMetricVariance:
        case E_PopulationMetricLowVariance:
        case E_PopulationMetricHighVariance:
        case E_PopulationMetricSum:
        case E_PopulationMetricLowSum:
        case E_PopulationMetricHighSum:
        case E_PopulationLatLong:
        case E_PopulationMaxVelocity:
        case E_PopulationMinVelocity:
        case E_PopulationMeanVelocity:
        case E_PopulationSumVelocity:
            return false;

        case E_PeersCount:
        case E_PeersLowCounts:
        case E_PeersHighCounts:
        case E_PeersDistinctCount:
        case E_PeersLowDistinctCount:
        case E_PeersHighDistinctCount:
        case E_PeersInfoContent:
        case E_PeersLowInfoContent:
        case E_PeersHighInfoContent:
        case E_PeersTimeOfDay:
        case E_PeersTimeOfWeek:
            return false;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return false;
}

bool isPopulation(EFunction function) {
    switch (function) {
        case E_IndividualCount:
        case E_IndividualNonZeroCount:
        case E_IndividualRareCount:
        case E_IndividualRareNonZeroCount:
        case E_IndividualRare:
        case E_IndividualLowCounts:
        case E_IndividualHighCounts:
        case E_IndividualLowNonZeroCount:
        case E_IndividualHighNonZeroCount:
        case E_IndividualDistinctCount:
        case E_IndividualLowDistinctCount:
        case E_IndividualHighDistinctCount:
        case E_IndividualInfoContent:
        case E_IndividualHighInfoContent:
        case E_IndividualLowInfoContent:
        case E_IndividualTimeOfDay:
        case E_IndividualTimeOfWeek:
        case E_IndividualMetric:
        case E_IndividualMetricMean:
        case E_IndividualMetricLowMean:
        case E_IndividualMetricHighMean:
        case E_IndividualMetricMedian:
        case E_IndividualMetricLowMedian:
        case E_IndividualMetricHighMedian:
        case E_IndividualMetricMin:
        case E_IndividualMetricMax:
        case E_IndividualMetricVariance:
        case E_IndividualMetricLowVariance:
        case E_IndividualMetricHighVariance:
        case E_IndividualMetricSum:
        case E_IndividualMetricLowSum:
        case E_IndividualMetricHighSum:
        case E_IndividualMetricNonNullSum:
        case E_IndividualMetricLowNonNullSum:
        case E_IndividualMetricHighNonNullSum:
        case E_IndividualLatLong:
        case E_IndividualMaxVelocity:
        case E_IndividualMinVelocity:
        case E_IndividualMeanVelocity:
        case E_IndividualSumVelocity:
            return false;

        case E_PopulationCount:
        case E_PopulationDistinctCount:
        case E_PopulationLowDistinctCount:
        case E_PopulationHighDistinctCount:
        case E_PopulationRare:
        case E_PopulationRareCount:
        case E_PopulationFreqRare:
        case E_PopulationFreqRareCount:
        case E_PopulationLowCounts:
        case E_PopulationHighCounts:
        case E_PopulationInfoContent:
        case E_PopulationLowInfoContent:
        case E_PopulationHighInfoContent:
        case E_PopulationTimeOfDay:
        case E_PopulationTimeOfWeek:
        case E_PopulationMetric:
        case E_PopulationMetricMean:
        case E_PopulationMetricLowMean:
        case E_PopulationMetricHighMean:
        case E_PopulationMetricMedian:
        case E_PopulationMetricLowMedian:
        case E_PopulationMetricHighMedian:
        case E_PopulationMetricMin:
        case E_PopulationMetricMax:
        case E_PopulationMetricVariance:
        case E_PopulationMetricLowVariance:
        case E_PopulationMetricHighVariance:
        case E_PopulationMetricSum:
        case E_PopulationMetricLowSum:
        case E_PopulationMetricHighSum:
        case E_PopulationLatLong:
        case E_PopulationMaxVelocity:
        case E_PopulationMinVelocity:
        case E_PopulationMeanVelocity:
        case E_PopulationSumVelocity:
            return true;

        case E_PeersCount:
        case E_PeersLowCounts:
        case E_PeersHighCounts:
        case E_PeersDistinctCount:
        case E_PeersLowDistinctCount:
        case E_PeersHighDistinctCount:
        case E_PeersInfoContent:
        case E_PeersLowInfoContent:
        case E_PeersHighInfoContent:
        case E_PeersTimeOfDay:
        case E_PeersTimeOfWeek:
            return false;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return false;
}

bool isPeers(EFunction function) {
    switch (function) {
        case E_IndividualCount:
        case E_IndividualNonZeroCount:
        case E_IndividualRareCount:
        case E_IndividualRareNonZeroCount:
        case E_IndividualRare:
        case E_IndividualLowCounts:
        case E_IndividualHighCounts:
        case E_IndividualLowNonZeroCount:
        case E_IndividualHighNonZeroCount:
        case E_IndividualDistinctCount:
        case E_IndividualLowDistinctCount:
        case E_IndividualHighDistinctCount:
        case E_IndividualInfoContent:
        case E_IndividualHighInfoContent:
        case E_IndividualLowInfoContent:
        case E_IndividualTimeOfDay:
        case E_IndividualTimeOfWeek:
        case E_IndividualMetric:
        case E_IndividualMetricMean:
        case E_IndividualMetricLowMean:
        case E_IndividualMetricHighMean:
        case E_IndividualMetricMedian:
        case E_IndividualMetricLowMedian:
        case E_IndividualMetricHighMedian:
        case E_IndividualMetricMin:
        case E_IndividualMetricMax:
        case E_IndividualMetricVariance:
        case E_IndividualMetricLowVariance:
        case E_IndividualMetricHighVariance:
        case E_IndividualMetricSum:
        case E_IndividualMetricLowSum:
        case E_IndividualMetricHighSum:
        case E_IndividualMetricNonNullSum:
        case E_IndividualMetricLowNonNullSum:
        case E_IndividualMetricHighNonNullSum:
        case E_IndividualLatLong:
        case E_IndividualMaxVelocity:
        case E_IndividualMinVelocity:
        case E_IndividualMeanVelocity:
        case E_IndividualSumVelocity:
        case E_PopulationCount:
        case E_PopulationDistinctCount:
        case E_PopulationLowDistinctCount:
        case E_PopulationHighDistinctCount:
        case E_PopulationRare:
        case E_PopulationRareCount:
        case E_PopulationFreqRare:
        case E_PopulationFreqRareCount:
        case E_PopulationLowCounts:
        case E_PopulationHighCounts:
        case E_PopulationInfoContent:
        case E_PopulationLowInfoContent:
        case E_PopulationHighInfoContent:
        case E_PopulationTimeOfDay:
        case E_PopulationTimeOfWeek:
        case E_PopulationMetric:
        case E_PopulationMetricMean:
        case E_PopulationMetricLowMean:
        case E_PopulationMetricHighMean:
        case E_PopulationMetricMedian:
        case E_PopulationMetricLowMedian:
        case E_PopulationMetricHighMedian:
        case E_PopulationMetricMin:
        case E_PopulationMetricMax:
        case E_PopulationMetricVariance:
        case E_PopulationMetricLowVariance:
        case E_PopulationMetricHighVariance:
        case E_PopulationMetricSum:
        case E_PopulationMetricLowSum:
        case E_PopulationMetricHighSum:
        case E_PopulationLatLong:
        case E_PopulationMaxVelocity:
        case E_PopulationMinVelocity:
        case E_PopulationMeanVelocity:
        case E_PopulationSumVelocity:
            return false;

        case E_PeersCount:
        case E_PeersLowCounts:
        case E_PeersHighCounts:
        case E_PeersDistinctCount:
        case E_PeersLowDistinctCount:
        case E_PeersHighDistinctCount:
        case E_PeersInfoContent:
        case E_PeersLowInfoContent:
        case E_PeersHighInfoContent:
        case E_PeersTimeOfDay:
        case E_PeersTimeOfWeek:
            return true;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return false;
}

bool isMetric(EFunction function) {
    switch (function) {
        case E_IndividualCount:
        case E_IndividualNonZeroCount:
        case E_IndividualRareCount:
        case E_IndividualRareNonZeroCount:
        case E_IndividualRare:
        case E_IndividualLowCounts:
        case E_IndividualHighCounts:
        case E_IndividualLowNonZeroCount:
        case E_IndividualHighNonZeroCount:
        case E_IndividualDistinctCount:
        case E_IndividualLowDistinctCount:
        case E_IndividualHighDistinctCount:
        case E_IndividualInfoContent:
        case E_IndividualHighInfoContent:
        case E_IndividualLowInfoContent:
        case E_IndividualTimeOfDay:
        case E_IndividualTimeOfWeek:
            return false;

        case E_IndividualMetric:
        case E_IndividualMetricMean:
        case E_IndividualMetricLowMean:
        case E_IndividualMetricHighMean:
        case E_IndividualMetricMedian:
        case E_IndividualMetricLowMedian:
        case E_IndividualMetricHighMedian:
        case E_IndividualMetricMin:
        case E_IndividualMetricMax:
        case E_IndividualMetricVariance:
        case E_IndividualMetricLowVariance:
        case E_IndividualMetricHighVariance:
        case E_IndividualMetricSum:
        case E_IndividualMetricLowSum:
        case E_IndividualMetricHighSum:
        case E_IndividualMetricNonNullSum:
        case E_IndividualMetricLowNonNullSum:
        case E_IndividualMetricHighNonNullSum:
        case E_IndividualLatLong:
        case E_IndividualMaxVelocity:
        case E_IndividualMinVelocity:
        case E_IndividualMeanVelocity:
        case E_IndividualSumVelocity:
            return true;

        case E_PopulationCount:
        case E_PopulationDistinctCount:
        case E_PopulationLowDistinctCount:
        case E_PopulationHighDistinctCount:
        case E_PopulationRare:
        case E_PopulationRareCount:
        case E_PopulationFreqRare:
        case E_PopulationFreqRareCount:
        case E_PopulationLowCounts:
        case E_PopulationHighCounts:
        case E_PopulationInfoContent:
        case E_PopulationLowInfoContent:
        case E_PopulationHighInfoContent:
        case E_PopulationTimeOfDay:
        case E_PopulationTimeOfWeek:
            return false;

        case E_PopulationMetric:
        case E_PopulationMetricMean:
        case E_PopulationMetricLowMean:
        case E_PopulationMetricHighMean:
        case E_PopulationMetricMedian:
        case E_PopulationMetricLowMedian:
        case E_PopulationMetricHighMedian:
        case E_PopulationMetricMin:
        case E_PopulationMetricMax:
        case E_PopulationMetricVariance:
        case E_PopulationMetricLowVariance:
        case E_PopulationMetricHighVariance:
        case E_PopulationMetricSum:
        case E_PopulationMetricLowSum:
        case E_PopulationMetricHighSum:
        case E_PopulationLatLong:
        case E_PopulationMaxVelocity:
        case E_PopulationMinVelocity:
        case E_PopulationMeanVelocity:
        case E_PopulationSumVelocity:
            return true;

        case E_PeersCount:
        case E_PeersLowCounts:
        case E_PeersHighCounts:
        case E_PeersDistinctCount:
        case E_PeersLowDistinctCount:
        case E_PeersHighDistinctCount:
        case E_PeersInfoContent:
        case E_PeersLowInfoContent:
        case E_PeersHighInfoContent:
        case E_PeersTimeOfDay:
        case E_PeersTimeOfWeek:
            return false;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return false;
}

bool isForecastSupported(EFunction function) {
    switch (function) {
        case E_IndividualCount:
        case E_IndividualNonZeroCount:
        case E_IndividualRareCount:
        case E_IndividualRareNonZeroCount:
            return true;
        case E_IndividualRare:
            return false;
        case E_IndividualLowCounts:
        case E_IndividualHighCounts:
        case E_IndividualLowNonZeroCount:
        case E_IndividualHighNonZeroCount:
        case E_IndividualDistinctCount:
        case E_IndividualLowDistinctCount:
        case E_IndividualHighDistinctCount:
            return true;
        case E_IndividualInfoContent:
        case E_IndividualHighInfoContent:
        case E_IndividualLowInfoContent:
        case E_IndividualTimeOfDay:
        case E_IndividualTimeOfWeek:
            return false;

        case E_IndividualMetric:
        case E_IndividualMetricMean:
        case E_IndividualMetricLowMean:
        case E_IndividualMetricHighMean:
        case E_IndividualMetricMedian:
        case E_IndividualMetricLowMedian:
        case E_IndividualMetricHighMedian:
        case E_IndividualMetricMin:
        case E_IndividualMetricMax:
        case E_IndividualMetricVariance:
        case E_IndividualMetricLowVariance:
        case E_IndividualMetricHighVariance:
        case E_IndividualMetricSum:
        case E_IndividualMetricLowSum:
        case E_IndividualMetricHighSum:
        case E_IndividualMetricNonNullSum:
        case E_IndividualMetricLowNonNullSum:
        case E_IndividualMetricHighNonNullSum:
            return true;
        case E_IndividualLatLong:
            return false;
        case E_IndividualMaxVelocity:
        case E_IndividualMinVelocity:
        case E_IndividualMeanVelocity:
        case E_IndividualSumVelocity:
            return true;

        case E_PopulationCount:
        case E_PopulationDistinctCount:
        case E_PopulationLowDistinctCount:
        case E_PopulationHighDistinctCount:
        case E_PopulationRare:
        case E_PopulationRareCount:
        case E_PopulationFreqRare:
        case E_PopulationFreqRareCount:
        case E_PopulationLowCounts:
        case E_PopulationHighCounts:
        case E_PopulationInfoContent:
        case E_PopulationLowInfoContent:
        case E_PopulationHighInfoContent:
        case E_PopulationTimeOfDay:
        case E_PopulationTimeOfWeek:
            return false;

        case E_PopulationMetric:
        case E_PopulationMetricMean:
        case E_PopulationMetricLowMean:
        case E_PopulationMetricHighMean:
        case E_PopulationMetricMedian:
        case E_PopulationMetricLowMedian:
        case E_PopulationMetricHighMedian:
        case E_PopulationMetricMin:
        case E_PopulationMetricMax:
        case E_PopulationMetricVariance:
        case E_PopulationMetricLowVariance:
        case E_PopulationMetricHighVariance:
        case E_PopulationMetricSum:
        case E_PopulationMetricLowSum:
        case E_PopulationMetricHighSum:
        case E_PopulationLatLong:
        case E_PopulationMaxVelocity:
        case E_PopulationMinVelocity:
        case E_PopulationMeanVelocity:
        case E_PopulationSumVelocity:
            return false;

        case E_PeersCount:
        case E_PeersLowCounts:
        case E_PeersHighCounts:
        case E_PeersDistinctCount:
        case E_PeersLowDistinctCount:
        case E_PeersHighDistinctCount:
        case E_PeersInfoContent:
        case E_PeersLowInfoContent:
        case E_PeersHighInfoContent:
        case E_PeersTimeOfDay:
        case E_PeersTimeOfWeek:
            return false;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return false;
}


namespace {

typedef std::map<model_t::EFeature, TFunctionVec> TFeatureFunctionVecMap;
typedef TFeatureFunctionVecMap::iterator TFeatureFunctionVecMapItr;
typedef TFeatureFunctionVecMap::const_iterator TFeatureFunctionVecMapCItr;

namespace detail {

const model_t::EFeature INDIVIDUAL_COUNT_FEATURES[] =
{
    model_t::E_IndividualCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_NON_ZERO_COUNT_FEATURES[] =
{
    model_t::E_IndividualNonZeroCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_RARE_COUNT_FEATURES[] =
{
    model_t::E_IndividualCountByBucketAndPerson,
    model_t::E_IndividualTotalBucketCountByPerson,
};
const model_t::EFeature INDIVIDUAL_RARE_NON_ZERO_COUNT_FEATURES[] =
{
    model_t::E_IndividualNonZeroCountByBucketAndPerson,
    model_t::E_IndividualTotalBucketCountByPerson
};
const model_t::EFeature INDIVIDUAL_RARE_FEATURES[] =
{
    model_t::E_IndividualTotalBucketCountByPerson,
    model_t::E_IndividualIndicatorOfBucketPerson
};
const model_t::EFeature INDIVIDUAL_LOW_COUNTS_FEATURES[] =
{
    model_t::E_IndividualLowCountsByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_HIGH_COUNTS_FEATURES[] =
{
    model_t::E_IndividualHighCountsByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_LOW_NON_ZERO_COUNT_FEATURES[] =
{
    model_t::E_IndividualLowNonZeroCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_HIGH_NON_ZERO_COUNT_FEATURES[] =
{
    model_t::E_IndividualHighNonZeroCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_IndividualUniqueCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_LOW_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_IndividualLowUniqueCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_HIGH_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_IndividualHighUniqueCountByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_INFO_CONTENT_FEATURES[] =
{
    model_t::E_IndividualInfoContentByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_HIGH_INFO_CONTENT_FEATURES[] =
{
    model_t::E_IndividualHighInfoContentByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_LOW_INFO_CONTENT_FEATURES[] =
{
    model_t::E_IndividualLowInfoContentByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_TIME_OF_DAY_FEATURES[] =
{
    model_t::E_IndividualTimeOfDayByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_TIME_OF_WEEK_FEATURES[] =
{
    model_t::E_IndividualTimeOfWeekByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_FEATURES[] =
{
    model_t::E_IndividualMeanByPerson,
    model_t::E_IndividualMinByPerson,
    model_t::E_IndividualMaxByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_MEAN_FEATURES[] =
{
    model_t::E_IndividualMeanByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_LOW_MEAN_FEATURES[] =
{
    model_t::E_IndividualLowMeanByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_HIGH_MEAN_FEATURES[] =
{
    model_t::E_IndividualHighMeanByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_MEDIAN_FEATURES[] =
{
    model_t::E_IndividualMedianByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_LOW_MEDIAN_FEATURES[] =
{
    model_t::E_IndividualLowMedianByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_HIGH_MEDIAN_FEATURES[] =
{
    model_t::E_IndividualHighMedianByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_MIN_FEATURES[] =
{
    model_t::E_IndividualMinByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_MAX_FEATURES[] =
{
    model_t::E_IndividualMaxByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_VARIANCE_FEATURES[] =
{
    model_t::E_IndividualVarianceByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_LOW_VARIANCE_FEATURES[] =
{
    model_t::E_IndividualLowVarianceByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_HIGH_VARIANCE_FEATURES[] =
{
    model_t::E_IndividualHighVarianceByPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_SUM_FEATURES[] =
{
    model_t::E_IndividualSumByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_LOW_SUM_FEATURES[] =
{
    model_t::E_IndividualLowSumByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_HIGH_SUM_FEATURES[] =
{
    model_t::E_IndividualHighSumByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_NON_NULL_SUM_FEATURES[] =
{
    model_t::E_IndividualNonNullSumByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_LOW_NON_NULL_SUM_FEATURES[] =
{
    model_t::E_IndividualLowNonNullSumByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_METRIC_HIGH_NON_NULL_SUM_FEATURES[] =
{
    model_t::E_IndividualHighNonNullSumByBucketAndPerson
};
const model_t::EFeature INDIVIDUAL_LAT_LONG_FEATURES[] =
{
    model_t::E_IndividualMeanLatLongByPerson
};
const model_t::EFeature INDIVIDUAL_MAX_VELOCITY_FEATURES[] =
{
    model_t::E_IndividualMaxVelocityByPerson
};
const model_t::EFeature INDIVIDUAL_MIN_VELOCITY_FEATURES[] =
{
    model_t::E_IndividualMinVelocityByPerson
};
const model_t::EFeature INDIVIDUAL_MEAN_VELOCITY_FEATURES[] =
{
    model_t::E_IndividualMeanVelocityByPerson
};
const model_t::EFeature INDIVIDUAL_SUM_VELOCITY_FEATURES[] =
{
    model_t::E_IndividualSumVelocityByPerson
};
const model_t::EFeature POPULATION_COUNT_FEATURES[] =
{
    model_t::E_PopulationCountByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_PopulationUniqueCountByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_LOW_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_PopulationLowUniqueCountByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_HIGH_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_PopulationHighUniqueCountByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_RARE_FEATURES[] =
{
    model_t::E_PopulationIndicatorOfBucketPersonAndAttribute,
    model_t::E_PopulationUniquePersonCountByAttribute
};
const model_t::EFeature POPULATION_RARE_COUNT_FEATURES[] =
{
    model_t::E_PopulationCountByBucketPersonAndAttribute,
    model_t::E_PopulationUniquePersonCountByAttribute
};
const model_t::EFeature POPULATION_FREQ_RARE_FEATURES[] =
{
    model_t::E_PopulationAttributeTotalCountByPerson,
    model_t::E_PopulationIndicatorOfBucketPersonAndAttribute,
    model_t::E_PopulationUniquePersonCountByAttribute
};
const model_t::EFeature POPULATION_FREQ_RARE_COUNT_FEATURES[] =
{
    model_t::E_PopulationAttributeTotalCountByPerson,
    model_t::E_PopulationCountByBucketPersonAndAttribute,
    model_t::E_PopulationUniquePersonCountByAttribute
};
const model_t::EFeature POPULATION_LOW_COUNTS_FEATURES[] =
{
    model_t::E_PopulationLowCountsByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_HIGH_COUNTS_FEATURES[] =
{
    model_t::E_PopulationHighCountsByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_INFO_CONTENT_FEATURES[] =
{
    model_t::E_PopulationInfoContentByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_LOW_INFO_CONTENT_FEATURES[] =
{
    model_t::E_PopulationLowInfoContentByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_HIGH_INFO_CONTENT_FEATURES[] =
{
    model_t::E_PopulationHighInfoContentByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_TIME_OF_DAY_FEATURES[] =
{
    model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_TIME_OF_WEEK_FEATURES[] =
{
    model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_FEATURES[] =
{
    model_t::E_PopulationMeanByPersonAndAttribute,
    model_t::E_PopulationMinByPersonAndAttribute,
    model_t::E_PopulationMaxByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_MEAN_FEATURES[] =
{
    model_t::E_PopulationMeanByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_LOW_MEAN_FEATURES[] =
{
    model_t::E_PopulationLowMeanByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_HIGH_MEAN_FEATURES[] =
{
    model_t::E_PopulationHighMeanByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_MEDIAN_FEATURES[] =
{
    model_t::E_PopulationMedianByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_LOW_MEDIAN_FEATURES[] =
{
    model_t::E_PopulationLowMedianByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_HIGH_MEDIAN_FEATURES[] =
{
    model_t::E_PopulationHighMedianByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_MIN_FEATURES[] =
{
    model_t::E_PopulationMinByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_MAX_FEATURES[] =
{
    model_t::E_PopulationMaxByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_VARIANCE_FEATURES[] =
{
    model_t::E_PopulationVarianceByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_LOW_VARIANCE_FEATURES[] =
{
    model_t::E_PopulationLowVarianceByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_HIGH_VARIANCE_FEATURES[] =
{
    model_t::E_PopulationHighVarianceByPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_SUM_FEATURES[] =
{
    model_t::E_PopulationSumByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_LOW_SUM_FEATURES[] =
{
    model_t::E_PopulationLowSumByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_METRIC_HIGH_SUM_FEATURES[] =
{
    model_t::E_PopulationHighSumByBucketPersonAndAttribute
};
const model_t::EFeature POPULATION_LAT_LONG_FEATURES[] =
{
    model_t::E_PopulationMeanLatLongByPersonAndAttribute
};
const model_t::EFeature POPULATION_MAX_VELOCITY_FEATURES[] =
{
    model_t::E_PopulationMaxVelocityByPersonAndAttribute
};
const model_t::EFeature POPULATION_MIN_VELOCITY_FEATURES[] =
{
    model_t::E_PopulationMinVelocityByPersonAndAttribute
};
const model_t::EFeature POPULATION_MEAN_VELOCITY_FEATURES[] =
{
    model_t::E_PopulationMeanVelocityByPersonAndAttribute
};
const model_t::EFeature POPULATION_SUM_VELOCITY_FEATURES[] =
{
    model_t::E_PopulationSumVelocityByPersonAndAttribute
};
const model_t::EFeature PEERS_COUNT_FEATURES[] =
{
    model_t::E_PeersCountByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_PeersUniqueCountByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_LOW_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_PeersLowUniqueCountByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_HIGH_DISTINCT_COUNT_FEATURES[] =
{
    model_t::E_PeersHighUniqueCountByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_LOW_COUNTS_FEATURES[] =
{
    model_t::E_PeersLowCountsByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_HIGH_COUNTS_FEATURES[] =
{
    model_t::E_PeersHighCountsByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_INFO_CONTENT_FEATURES[] =
{
    model_t::E_PeersInfoContentByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_LOW_INFO_CONTENT_FEATURES[] =
{
    model_t::E_PeersLowInfoContentByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_HIGH_INFO_CONTENT_FEATURES[] =
{
    model_t::E_PeersHighInfoContentByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_TIME_OF_DAY_FEATURES[] =
{
    model_t::E_PeersTimeOfDayByBucketPersonAndAttribute
};
const model_t::EFeature PEERS_TIME_OF_WEEK_FEATURES[] =
{
    model_t::E_PeersTimeOfWeekByBucketPersonAndAttribute
};

// Function names
const std::string COUNT("count");
const std::string NON_ZERO_COUNT("non_zero_count");
const std::string RARE_NON_ZERO_COUNT("rare_non_zero_count");
const std::string RARE("rare");
const std::string LOW_COUNT("low_count");
const std::string HIGH_COUNT("high_count");
const std::string LOW_NON_ZERO_COUNT("low_non_zero_count");
const std::string HIGH_NON_ZERO_COUNT("high_non_zero_count");
const std::string DISTINCT_COUNT("distinct_count");
const std::string LOW_DISTINCT_COUNT("low_distinct_count");
const std::string HIGH_DISTINCT_COUNT("high_distinct_count");
const std::string INFO_CONTENT("info_content");
const std::string HIGH_INFO_CONTENT("high_info_content");
const std::string LOW_INFO_CONTENT("low_info_content");
const std::string TIME_OF_DAY("time_of_day");
const std::string TIME_OF_WEEK("time_of_week");
const std::string METRIC("metric");
const std::string MEAN("mean");
const std::string LOW_MEAN("low_mean");
const std::string HIGH_MEAN("high_mean");
const std::string MEDIAN("median");
const std::string LOW_MEDIAN("low_median");
const std::string HIGH_MEDIAN("high_median");
const std::string MIN("min");
const std::string MAX("max");
const std::string VARIANCE("varp");
const std::string LOW_VARIANCE("low_varp");
const std::string HIGH_VARIANCE("high_varp");
const std::string SUM("sum");
const std::string LOW_SUM("low_sum");
const std::string HIGH_SUM("high_sum");
const std::string NON_NULL_SUM("non_null_sum");
const std::string LOW_NON_NULL_SUM("low_non_null_sum");
const std::string HIGH_NON_NULL_SUM("high_non_null_sum");
const std::string LAT_LONG("lat_long");
const std::string RARE_COUNT("rare_count");
const std::string FREQ_RARE("freq_rare");
const std::string FREQ_RARE_COUNT("freq_rare_count");
const std::string MAX_VELOCITY("max_velocity");
const std::string MIN_VELOCITY("min_velocity");
const std::string MEAN_VELOCITY("mean_velocity");
const std::string SUM_VELOCITY("sum_velocity");
const std::string UNEXPECTED_FUNCTION("-");

}

#define BEGIN(x) x
#define END(x) x + sizeof(x)/sizeof(x[0])

//! The features for the count by function.
const TFeatureVec INDIVIDUAL_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_COUNT_FEATURES),
                                            END(detail::INDIVIDUAL_COUNT_FEATURES));

//! The features for the non-zero count by function.
const TFeatureVec INDIVIDUAL_NON_ZERO_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_NON_ZERO_COUNT_FEATURES),
                                                     END(detail::INDIVIDUAL_NON_ZERO_COUNT_FEATURES));

//! The features for the rare count by function.
const TFeatureVec INDIVIDUAL_RARE_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_RARE_COUNT_FEATURES),
                                                 END(detail::INDIVIDUAL_RARE_COUNT_FEATURES));

//! The features for the rare non-zero count by function.
const TFeatureVec INDIVIDUAL_RARE_NON_ZERO_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_RARE_NON_ZERO_COUNT_FEATURES),
                                                          END(detail::INDIVIDUAL_RARE_NON_ZERO_COUNT_FEATURES));

//! The features for the rare in time by function.
const TFeatureVec INDIVIDUAL_RARE_FEATURES(BEGIN(detail::INDIVIDUAL_RARE_FEATURES),
                                           END(detail::INDIVIDUAL_RARE_FEATURES));

//! The features for the low count by function.
const TFeatureVec INDIVIDUAL_LOW_COUNTS_FEATURES(BEGIN(detail::INDIVIDUAL_LOW_COUNTS_FEATURES),
                                                 END(detail::INDIVIDUAL_LOW_COUNTS_FEATURES));

//! The features for the high count by function.
const TFeatureVec INDIVIDUAL_HIGH_COUNTS_FEATURES(BEGIN(detail::INDIVIDUAL_HIGH_COUNTS_FEATURES),
                                                  END(detail::INDIVIDUAL_HIGH_COUNTS_FEATURES));

//! The features for the low non zero count by function.
const TFeatureVec INDIVIDUAL_LOW_NON_ZERO_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_LOW_NON_ZERO_COUNT_FEATURES),
                                                         END(detail::INDIVIDUAL_LOW_NON_ZERO_COUNT_FEATURES));

//! The features for the high non zero count by function.
const TFeatureVec INDIVIDUAL_HIGH_NON_ZERO_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_HIGH_NON_ZERO_COUNT_FEATURES),
                                                          END(detail::INDIVIDUAL_HIGH_NON_ZERO_COUNT_FEATURES));

//! The features for the distinct count function.
const TFeatureVec INDIVIDUAL_DISTINCT_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_DISTINCT_COUNT_FEATURES),
                                                     END(detail::INDIVIDUAL_DISTINCT_COUNT_FEATURES));

//! The features for the distinct count function.
const TFeatureVec INDIVIDUAL_LOW_DISTINCT_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_LOW_DISTINCT_COUNT_FEATURES),
                                                         END(detail::INDIVIDUAL_LOW_DISTINCT_COUNT_FEATURES));

//! The features for the distinct count function.
const TFeatureVec INDIVIDUAL_HIGH_DISTINCT_COUNT_FEATURES(BEGIN(detail::INDIVIDUAL_HIGH_DISTINCT_COUNT_FEATURES),
                                                          END(detail::INDIVIDUAL_HIGH_DISTINCT_COUNT_FEATURES));

//! The features for the individual info_content function
const TFeatureVec INDIVIDUAL_INFO_CONTENT_FEATURES(BEGIN(detail::INDIVIDUAL_INFO_CONTENT_FEATURES),
                                                   END(detail::INDIVIDUAL_INFO_CONTENT_FEATURES));

//! The features for the individual high_info_content function
const TFeatureVec INDIVIDUAL_HIGH_INFO_CONTENT_FEATURES(BEGIN(detail::INDIVIDUAL_HIGH_INFO_CONTENT_FEATURES),
                                                        END(detail::INDIVIDUAL_HIGH_INFO_CONTENT_FEATURES));

//! The features for the individual low_info_content function
const TFeatureVec INDIVIDUAL_LOW_INFO_CONTENT_FEATURES(BEGIN(detail::INDIVIDUAL_LOW_INFO_CONTENT_FEATURES),
                                                       END(detail::INDIVIDUAL_LOW_INFO_CONTENT_FEATURES));

//! The features for the time-of-day function.
const TFeatureVec INDIVIDUAL_TIME_OF_DAY_FEATURES(BEGIN(detail::INDIVIDUAL_TIME_OF_DAY_FEATURES),
                                                  END(detail::INDIVIDUAL_TIME_OF_DAY_FEATURES));

//! The features for the time-of-week function.
const TFeatureVec INDIVIDUAL_TIME_OF_WEEK_FEATURES(BEGIN(detail::INDIVIDUAL_TIME_OF_WEEK_FEATURES),
                                                   END(detail::INDIVIDUAL_TIME_OF_WEEK_FEATURES));

//! The features for the metric by function.
const TFeatureVec INDIVIDUAL_METRIC_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_FEATURES),
                                             END(detail::INDIVIDUAL_METRIC_FEATURES));

//! The features for the metric mean by function.
const TFeatureVec INDIVIDUAL_METRIC_MEAN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_MEAN_FEATURES),
                                                  END(detail::INDIVIDUAL_METRIC_MEAN_FEATURES));

//! The features for the metric low mean by function.
const TFeatureVec INDIVIDUAL_METRIC_LOW_MEAN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_LOW_MEAN_FEATURES),
                                                      END(detail::INDIVIDUAL_METRIC_LOW_MEAN_FEATURES));

//! The features for the metric high mean by function.
const TFeatureVec INDIVIDUAL_METRIC_HIGH_MEAN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_HIGH_MEAN_FEATURES),
                                                       END(detail::INDIVIDUAL_METRIC_HIGH_MEAN_FEATURES));

//! The features for the metric median by function.
const TFeatureVec INDIVIDUAL_METRIC_MEDIAN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_MEDIAN_FEATURES),
                                                    END(detail::INDIVIDUAL_METRIC_MEDIAN_FEATURES));

//! The features for the metric low median by function.
const TFeatureVec INDIVIDUAL_METRIC_LOW_MEDIAN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_LOW_MEDIAN_FEATURES),
                                                        END(detail::INDIVIDUAL_METRIC_LOW_MEDIAN_FEATURES));

//! The features for the metric high median by function.
const TFeatureVec INDIVIDUAL_METRIC_HIGH_MEDIAN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_HIGH_MEDIAN_FEATURES),
                                                         END(detail::INDIVIDUAL_METRIC_HIGH_MEDIAN_FEATURES));

//! The features for the metric min by function.
const TFeatureVec INDIVIDUAL_METRIC_MIN_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_MIN_FEATURES),
                                                 END(detail::INDIVIDUAL_METRIC_MIN_FEATURES));

//! The features for the metric max by function.
const TFeatureVec INDIVIDUAL_METRIC_MAX_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_MAX_FEATURES),
                                                 END(detail::INDIVIDUAL_METRIC_MAX_FEATURES));

//! The features for the metric variance by function.
const TFeatureVec INDIVIDUAL_METRIC_VARIANCE_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_VARIANCE_FEATURES),
                                                      END(detail::INDIVIDUAL_METRIC_VARIANCE_FEATURES));

//! The features for the metric low variance by function.
const TFeatureVec INDIVIDUAL_METRIC_LOW_VARIANCE_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_LOW_VARIANCE_FEATURES),
                                                          END(detail::INDIVIDUAL_METRIC_LOW_VARIANCE_FEATURES));

//! The features for the metric high variance by function.
const TFeatureVec INDIVIDUAL_METRIC_HIGH_VARIANCE_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_HIGH_VARIANCE_FEATURES),
                                                           END(detail::INDIVIDUAL_METRIC_HIGH_VARIANCE_FEATURES));

//! The features for the metric sum by function.
const TFeatureVec INDIVIDUAL_METRIC_SUM_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_SUM_FEATURES),
                                                 END(detail::INDIVIDUAL_METRIC_SUM_FEATURES));

//! The features for the metric low sum by function.
const TFeatureVec INDIVIDUAL_METRIC_LOW_SUM_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_LOW_SUM_FEATURES),
                                                     END(detail::INDIVIDUAL_METRIC_LOW_SUM_FEATURES));

//! The features for the metric high sum by function.
const TFeatureVec INDIVIDUAL_METRIC_HIGH_SUM_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_HIGH_SUM_FEATURES),
                                                      END(detail::INDIVIDUAL_METRIC_HIGH_SUM_FEATURES));

//! The features for the metric non-null sum by function.
const TFeatureVec INDIVIDUAL_METRIC_NON_NULL_SUM_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_NON_NULL_SUM_FEATURES),
                                                          END(detail::INDIVIDUAL_METRIC_NON_NULL_SUM_FEATURES));

//! The features for the metric low non-null sum by function.
const TFeatureVec INDIVIDUAL_METRIC_LOW_NON_NULL_SUM_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_LOW_NON_NULL_SUM_FEATURES),
                                                              END(detail::INDIVIDUAL_METRIC_LOW_NON_NULL_SUM_FEATURES));

//! The features for the metric high non-null sum by function.
const TFeatureVec INDIVIDUAL_METRIC_HIGH_NON_NULL_SUM_FEATURES(BEGIN(detail::INDIVIDUAL_METRIC_HIGH_NON_NULL_SUM_FEATURES),
                                                               END(detail::INDIVIDUAL_METRIC_HIGH_NON_NULL_SUM_FEATURES));

//! The features for the metric latitude and longitude by function.
const TFeatureVec INDIVIDUAL_LAT_LONG_FEATURES(BEGIN(detail::INDIVIDUAL_LAT_LONG_FEATURES),
                                               END(detail::INDIVIDUAL_LAT_LONG_FEATURES));

//! The features for the metric max velocity by function.
const TFeatureVec INDIVIDUAL_MAX_VELOCITY_FEATURES(BEGIN(detail::INDIVIDUAL_MAX_VELOCITY_FEATURES),
                                                   END(detail::INDIVIDUAL_MAX_VELOCITY_FEATURES));

//! The features for the metric min velocity by function.
const TFeatureVec INDIVIDUAL_MIN_VELOCITY_FEATURES(BEGIN(detail::INDIVIDUAL_MIN_VELOCITY_FEATURES),
                                                   END(detail::INDIVIDUAL_MIN_VELOCITY_FEATURES));

//! The features for the metric mean velocity by function.
const TFeatureVec INDIVIDUAL_MEAN_VELOCITY_FEATURES(BEGIN(detail::INDIVIDUAL_MEAN_VELOCITY_FEATURES),
                                                    END(detail::INDIVIDUAL_MEAN_VELOCITY_FEATURES));

//! The features for the metric sum velocity by function.
const TFeatureVec INDIVIDUAL_SUM_VELOCITY_FEATURES(BEGIN(detail::INDIVIDUAL_SUM_VELOCITY_FEATURES),
                                                   END(detail::INDIVIDUAL_SUM_VELOCITY_FEATURES));

//! The features for the count over function.
const TFeatureVec POPULATION_COUNT_FEATURES(BEGIN(detail::POPULATION_COUNT_FEATURES),
                                            END(detail::POPULATION_COUNT_FEATURES));

//! The features for the distinct count over function.
const TFeatureVec POPULATION_DISTINCT_COUNT_FEATURES(BEGIN(detail::POPULATION_DISTINCT_COUNT_FEATURES),
                                                     END(detail::POPULATION_DISTINCT_COUNT_FEATURES));

//! The features for the low distinct count over function.
const TFeatureVec POPULATION_LOW_DISTINCT_COUNT_FEATURES(BEGIN(detail::POPULATION_LOW_DISTINCT_COUNT_FEATURES),
                                                         END(detail::POPULATION_LOW_DISTINCT_COUNT_FEATURES));

//! The features for the high distinct count over function.
const TFeatureVec POPULATION_HIGH_DISTINCT_COUNT_FEATURES(BEGIN(detail::POPULATION_HIGH_DISTINCT_COUNT_FEATURES),
                                                          END(detail::POPULATION_HIGH_DISTINCT_COUNT_FEATURES));

//! The features for the rare over function.
const TFeatureVec POPULATION_RARE_FEATURES(BEGIN(detail::POPULATION_RARE_FEATURES),
                                           END(detail::POPULATION_RARE_FEATURES));

//! The features for the rare count over function.
const TFeatureVec POPULATION_RARE_COUNT_FEATURES(BEGIN(detail::POPULATION_RARE_COUNT_FEATURES),
                                                 END(detail::POPULATION_RARE_COUNT_FEATURES));

//! The features for the rare in population over function.
const TFeatureVec POPULATION_FREQ_RARE_FEATURES(BEGIN(detail::POPULATION_FREQ_RARE_FEATURES),
                                                END(detail::POPULATION_FREQ_RARE_FEATURES));

//! The features for the frequent rare count over function.
const TFeatureVec POPULATION_FREQ_RARE_COUNT_FEATURES(BEGIN(detail::POPULATION_FREQ_RARE_COUNT_FEATURES),
                                                      END(detail::POPULATION_FREQ_RARE_COUNT_FEATURES));

//! The features for the low count over function.
const TFeatureVec POPULATION_LOW_COUNTS_FEATURES(BEGIN(detail::POPULATION_LOW_COUNTS_FEATURES),
                                                 END(detail::POPULATION_LOW_COUNTS_FEATURES));

//! The features for the high count over function.
const TFeatureVec POPULATION_HIGH_COUNTS_FEATURES(BEGIN(detail::POPULATION_HIGH_COUNTS_FEATURES),
                                                  END(detail::POPULATION_HIGH_COUNTS_FEATURES));

//! The features for the information content over function.
const TFeatureVec POPULATION_INFO_CONTENT_FEATURES(BEGIN(detail::POPULATION_INFO_CONTENT_FEATURES),
                                                   END(detail::POPULATION_INFO_CONTENT_FEATURES));

//! The features for the low information content over function.
const TFeatureVec POPULATION_LOW_INFO_CONTENT_FEATURES(BEGIN(detail::POPULATION_LOW_INFO_CONTENT_FEATURES),
                                                       END(detail::POPULATION_LOW_INFO_CONTENT_FEATURES));

//! The features for the high information content over function.
const TFeatureVec POPULATION_HIGH_INFO_CONTENT_FEATURES(BEGIN(detail::POPULATION_HIGH_INFO_CONTENT_FEATURES),
                                                        END(detail::POPULATION_HIGH_INFO_CONTENT_FEATURES));

//! The features for the time_of_week over function.
const TFeatureVec POPULATION_TIME_OF_DAY_FEATURES(BEGIN(detail::POPULATION_TIME_OF_DAY_FEATURES),
                                                  END(detail::POPULATION_TIME_OF_DAY_FEATURES));

//! The features for the time_of_week over function.
const TFeatureVec POPULATION_TIME_OF_WEEK_FEATURES(BEGIN(detail::POPULATION_TIME_OF_WEEK_FEATURES),
                                                   END(detail::POPULATION_TIME_OF_WEEK_FEATURES));

//! The features for the metric over function.
const TFeatureVec POPULATION_METRIC_FEATURES(BEGIN(detail::POPULATION_METRIC_FEATURES),
                                             END(detail::POPULATION_METRIC_FEATURES));

//! The features for the metric mean over function.
const TFeatureVec POPULATION_METRIC_MEAN_FEATURES(BEGIN(detail::POPULATION_METRIC_MEAN_FEATURES),
                                                  END(detail::POPULATION_METRIC_MEAN_FEATURES));

//! The features for the metric low mean over function.
const TFeatureVec POPULATION_METRIC_LOW_MEAN_FEATURES(BEGIN(detail::POPULATION_METRIC_LOW_MEAN_FEATURES),
                                                      END(detail::POPULATION_METRIC_LOW_MEAN_FEATURES));

//! The features for the metric high mean over function.
const TFeatureVec POPULATION_METRIC_HIGH_MEAN_FEATURES(BEGIN(detail::POPULATION_METRIC_HIGH_MEAN_FEATURES),
                                                       END(detail::POPULATION_METRIC_HIGH_MEAN_FEATURES));

//! The features for the metric median over function.
const TFeatureVec POPULATION_METRIC_MEDIAN_FEATURES(BEGIN(detail::POPULATION_METRIC_MEDIAN_FEATURES),
                                                    END(detail::POPULATION_METRIC_MEDIAN_FEATURES));

//! The features for the metric low median over function.
const TFeatureVec POPULATION_METRIC_LOW_MEDIAN_FEATURES(BEGIN(detail::POPULATION_METRIC_LOW_MEDIAN_FEATURES),
                                                        END(detail::POPULATION_METRIC_LOW_MEDIAN_FEATURES));

//! The features for the metric high median over function.
const TFeatureVec POPULATION_METRIC_HIGH_MEDIAN_FEATURES(BEGIN(detail::POPULATION_METRIC_HIGH_MEDIAN_FEATURES),
                                                         END(detail::POPULATION_METRIC_HIGH_MEDIAN_FEATURES));

//! The features for the metric min over function.
const TFeatureVec POPULATION_METRIC_MIN_FEATURES(BEGIN(detail::POPULATION_METRIC_MIN_FEATURES),
                                                 END(detail::POPULATION_METRIC_MIN_FEATURES));

//! The features for the metric max over function.
const TFeatureVec POPULATION_METRIC_MAX_FEATURES(BEGIN(detail::POPULATION_METRIC_MAX_FEATURES),
                                                 END(detail::POPULATION_METRIC_MAX_FEATURES));

//! The features for the metric variance over function.
const TFeatureVec POPULATION_METRIC_VARIANCE_FEATURES(BEGIN(detail::POPULATION_METRIC_VARIANCE_FEATURES),
                                                      END(detail::POPULATION_METRIC_VARIANCE_FEATURES));

//! The features for the metric low variance over function.
const TFeatureVec POPULATION_METRIC_LOW_VARIANCE_FEATURES(BEGIN(detail::POPULATION_METRIC_LOW_VARIANCE_FEATURES),
                                                          END(detail::POPULATION_METRIC_LOW_VARIANCE_FEATURES));

//! The features for the metric high variance over function.
const TFeatureVec POPULATION_METRIC_HIGH_VARIANCE_FEATURES(BEGIN(detail::POPULATION_METRIC_HIGH_VARIANCE_FEATURES),
                                                           END(detail::POPULATION_METRIC_HIGH_VARIANCE_FEATURES));

//! The features for the metric sum over function.
const TFeatureVec POPULATION_METRIC_SUM_FEATURES(BEGIN(detail::POPULATION_METRIC_SUM_FEATURES),
                                                 END(detail::POPULATION_METRIC_SUM_FEATURES));
//! The features for the metric low sum over function.
const TFeatureVec POPULATION_METRIC_LOW_SUM_FEATURES(BEGIN(detail::POPULATION_METRIC_LOW_SUM_FEATURES),
                                                     END(detail::POPULATION_METRIC_LOW_SUM_FEATURES));

//! The features for the metric high sum over function.
const TFeatureVec POPULATION_METRIC_HIGH_SUM_FEATURES(BEGIN(detail::POPULATION_METRIC_HIGH_SUM_FEATURES),
                                                      END(detail::POPULATION_METRIC_HIGH_SUM_FEATURES));

//! The features for the metric lat/long over function.
const TFeatureVec POPULATION_LAT_LONG_FEATURES(BEGIN(detail::POPULATION_LAT_LONG_FEATURES),
                                               END(detail::POPULATION_LAT_LONG_FEATURES));

//! The features for the metric max velocity over function.
const TFeatureVec POPULATION_MAX_VELOCITY_FEATURES(BEGIN(detail::POPULATION_MAX_VELOCITY_FEATURES),
                                                   END(detail::POPULATION_MAX_VELOCITY_FEATURES));

//! The features for the metric min velocity over function.
const TFeatureVec POPULATION_MIN_VELOCITY_FEATURES(BEGIN(detail::POPULATION_MIN_VELOCITY_FEATURES),
                                                   END(detail::POPULATION_MIN_VELOCITY_FEATURES));

//! The features for the metric mean velocity over function.
const TFeatureVec POPULATION_MEAN_VELOCITY_FEATURES(BEGIN(detail::POPULATION_MEAN_VELOCITY_FEATURES),
                                                    END(detail::POPULATION_MEAN_VELOCITY_FEATURES));

//! The features for the metric sum velocity over function.
const TFeatureVec POPULATION_SUM_VELOCITY_FEATURES(BEGIN(detail::POPULATION_SUM_VELOCITY_FEATURES),
                                                   END(detail::POPULATION_SUM_VELOCITY_FEATURES));

//! The features for the count over function.
const TFeatureVec PEERS_COUNT_FEATURES(BEGIN(detail::PEERS_COUNT_FEATURES),
                                       END(detail::PEERS_COUNT_FEATURES));

//! The features for the low count over function.
const TFeatureVec PEERS_LOW_COUNTS_FEATURES(BEGIN(detail::PEERS_LOW_COUNTS_FEATURES),
                                            END(detail::PEERS_LOW_COUNTS_FEATURES));

//! The features for the high count over function.
const TFeatureVec PEERS_HIGH_COUNTS_FEATURES(BEGIN(detail::PEERS_HIGH_COUNTS_FEATURES),
                                             END(detail::PEERS_HIGH_COUNTS_FEATURES));

//! The features for the distinct count over function.
const TFeatureVec PEERS_DISTINCT_COUNT_FEATURES(BEGIN(detail::PEERS_DISTINCT_COUNT_FEATURES),
                                                END(detail::PEERS_DISTINCT_COUNT_FEATURES));

//! The features for the low distinct count over function.
const TFeatureVec PEERS_LOW_DISTINCT_COUNT_FEATURES(BEGIN(detail::PEERS_LOW_DISTINCT_COUNT_FEATURES),
                                                    END(detail::PEERS_LOW_DISTINCT_COUNT_FEATURES));

//! The features for the high distinct count over function.
const TFeatureVec PEERS_HIGH_DISTINCT_COUNT_FEATURES(BEGIN(detail::PEERS_HIGH_DISTINCT_COUNT_FEATURES),
                                                     END(detail::PEERS_HIGH_DISTINCT_COUNT_FEATURES));

//! The features for the information content over function.
const TFeatureVec PEERS_INFO_CONTENT_FEATURES(BEGIN(detail::PEERS_INFO_CONTENT_FEATURES),
                                              END(detail::PEERS_INFO_CONTENT_FEATURES));

//! The features for the low information content over function.
const TFeatureVec PEERS_LOW_INFO_CONTENT_FEATURES(BEGIN(detail::PEERS_LOW_INFO_CONTENT_FEATURES),
                                                  END(detail::PEERS_LOW_INFO_CONTENT_FEATURES));

//! The features for the high information content over function.
const TFeatureVec PEERS_HIGH_INFO_CONTENT_FEATURES(BEGIN(detail::PEERS_HIGH_INFO_CONTENT_FEATURES),
                                                   END(detail::PEERS_HIGH_INFO_CONTENT_FEATURES));

//! The features for the time_of_week over function.
const TFeatureVec PEERS_TIME_OF_DAY_FEATURES(BEGIN(detail::PEERS_TIME_OF_DAY_FEATURES),
                                             END(detail::PEERS_TIME_OF_DAY_FEATURES));

//! The features for the time_of_week over function.
const TFeatureVec PEERS_TIME_OF_WEEK_FEATURES(BEGIN(detail::PEERS_TIME_OF_WEEK_FEATURES),
                                              END(detail::PEERS_TIME_OF_WEEK_FEATURES));

const TFeatureVec EMPTY_FEATURES;
const TFunctionVec EMPTY_FUNCTIONS;

#undef BEGIN
#undef END

//! Add the features corresponding to \p function to \p map.
void addFeatures(EFunction function,
                 TFeatureFunctionVecMap &map) {
    const TFeatureVec &features = function_t::features(function);
    for (std::size_t i = 0u; i < features.size(); ++i) {
        map[features[i]].push_back(function);
    }
}

//! Build a map from features to the functions which include them.
TFeatureFunctionVecMap buildFeatureFunctionMap(void) {
    TFeatureFunctionVecMap result;

    // This is written like this to generate a compiler warning
    // when a new function is added. This map must include every
    // function so add a case if you add a new function.

    switch (E_IndividualCount) {
        // The fall-through is intentional in this switch: the switched-on value
        // selects the first case and then all the calls to addFeatures() are
        // made
        case E_IndividualCount:
            addFeatures(E_IndividualCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualNonZeroCount:
            addFeatures(E_IndividualNonZeroCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualRareCount:
            addFeatures(E_IndividualRareCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualRareNonZeroCount:
            addFeatures(E_IndividualRareNonZeroCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualRare:
            addFeatures(E_IndividualRare, result);
            BOOST_FALLTHROUGH;
        case E_IndividualLowCounts:
            addFeatures(E_IndividualLowCounts, result);
            BOOST_FALLTHROUGH;
        case E_IndividualHighCounts:
            addFeatures(E_IndividualHighCounts, result);
            BOOST_FALLTHROUGH;
        case E_IndividualLowNonZeroCount:
            addFeatures(E_IndividualLowNonZeroCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualHighNonZeroCount:
            addFeatures(E_IndividualHighNonZeroCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualDistinctCount:
            addFeatures(E_IndividualDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualLowDistinctCount:
            addFeatures(E_IndividualLowDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualHighDistinctCount:
            addFeatures(E_IndividualHighDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_IndividualInfoContent:
            addFeatures(E_IndividualInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_IndividualHighInfoContent:
            addFeatures(E_IndividualHighInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_IndividualLowInfoContent:
            addFeatures(E_IndividualLowInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_IndividualTimeOfDay:
            addFeatures(E_IndividualTimeOfDay, result);
            BOOST_FALLTHROUGH;
        case E_IndividualTimeOfWeek:
            addFeatures(E_IndividualTimeOfWeek, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetric:
            addFeatures(E_IndividualMetric, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricMean:
            addFeatures(E_IndividualMetricMean, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricLowMean:
            addFeatures(E_IndividualMetricLowMean, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricHighMean:
            addFeatures(E_IndividualMetricHighMean, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricMedian:
            addFeatures(E_IndividualMetricMedian, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricLowMedian:
            addFeatures(E_IndividualMetricLowMedian, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricHighMedian:
            addFeatures(E_IndividualMetricHighMedian, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricMin:
            addFeatures(E_IndividualMetricMin, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricMax:
            addFeatures(E_IndividualMetricMax, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricSum:
            addFeatures(E_IndividualMetricSum, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricVariance:
            addFeatures(E_IndividualMetricVariance, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricLowVariance:
            addFeatures(E_IndividualMetricLowVariance, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricHighVariance:
            addFeatures(E_IndividualMetricHighVariance, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricLowSum:
            addFeatures(E_IndividualMetricLowSum, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricHighSum:
            addFeatures(E_IndividualMetricHighSum, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricNonNullSum:
            addFeatures(E_IndividualMetricNonNullSum, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricLowNonNullSum:
            addFeatures(E_IndividualMetricLowNonNullSum, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMetricHighNonNullSum:
            addFeatures(E_IndividualMetricHighNonNullSum, result);
            BOOST_FALLTHROUGH;
        case E_IndividualLatLong:
            addFeatures(E_IndividualLatLong, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMaxVelocity:
            addFeatures(E_IndividualMaxVelocity, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMinVelocity:
            addFeatures(E_IndividualMinVelocity, result);
            BOOST_FALLTHROUGH;
        case E_IndividualMeanVelocity:
            addFeatures(E_IndividualMeanVelocity, result);
            BOOST_FALLTHROUGH;
        case E_IndividualSumVelocity:
            addFeatures(E_IndividualSumVelocity, result);
            BOOST_FALLTHROUGH;
        case E_PopulationCount:
            addFeatures(E_PopulationCount, result);
            BOOST_FALLTHROUGH;
        case E_PopulationDistinctCount:
            addFeatures(E_PopulationDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_PopulationLowDistinctCount:
            addFeatures(E_PopulationLowDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_PopulationHighDistinctCount:
            addFeatures(E_PopulationHighDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_PopulationRare:
            addFeatures(E_PopulationRare, result);
            BOOST_FALLTHROUGH;
        case E_PopulationRareCount:
            addFeatures(E_PopulationRareCount, result);
            BOOST_FALLTHROUGH;
        case E_PopulationFreqRare:
            addFeatures(E_PopulationFreqRare, result);
            BOOST_FALLTHROUGH;
        case E_PopulationFreqRareCount:
            addFeatures(E_PopulationFreqRareCount, result);
            BOOST_FALLTHROUGH;
        case E_PopulationLowCounts:
            addFeatures(E_PopulationLowCounts, result);
            BOOST_FALLTHROUGH;
        case E_PopulationHighCounts:
            addFeatures(E_PopulationHighCounts, result);
            BOOST_FALLTHROUGH;
        case E_PopulationInfoContent:
            addFeatures(E_PopulationInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_PopulationLowInfoContent:
            addFeatures(E_PopulationLowInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_PopulationHighInfoContent:
            addFeatures(E_PopulationHighInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_PopulationTimeOfDay:
            addFeatures(E_PopulationTimeOfDay, result);
            BOOST_FALLTHROUGH;
        case E_PopulationTimeOfWeek:
            addFeatures(E_PopulationTimeOfWeek, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetric:
            addFeatures(E_PopulationMetric, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricMean:
            addFeatures(E_PopulationMetricMean, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricLowMean:
            addFeatures(E_PopulationMetricLowMean, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricHighMean:
            addFeatures(E_PopulationMetricHighMean, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricMedian:
            addFeatures(E_PopulationMetricMedian, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricLowMedian:
            addFeatures(E_PopulationMetricLowMedian, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricHighMedian:
            addFeatures(E_PopulationMetricHighMedian, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricMin:
            addFeatures(E_PopulationMetricMin, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricMax:
            addFeatures(E_PopulationMetricMax, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricVariance:
            addFeatures(E_PopulationMetricVariance, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricLowVariance:
            addFeatures(E_PopulationMetricLowVariance, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricHighVariance:
            addFeatures(E_PopulationMetricHighVariance, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricSum:
            addFeatures(E_PopulationMetricSum, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricLowSum:
            addFeatures(E_PopulationMetricLowSum, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMetricHighSum:
            addFeatures(E_PopulationMetricHighSum, result);
            BOOST_FALLTHROUGH;
        case E_PopulationLatLong:
            addFeatures(E_PopulationLatLong, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMaxVelocity:
            addFeatures(E_PopulationMaxVelocity, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMinVelocity:
            addFeatures(E_PopulationMinVelocity, result);
            BOOST_FALLTHROUGH;
        case E_PopulationMeanVelocity:
            addFeatures(E_PopulationMeanVelocity, result);
            BOOST_FALLTHROUGH;
        case E_PopulationSumVelocity:
            addFeatures(E_PopulationSumVelocity, result);
            BOOST_FALLTHROUGH;
        case E_PeersCount:
            addFeatures(E_PeersCount, result);
            BOOST_FALLTHROUGH;
        case E_PeersLowCounts:
            addFeatures(E_PeersLowCounts, result);
            BOOST_FALLTHROUGH;
        case E_PeersHighCounts:
            addFeatures(E_PeersHighCounts, result);
            BOOST_FALLTHROUGH;
        case E_PeersDistinctCount:
            addFeatures(E_PeersDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_PeersLowDistinctCount:
            addFeatures(E_PeersLowDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_PeersHighDistinctCount:
            addFeatures(E_PeersHighDistinctCount, result);
            BOOST_FALLTHROUGH;
        case E_PeersInfoContent:
            addFeatures(E_PeersInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_PeersLowInfoContent:
            addFeatures(E_PeersLowInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_PeersHighInfoContent:
            addFeatures(E_PeersHighInfoContent, result);
            BOOST_FALLTHROUGH;
        case E_PeersTimeOfDay:
            addFeatures(E_PeersTimeOfDay, result);
            BOOST_FALLTHROUGH;
        case E_PeersTimeOfWeek:
            addFeatures(E_PeersTimeOfWeek, result);
    }

    for (TFeatureFunctionVecMapItr i = result.begin(); i != result.end(); ++i) {
        std::sort(i->second.begin(), i->second.end());
    }

    return result;
}

const TFeatureFunctionVecMap FUNCTIONS_BY_FEATURE = buildFeatureFunctionMap();


//! Get the function with the fewest features.
EFunction mostSpecific(const TFunctionVec &functions) {
    if (functions.empty()) {
        LOG_ABORT("No functions specified");
    }

    EFunction   result = functions[0];
    std::size_t numberFeatures = features(functions[0]).size();
    for (std::size_t i = 1u; i < functions.size(); ++i) {
        std::size_t n = features(functions[i]).size();
        if (n < numberFeatures) {
            result = functions[i];
            numberFeatures = n;
        }
    }
    return result;
}

}

const TFeatureVec &features(EFunction function) {
    switch (function) {
        case E_IndividualCount:
            return INDIVIDUAL_COUNT_FEATURES;
        case E_IndividualNonZeroCount:
            return INDIVIDUAL_NON_ZERO_COUNT_FEATURES;
        case E_IndividualRareCount:
            return INDIVIDUAL_RARE_COUNT_FEATURES;
        case E_IndividualRareNonZeroCount:
            return INDIVIDUAL_RARE_NON_ZERO_COUNT_FEATURES;
        case E_IndividualRare:
            return INDIVIDUAL_RARE_FEATURES;
        case E_IndividualLowCounts:
            return INDIVIDUAL_LOW_COUNTS_FEATURES;
        case E_IndividualHighCounts:
            return INDIVIDUAL_HIGH_COUNTS_FEATURES;
        case E_IndividualLowNonZeroCount:
            return INDIVIDUAL_LOW_NON_ZERO_COUNT_FEATURES;
        case E_IndividualHighNonZeroCount:
            return INDIVIDUAL_HIGH_NON_ZERO_COUNT_FEATURES;
        case E_IndividualDistinctCount:
            return INDIVIDUAL_DISTINCT_COUNT_FEATURES;
        case E_IndividualLowDistinctCount:
            return INDIVIDUAL_LOW_DISTINCT_COUNT_FEATURES;
        case E_IndividualHighDistinctCount:
            return INDIVIDUAL_HIGH_DISTINCT_COUNT_FEATURES;
        case E_IndividualInfoContent:
            return INDIVIDUAL_INFO_CONTENT_FEATURES;
        case E_IndividualHighInfoContent:
            return INDIVIDUAL_HIGH_INFO_CONTENT_FEATURES;
        case E_IndividualLowInfoContent:
            return INDIVIDUAL_LOW_INFO_CONTENT_FEATURES;
        case E_IndividualTimeOfDay:
            return INDIVIDUAL_TIME_OF_DAY_FEATURES;
        case E_IndividualTimeOfWeek:
            return INDIVIDUAL_TIME_OF_WEEK_FEATURES;
        case E_IndividualMetric:
            return INDIVIDUAL_METRIC_FEATURES;
        case E_IndividualMetricMean:
            return INDIVIDUAL_METRIC_MEAN_FEATURES;
        case E_IndividualMetricLowMean:
            return INDIVIDUAL_METRIC_LOW_MEAN_FEATURES;
        case E_IndividualMetricHighMean:
            return INDIVIDUAL_METRIC_HIGH_MEAN_FEATURES;
        case E_IndividualMetricMedian:
            return INDIVIDUAL_METRIC_MEDIAN_FEATURES;
        case E_IndividualMetricLowMedian:
            return INDIVIDUAL_METRIC_LOW_MEDIAN_FEATURES;
        case E_IndividualMetricHighMedian:
            return INDIVIDUAL_METRIC_HIGH_MEDIAN_FEATURES;
        case E_IndividualMetricMin:
            return INDIVIDUAL_METRIC_MIN_FEATURES;
        case E_IndividualMetricMax:
            return INDIVIDUAL_METRIC_MAX_FEATURES;
        case E_IndividualMetricVariance:
            return INDIVIDUAL_METRIC_VARIANCE_FEATURES;
        case E_IndividualMetricLowVariance:
            return INDIVIDUAL_METRIC_LOW_VARIANCE_FEATURES;
        case E_IndividualMetricHighVariance:
            return INDIVIDUAL_METRIC_HIGH_VARIANCE_FEATURES;
        case E_IndividualMetricSum:
            return INDIVIDUAL_METRIC_SUM_FEATURES;
        case E_IndividualMetricLowSum:
            return INDIVIDUAL_METRIC_LOW_SUM_FEATURES;
        case E_IndividualMetricHighSum:
            return INDIVIDUAL_METRIC_HIGH_SUM_FEATURES;
        case E_IndividualMetricNonNullSum:
            return INDIVIDUAL_METRIC_NON_NULL_SUM_FEATURES;
        case E_IndividualMetricLowNonNullSum:
            return INDIVIDUAL_METRIC_LOW_NON_NULL_SUM_FEATURES;
        case E_IndividualMetricHighNonNullSum:
            return INDIVIDUAL_METRIC_HIGH_NON_NULL_SUM_FEATURES;
        case E_IndividualLatLong:
            return INDIVIDUAL_LAT_LONG_FEATURES;
        case E_IndividualMaxVelocity:
            return INDIVIDUAL_MAX_VELOCITY_FEATURES;
        case E_IndividualMinVelocity:
            return INDIVIDUAL_MIN_VELOCITY_FEATURES;
        case E_IndividualMeanVelocity:
            return INDIVIDUAL_MEAN_VELOCITY_FEATURES;
        case E_IndividualSumVelocity:
            return INDIVIDUAL_SUM_VELOCITY_FEATURES;
        case E_PopulationCount:
            return POPULATION_COUNT_FEATURES;
        case E_PopulationDistinctCount:
            return POPULATION_DISTINCT_COUNT_FEATURES;
        case E_PopulationLowDistinctCount:
            return POPULATION_LOW_DISTINCT_COUNT_FEATURES;
        case E_PopulationHighDistinctCount:
            return POPULATION_HIGH_DISTINCT_COUNT_FEATURES;
        case E_PopulationRare:
            return POPULATION_RARE_FEATURES;
        case E_PopulationRareCount:
            return POPULATION_RARE_COUNT_FEATURES;
        case E_PopulationFreqRare:
            return POPULATION_FREQ_RARE_FEATURES;
        case E_PopulationFreqRareCount:
            return POPULATION_FREQ_RARE_COUNT_FEATURES;
        case E_PopulationLowCounts:
            return POPULATION_LOW_COUNTS_FEATURES;
        case E_PopulationHighCounts:
            return POPULATION_HIGH_COUNTS_FEATURES;
        case E_PopulationInfoContent:
            return POPULATION_INFO_CONTENT_FEATURES;
        case E_PopulationLowInfoContent:
            return POPULATION_LOW_INFO_CONTENT_FEATURES;
        case E_PopulationHighInfoContent:
            return POPULATION_HIGH_INFO_CONTENT_FEATURES;
        case E_PopulationTimeOfDay:
            return POPULATION_TIME_OF_DAY_FEATURES;
        case E_PopulationTimeOfWeek:
            return POPULATION_TIME_OF_WEEK_FEATURES;
        case E_PopulationMetric:
            return POPULATION_METRIC_FEATURES;
        case E_PopulationMetricMean:
            return POPULATION_METRIC_MEAN_FEATURES;
        case E_PopulationMetricLowMean:
            return POPULATION_METRIC_LOW_MEAN_FEATURES;
        case E_PopulationMetricHighMean:
            return POPULATION_METRIC_HIGH_MEAN_FEATURES;
        case E_PopulationMetricMedian:
            return POPULATION_METRIC_MEDIAN_FEATURES;
        case E_PopulationMetricLowMedian:
            return POPULATION_METRIC_LOW_MEDIAN_FEATURES;
        case E_PopulationMetricHighMedian:
            return POPULATION_METRIC_HIGH_MEDIAN_FEATURES;
        case E_PopulationMetricMin:
            return POPULATION_METRIC_MIN_FEATURES;
        case E_PopulationMetricMax:
            return POPULATION_METRIC_MAX_FEATURES;
        case E_PopulationMetricVariance:
            return POPULATION_METRIC_VARIANCE_FEATURES;
        case E_PopulationMetricLowVariance:
            return POPULATION_METRIC_LOW_VARIANCE_FEATURES;
        case E_PopulationMetricHighVariance:
            return POPULATION_METRIC_HIGH_VARIANCE_FEATURES;
        case E_PopulationMetricSum:
            return POPULATION_METRIC_SUM_FEATURES;
        case E_PopulationMetricLowSum:
            return POPULATION_METRIC_LOW_SUM_FEATURES;
        case E_PopulationMetricHighSum:
            return POPULATION_METRIC_HIGH_SUM_FEATURES;
        case E_PopulationLatLong:
            return POPULATION_LAT_LONG_FEATURES;
        case E_PopulationMaxVelocity:
            return POPULATION_MAX_VELOCITY_FEATURES;
        case E_PopulationMinVelocity:
            return POPULATION_MIN_VELOCITY_FEATURES;
        case E_PopulationMeanVelocity:
            return POPULATION_MEAN_VELOCITY_FEATURES;
        case E_PopulationSumVelocity:
            return POPULATION_SUM_VELOCITY_FEATURES;
        case E_PeersCount:
            return PEERS_COUNT_FEATURES;
        case E_PeersLowCounts:
            return PEERS_LOW_COUNTS_FEATURES;
        case E_PeersHighCounts:
            return PEERS_HIGH_COUNTS_FEATURES;
        case E_PeersDistinctCount:
            return PEERS_DISTINCT_COUNT_FEATURES;
        case E_PeersLowDistinctCount:
            return PEERS_LOW_DISTINCT_COUNT_FEATURES;
        case E_PeersHighDistinctCount:
            return PEERS_HIGH_DISTINCT_COUNT_FEATURES;
        case E_PeersInfoContent:
            return PEERS_INFO_CONTENT_FEATURES;
        case E_PeersLowInfoContent:
            return PEERS_LOW_INFO_CONTENT_FEATURES;
        case E_PeersHighInfoContent:
            return PEERS_HIGH_INFO_CONTENT_FEATURES;
        case E_PeersTimeOfDay:
            return PEERS_TIME_OF_DAY_FEATURES;
        case E_PeersTimeOfWeek:
            return PEERS_TIME_OF_WEEK_FEATURES;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return EMPTY_FEATURES;
}

EFunction function(const TFeatureVec &features) {
    if (features.empty()) {
        LOG_ERROR("No features default to '" << print(E_IndividualCount) << "'");
        return E_IndividualCount;
    }

    TFunctionVec candidates;
    std::size_t  i = 0u;

    for (/**/; candidates.empty() && i < features.size(); ++i) {
        TFeatureFunctionVecMapCItr functionsItr = FUNCTIONS_BY_FEATURE.find(features[i]);
        if (functionsItr == FUNCTIONS_BY_FEATURE.end()) {
            LOG_WARN("No functions for feature " << model_t::print(features[i]))
            continue;
        }
        candidates = functionsItr->second;
    }

    LOG_TRACE("candidate " << core::CContainerPrinter::print(candidates));
    TFunctionVec fallback = candidates;

    TFunctionVec tmp;
    tmp.reserve(candidates.size());
    for (/**/; !candidates.empty() && i < features.size(); ++i) {
        TFeatureFunctionVecMapCItr functionsItr = FUNCTIONS_BY_FEATURE.find(features[i]);
        if (functionsItr == FUNCTIONS_BY_FEATURE.end()) {
            LOG_WARN("No functions for feature " << model_t::print(features[i]))
            continue;
        }

        LOG_TRACE("candidate = " << core::CContainerPrinter::print(functionsItr->second));
        std::set_intersection(candidates.begin(), candidates.end(),
                              functionsItr->second.begin(), functionsItr->second.end(),
                              std::back_inserter(tmp));
        candidates.swap(tmp);
        tmp.clear();
    }

    if (candidates.empty()) {
        EFunction result = mostSpecific(fallback);
        LOG_ERROR("Inconsistent features " << core::CContainerPrinter::print(features)
                                           << " defaulting to '" << print(result) << "'");
        return result;
    }

    return mostSpecific(candidates);
}

const std::string &name(EFunction function) {
    switch (function) {
        case E_IndividualCount:                return detail::COUNT;
        case E_IndividualNonZeroCount:         return detail::NON_ZERO_COUNT;
        case E_IndividualRareCount:            return detail::COUNT;
        case E_IndividualRareNonZeroCount:     return detail::RARE_NON_ZERO_COUNT;
        case E_IndividualRare:                 return detail::RARE;
        case E_IndividualLowCounts:            return detail::LOW_COUNT;
        case E_IndividualHighCounts:           return detail::HIGH_COUNT;
        case E_IndividualLowNonZeroCount:      return detail::LOW_NON_ZERO_COUNT;
        case E_IndividualHighNonZeroCount:     return detail::HIGH_NON_ZERO_COUNT;
        case E_IndividualDistinctCount:        return detail::DISTINCT_COUNT;
        case E_IndividualLowDistinctCount:     return detail::LOW_DISTINCT_COUNT;
        case E_IndividualHighDistinctCount:    return detail::HIGH_DISTINCT_COUNT;
        case E_IndividualInfoContent:          return detail::INFO_CONTENT;
        case E_IndividualHighInfoContent:      return detail::HIGH_INFO_CONTENT;
        case E_IndividualLowInfoContent:       return detail::LOW_INFO_CONTENT;
        case E_IndividualTimeOfDay:            return detail::TIME_OF_DAY;
        case E_IndividualTimeOfWeek:           return detail::TIME_OF_WEEK;
        case E_IndividualMetric:               return detail::METRIC;
        case E_IndividualMetricMean:           return detail::MEAN;
        case E_IndividualMetricLowMean:        return detail::LOW_MEAN;
        case E_IndividualMetricHighMean:       return detail::HIGH_MEAN;
        case E_IndividualMetricMedian:         return detail::MEDIAN;
        case E_IndividualMetricLowMedian:      return detail::LOW_MEDIAN;
        case E_IndividualMetricHighMedian:     return detail::HIGH_MEDIAN;
        case E_IndividualMetricMin:            return detail::MIN;
        case E_IndividualMetricMax:            return detail::MAX;
        case E_IndividualMetricVariance:       return detail::VARIANCE;
        case E_IndividualMetricLowVariance:    return detail::LOW_VARIANCE;
        case E_IndividualMetricHighVariance:   return detail::HIGH_VARIANCE;
        case E_IndividualMetricSum:            return detail::SUM;
        case E_IndividualMetricLowSum:         return detail::LOW_SUM;
        case E_IndividualMetricHighSum:        return detail::HIGH_SUM;
        case E_IndividualMetricNonNullSum:     return detail::NON_NULL_SUM;
        case E_IndividualMetricLowNonNullSum:  return detail::LOW_NON_NULL_SUM;
        case E_IndividualMetricHighNonNullSum: return detail::HIGH_NON_NULL_SUM;
        case E_IndividualLatLong:              return detail::LAT_LONG;
        case E_IndividualMaxVelocity:          return detail::MAX_VELOCITY;
        case E_IndividualMinVelocity:          return detail::MIN_VELOCITY;
        case E_IndividualMeanVelocity:         return detail::MEAN_VELOCITY;
        case E_IndividualSumVelocity:          return detail::SUM_VELOCITY;
        case E_PopulationCount:                return detail::COUNT;
        case E_PopulationDistinctCount:        return detail::DISTINCT_COUNT;
        case E_PopulationLowDistinctCount:     return detail::LOW_DISTINCT_COUNT;
        case E_PopulationHighDistinctCount:    return detail::HIGH_DISTINCT_COUNT;
        case E_PopulationRare:                 return detail::RARE;
        case E_PopulationRareCount:            return detail::RARE_COUNT;
        case E_PopulationFreqRare:             return detail::FREQ_RARE;
        case E_PopulationFreqRareCount:        return detail::FREQ_RARE_COUNT;
        case E_PopulationLowCounts:            return detail::LOW_COUNT;
        case E_PopulationHighCounts:           return detail::HIGH_COUNT;
        case E_PopulationInfoContent:          return detail::INFO_CONTENT;
        case E_PopulationLowInfoContent:       return detail::LOW_INFO_CONTENT;
        case E_PopulationHighInfoContent:      return detail::HIGH_INFO_CONTENT;
        case E_PopulationTimeOfDay:            return detail::TIME_OF_DAY;
        case E_PopulationTimeOfWeek:           return detail::TIME_OF_WEEK;
        case E_PopulationMetric:               return detail::METRIC;
        case E_PopulationMetricMean:           return detail::MEAN;
        case E_PopulationMetricLowMean:        return detail::LOW_MEAN;
        case E_PopulationMetricHighMean:       return detail::HIGH_MEAN;
        case E_PopulationMetricMedian:         return detail::MEDIAN;
        case E_PopulationMetricLowMedian:      return detail::LOW_MEDIAN;
        case E_PopulationMetricHighMedian:     return detail::HIGH_MEDIAN;
        case E_PopulationMetricMin:            return detail::MIN;
        case E_PopulationMetricMax:            return detail::MAX;
        case E_PopulationMetricVariance:       return detail::VARIANCE;
        case E_PopulationMetricLowVariance:    return detail::LOW_VARIANCE;
        case E_PopulationMetricHighVariance:   return detail::HIGH_VARIANCE;
        case E_PopulationMetricSum:            return detail::SUM;
        case E_PopulationMetricLowSum:         return detail::LOW_SUM;
        case E_PopulationMetricHighSum:        return detail::HIGH_SUM;
        case E_PopulationLatLong:              return detail::LAT_LONG;
        case E_PopulationMaxVelocity:          return detail::MAX_VELOCITY;
        case E_PopulationMinVelocity:          return detail::MIN_VELOCITY;
        case E_PopulationMeanVelocity:         return detail::MEAN_VELOCITY;
        case E_PopulationSumVelocity:          return detail::SUM_VELOCITY;
        case E_PeersCount:                     return detail::COUNT;
        case E_PeersLowCounts:                 return detail::LOW_COUNT;
        case E_PeersHighCounts:                return detail::HIGH_COUNT;
        case E_PeersDistinctCount:             return detail::DISTINCT_COUNT;
        case E_PeersLowDistinctCount:          return detail::LOW_DISTINCT_COUNT;
        case E_PeersHighDistinctCount:         return detail::HIGH_DISTINCT_COUNT;
        case E_PeersInfoContent:               return detail::INFO_CONTENT;
        case E_PeersLowInfoContent:            return detail::LOW_INFO_CONTENT;
        case E_PeersHighInfoContent:           return detail::HIGH_INFO_CONTENT;
        case E_PeersTimeOfDay:                 return detail::TIME_OF_DAY;
        case E_PeersTimeOfWeek:                return detail::TIME_OF_WEEK;
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return detail::UNEXPECTED_FUNCTION;
}

std::string print(EFunction function) {
    switch (function) {
        case E_IndividualCount:                return "individual count";
        case E_IndividualNonZeroCount:         return "individual non-zero count";
        case E_IndividualRareCount:            return "individual rare count";
        case E_IndividualRareNonZeroCount:     return "individual rare non-zero count";
        case E_IndividualRare:                 return "individual rare";
        case E_IndividualLowCounts:            return "individual low counts";
        case E_IndividualHighCounts:           return "individual high counts";
        case E_IndividualLowNonZeroCount:      return "individual low non-zero count";
        case E_IndividualHighNonZeroCount:     return "individual high non-zero count";
        case E_IndividualDistinctCount:        return "individual distinct count";
        case E_IndividualLowDistinctCount:     return "individual low distinct count";
        case E_IndividualHighDistinctCount:    return "individual high distinct count";
        case E_IndividualInfoContent:          return "individual info_content";
        case E_IndividualHighInfoContent:      return "individual high_info_content";
        case E_IndividualLowInfoContent:       return "individual low_info_content";
        case E_IndividualTimeOfDay:            return "individual time-of-day";
        case E_IndividualTimeOfWeek:           return "individual time-of-week";
        case E_IndividualMetric:               return "individual metric";
        case E_IndividualMetricMean:           return "individual metric mean";
        case E_IndividualMetricLowMean:        return "individual metric low mean";
        case E_IndividualMetricHighMean:       return "individual metric high mean";
        case E_IndividualMetricMedian:         return "individual metric median";
        case E_IndividualMetricLowMedian:      return "individual metric low median";
        case E_IndividualMetricHighMedian:     return "individual metric high median";
        case E_IndividualMetricMin:            return "individual metric minimum";
        case E_IndividualMetricMax:            return "individual metric maximum";
        case E_IndividualMetricVariance:       return "individual metric variance";
        case E_IndividualMetricLowVariance:    return "individual metric low variance";
        case E_IndividualMetricHighVariance:   return "individual metric high variance";
        case E_IndividualMetricSum:            return "individual metric sum";
        case E_IndividualMetricLowSum:         return "individual metric low sum";
        case E_IndividualMetricHighSum:        return "individual metric high sum";
        case E_IndividualMetricNonNullSum:     return "individual metric non-null sum";
        case E_IndividualMetricLowNonNullSum:  return "individual metric low non-null sum";
        case E_IndividualMetricHighNonNullSum: return "individual high non-null sum";
        case E_IndividualLatLong:              return "individual latitude/longitude";
        case E_IndividualMaxVelocity:          return "individual max velocity";
        case E_IndividualMinVelocity:          return "individual min velocity";
        case E_IndividualMeanVelocity:         return "individual mean velocity";
        case E_IndividualSumVelocity:          return "individual sum velocity";
        case E_PopulationCount:                return "population count";
        case E_PopulationDistinctCount:        return "population distinct count";
        case E_PopulationLowDistinctCount:     return "population low distinct count";
        case E_PopulationHighDistinctCount:    return "population high distinct count";
        case E_PopulationRare:                 return "population rare";
        case E_PopulationRareCount:            return "population rare count";
        case E_PopulationFreqRare:             return "population frequent rare";
        case E_PopulationFreqRareCount:        return "population frequent rare count";
        case E_PopulationLowCounts:            return "population low count";
        case E_PopulationHighCounts:           return "population high count";
        case E_PopulationInfoContent:          return "population information content";
        case E_PopulationLowInfoContent:       return "population low information content";
        case E_PopulationHighInfoContent:      return "population high information content";
        case E_PopulationTimeOfDay:            return "population time-of-day";
        case E_PopulationTimeOfWeek:           return "population time-of-week";
        case E_PopulationMetric:               return "population metric";
        case E_PopulationMetricMean:           return "population metric mean";
        case E_PopulationMetricLowMean:        return "population metric low mean";
        case E_PopulationMetricHighMean:       return "population metric high mean";
        case E_PopulationMetricMedian:         return "population metric median";
        case E_PopulationMetricLowMedian:      return "population metric low median";
        case E_PopulationMetricHighMedian:     return "population metric high median";
        case E_PopulationMetricMin:            return "population metric minimum";
        case E_PopulationMetricMax:            return "population metric maximum";
        case E_PopulationMetricVariance:       return "population metric variance";
        case E_PopulationMetricLowVariance:    return "population metric low variance";
        case E_PopulationMetricHighVariance:   return "population metric high variance";
        case E_PopulationMetricSum:            return "population metric sum";
        case E_PopulationMetricLowSum:         return "population metric low sum";
        case E_PopulationMetricHighSum:        return "population metric high sum";
        case E_PopulationLatLong:              return "population latitude/longitude";
        case E_PopulationMaxVelocity:          return "population max velocity";
        case E_PopulationMinVelocity:          return "population min velocity";
        case E_PopulationMeanVelocity:         return "population mean velocity";
        case E_PopulationSumVelocity:          return "population sum velocity";
        case E_PeersCount:                     return "peers count";
        case E_PeersLowCounts:                 return "peers low count";
        case E_PeersHighCounts:                return "peers high count";
        case E_PeersDistinctCount:             return "peers distinct count";
        case E_PeersLowDistinctCount:          return "peers low distinct count";
        case E_PeersHighDistinctCount:         return "peers high distinct count";
        case E_PeersInfoContent:               return "peers information content";
        case E_PeersLowInfoContent:            return "peers low information content";
        case E_PeersHighInfoContent:           return "peers high information content";
        case E_PeersTimeOfDay:                 return "peers time-of-day";
        case E_PeersTimeOfWeek:                return "peers time-of-week";
    }

    LOG_ERROR("Unexpected function = " << static_cast<int>(function));
    return "-";
}

std::ostream &operator<<(std::ostream &o, EFunction function) {
    return o << print(function);
}

}
}
}

