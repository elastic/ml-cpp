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

#ifndef INCLUDED_ml_model_function_t_FunctionTypes_h
#define INCLUDED_ml_model_function_t_FunctionTypes_h

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <iosfwd>
#include <string>
#include <vector>

namespace ml
{
namespace model
{
namespace function_t
{

//! An enumeration of possible functions we can run on a data stream
//! on which we do anomaly detection. These map to a set of data
//! features which we'll model (see function_t::features for details).
//! The possible functions are:
//!   -# Individual count: for which we look at all bucket counts and
//!      is intended to detect large deviations in the rate of messages
//!      w.r.t. the history of a single category of categorical data.
//!   -# Individual non-zero count: for which we look at the non-zero
//!      bucket counts and is intended to detect large deviations in
//!      the rate of messages w.r.t. the history of a single category
//!      of categorical data for sparse data.
//!   -# Individual rare count: for which we look at all bucket counts
//!      and calculate the relative rarity (in time) of each category
//!      and is intended to detect both rare messages and large deviations
//!      in the rate of messages w.r.t. the history of categorical data.
//!   -# Individual rare non-zero count: for which we look at the non-zero
//!      bucket counts and calculate the relative rarity (in time) of each
//!      category and is intended to detect rare messages and large
//!      deviations in the rate of messages w.r.t. the history sparse
//!      categorical data.
//!   -# Individual distinct count: for which we look at the number of
//!      unique field values per bucket per person
//!   -# Individual info content: for which we look at the entropy or
//!      compressibility of field values per bucket per person
//!   -# Individual metric: for which we look at the minimum, mean and
//!      maximum values in a bucket for a single metric time series over
//!      time and is our default analysis for metric data.
//!   -# Individual metric mean: for which we look at just the mean
//!      value in a bucket for a single metric time series over time.
//!   -# Individual metric min: for which we look at just the minimum
//!      value in a bucket for a single metric time series over time.
//!   -# Individual metric max: for which we look at just the maximum
//!      value in a bucket for a single metric time series over time.
//!   -# Individual metric sum: for which we look at the sum in a bucket
//!      for a single metric time series over time treating empty buckets
//!      as zero.
//!   -# Individual metric non-zero sum: for which we look at the sum
//!      in a bucket for a single metric time series over time ignoring
//!      empty buckets.
//!   -# Population count: for which we look at the number of messages
//!      person generates in a bucket optionally partitioned by category.
//!      This is used for analyzing categorical data as a population.
//!   -# Population distinct count: for which we look at the number of
//!      different categories a person in a population has in a bucket.
//!      This is used for analyzing categorical data as a population.
//!   -# Population rare: for which we analyze the number of people
//!      with each category over all time. This is used for analyzing
//!      categorical data as a population.
//!   -# Population rare count: for which we analyze the number of people
//!      with each category over all time and also the number of messages
//!      each person generates in a bucket partitioned by category. This
//!      is used for analyzing categorical data as a population.
//!   -# Population frequent rare: for which we analyze the number of
//!      people with each category over all time and weight the person's
//!      categories according to their relative frequency when computing
//!      their overall probability. This is used for analyzing categorical
//!      data as a population.
//!   -# Population frequent rare count: for which we analyze the number
//!      of people with each category over all time and also the number
//!      of messages each person generates in a bucket partitioned by a
//!      category and weight the person's categories according to their
//!      relative frequency when computing their overall probability.
//!      This is used for analyzing categorical data as a population.
//!   -# Population distinct count: for which we analyse the number of
//!      unique field values per bucket per person and attribute
//!   -# Population info content: for which we look at the entropy or
//!      compressibility of field values per bucket per person and attribute
//!   -# Population metric: for which we look at the minimum, mean and
//!      maximum values each person generates in a bucket optionally
//!      partitioned by category. This is used for analyzing metric data
//!      as a population.
//!   -# Population metric minimum: for which we look at the minimum each
//!      person generates in a bucket optionally partitioned by category.
//!      This is used for analyzing metric data as a population.
//!   -# Population metric maximum: for which we look at the maximum each
//!      person generates in a bucket optionally partitioned by category.
//!      This is used for analyzing metric data as a population.
//!   -# Population metric sum: for which we look at the sum of the metric
//!      values each person generates in a bucket optionally partitioned by
//!      a category. This is used for analyzing metric data as a population.
enum EFunction
{
    // IMPORTANT: The integer values associated with these enum values are
    // stored in persisted state.  DO NOT CHANGE EXISTING NUMBERS, as this
    // will invalidate persisted state.  Any new enum values that are added
    // should be given new numbers.

    // Event rate functions
    E_IndividualCount = 0,
    E_IndividualNonZeroCount = 1,
    E_IndividualRareCount = 2,
    E_IndividualRareNonZeroCount = 3,
    E_IndividualRare = 4,
    E_IndividualLowCounts = 5,
    E_IndividualHighCounts = 6,
    E_IndividualLowNonZeroCount = 7,
    E_IndividualHighNonZeroCount = 8,
    E_IndividualDistinctCount = 9,
    E_IndividualLowDistinctCount = 10,
    E_IndividualHighDistinctCount = 11,
    E_IndividualInfoContent = 12,
    E_IndividualLowInfoContent = 13,
    E_IndividualHighInfoContent = 14,
    E_IndividualTimeOfDay = 15,
    E_IndividualTimeOfWeek = 16,

    // Metric functions
    E_IndividualMetric = 100,
    E_IndividualMetricMean = 101,
    E_IndividualMetricMin = 102,
    E_IndividualMetricMax = 103,
    E_IndividualMetricSum = 104,
    E_IndividualMetricLowMean = 105,
    E_IndividualMetricHighMean = 106,
    E_IndividualMetricLowSum = 107,
    E_IndividualMetricHighSum = 108,
    E_IndividualMetricNonNullSum = 109,
    E_IndividualMetricLowNonNullSum = 110,
    E_IndividualMetricHighNonNullSum = 111,
    E_IndividualLatLong = 112,
    E_IndividualMinVelocity = 113,
    E_IndividualMaxVelocity = 114,
    E_IndividualMeanVelocity = 115,
    E_IndividualSumVelocity = 116,
    E_IndividualMetricMedian = 117,
    E_IndividualMetricVariance = 118,
    E_IndividualMetricLowVariance = 119,
    E_IndividualMetricHighVariance = 120,
    E_IndividualMetricLowMedian = 121,
    E_IndividualMetricHighMedian = 122,

    // Population event rate functions
    E_PopulationCount = 200,
    E_PopulationDistinctCount = 201,
    E_PopulationRare = 202,
    E_PopulationRareCount = 203,
    E_PopulationFreqRare = 204,
    E_PopulationFreqRareCount = 205,
    E_PopulationLowCounts = 206,
    E_PopulationHighCounts = 207,
    E_PopulationInfoContent = 208,
    E_PopulationLowInfoContent = 209,
    E_PopulationHighInfoContent = 210,
    E_PopulationLowDistinctCount = 211,
    E_PopulationHighDistinctCount = 212,
    E_PopulationTimeOfDay = 213,
    E_PopulationTimeOfWeek = 214,

    // Population metric functions
    E_PopulationMetric = 300,
    E_PopulationMetricMean = 301,
    E_PopulationMetricMin = 302,
    E_PopulationMetricMax = 303,
    E_PopulationMetricSum = 304,
    E_PopulationMetricLowMean = 305,
    E_PopulationMetricHighMean = 306,
    E_PopulationMetricLowSum = 307,
    E_PopulationMetricHighSum = 308,
    E_PopulationLatLong = 309,
    E_PopulationMinVelocity = 310,
    E_PopulationMaxVelocity = 311,
    E_PopulationMeanVelocity = 312,
    E_PopulationSumVelocity = 313,
    E_PopulationMetricMedian = 314,
    E_PopulationMetricVariance = 315,
    E_PopulationMetricLowVariance = 316,
    E_PopulationMetricHighVariance = 317,
    E_PopulationMetricLowMedian = 318,
    E_PopulationMetricHighMedian = 319,

    // Peers event rate functions
    E_PeersCount = 400,
    E_PeersLowCounts = 401,
    E_PeersHighCounts = 402,
    E_PeersDistinctCount = 403,
    E_PeersLowDistinctCount = 404,
    E_PeersHighDistinctCount = 405,
    E_PeersInfoContent = 406,
    E_PeersLowInfoContent = 407,
    E_PeersHighInfoContent = 408,
    //E_PeersRare = 409,
    //E_PeersRareCount = 410,
    //E_PeersFreqRare = 411,
    //E_PeersFreqRareCount = 412,
    E_PeersTimeOfDay = 413,
    E_PeersTimeOfWeek = 414
};

using TFunctionVec = std::vector<EFunction>;

//! Is this function for use with the individual models?
MODEL_EXPORT
bool isIndividual(EFunction function);

//! Is this function for use with the population models?
MODEL_EXPORT
bool isPopulation(EFunction function);

//! Is this function for use with the peer group models?
MODEL_EXPORT
bool isPeers(EFunction function);

//! Is this function for use with metric models?
MODEL_EXPORT
bool isMetric(EFunction function);

//! Is this function for use with forecast?
MODEL_EXPORT
bool isForecastSupported(EFunction function);

//! Get the mapping from function to data features.
MODEL_EXPORT
const model_t::TFeatureVec &features(EFunction function);

//! The inverse mapping from features to function.
MODEL_EXPORT
EFunction function(const model_t::TFeatureVec &features);

//! Get the name of \p function.
MODEL_EXPORT
const std::string &name(EFunction function);

//! Get a string description of \p function.
MODEL_EXPORT
std::string print(EFunction function);

//! Overload std stream << operator.
MODEL_EXPORT
std::ostream &operator<<(std::ostream &o, EFunction function);

}
}
}

#endif // INCLUDED_ml_model_function_t_FunctionTypes_h

