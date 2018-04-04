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

#ifndef INCLUDED_ml_model_t_ModelTypes_h
#define INCLUDED_ml_model_t_ModelTypes_h

#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/MathsTypes.h>

#include <model/ImportExport.h>

#include <functional>
#include <string>
#include <utility>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CMultivariatePrior;
class CPrior;
class CTimeSeriesDecompositionInterface;
}
namespace model {
class CInfluenceCalculator;
struct SModelParams;
}

namespace model_t {

using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
using TDouble1VecDouble1VecPr = std::pair<TDouble1Vec, TDouble1Vec>;
using TInfluenceCalculatorCPtr = boost::shared_ptr<const model::CInfluenceCalculator>;

//! The types of model available.
//!
//! Rate models are used for data categories that don't have
//! associated values, i.e. log messages, etc. Metric models
//! are used for data categories that have an associated value,
//! i.e. CPU utilization, etc.
enum EModelType { E_Counting, E_EventRateOnline, E_MetricOnline };

//! Get a string description of a model type.
MODEL_EXPORT
std::string print(EModelType type);

//! \brief Describes the result type.
//!
//! DESCRIPTION:\n
//! The types of the results:
//!   -# Interim: if the bucket isn't complete.
//!   -# Final: if the bucket is complete.
//!   -# Unconditional: if a result corresponds to a feature value which
//!      is unusual in its own right.
//!   -# Conditional: if a result corresponds to a feature value which
//!      is unusual only after conditioning on some other.
//!
//! IMPLEMENTATION:\n
//! Uses 1-of-n bitwise encoding of the different result binary descriptors
//! so the size is just that of an unsigned int.
class MODEL_EXPORT CResultType {
public:
    enum EInterimOrFinal { E_Interim = 1, E_Final = 2 };

    enum EConditionalOrUnconditional { E_Unconditional = 4, E_Conditional = 8 };

public:
    CResultType() : m_Type(0) {}
    CResultType(EInterimOrFinal type) : m_Type(type) {}
    CResultType(EConditionalOrUnconditional type) : m_Type(type) {}
    explicit CResultType(unsigned int type) : m_Type(type) {}

    //! Set whether or not this is interim.
    void set(EInterimOrFinal type) {
        m_Type = m_Type & (E_Unconditional | E_Conditional);
        m_Type = m_Type | type;
    }

    //! Set whether or not this is conditional.
    void set(EConditionalOrUnconditional type) {
        m_Type = m_Type & (E_Interim | E_Final);
        m_Type = m_Type | type;
    }

    //! Check if this is interim.
    bool isInterim() const { return (m_Type & static_cast<unsigned int>(E_Interim)) != 0; }

    //! Get as interim or final enumeration.
    EInterimOrFinal asInterimOrFinal() const { return this->isInterim() ? E_Interim : E_Final; }

    //! Check if this is unconditional.
    bool isUnconditional() const { return (m_Type & static_cast<unsigned int>(E_Unconditional)) != 0; }

    //! Get as conditional or unconditional enumeration.
    EConditionalOrUnconditional asConditionalOrUnconditional() const { return this->isUnconditional() ? E_Unconditional : E_Conditional; }

    //! Get as an unsigned integer.
    unsigned int asUint() const { return m_Type; }

private:
    //! Encodes the result type.
    unsigned int m_Type;
};

//! The feature naming is systematic and all subsequent feature
//! names should conform to the following syntax:\n
//!   E_(Individual|Population|Peers)\<feature name\>[By|Of][Bucket][And][Person][And][Attribute]\n
//!
//! In particular, the Individual prefix denotes that the feature
//! is to be analyzed individually in time whereas the Population
//! prefix that should be analyzed as part of a population, the
//! Bucket suffix denotes that the feature applies to a single
//! bucketing interval, the Person suffix that it relates to a
//! single person, the Attribute suffix that it relates to a single
//! attribute of a person, \<feature name\> describes the sort of
//! data being collected and "by" and "and" should be inserted so
//! that the name scans properly. For example,\n
//!   E_IndividualCountByBucketAndPerson
//!
//! is the count of events a person generates in a single bucketing
//! interval to be analyzed individually over time.
//!
//! The possible data features we extract are:
//!   -# IndividualCountByBucketAndPerson: is the count of events for
//!      each person in a bucketing interval.
//!   -# IndividualNonZeroCountByBucketAndPerson: is the non-zero count
//!      of events for each person in a bucketing interval.
//!   -# IndividualTotalBucketCountByPerson: is the count of bucketing
//!      intervals containing each person over all time.
//!   -# IndividualIndicatorOfBucketPerson: is the indicator of the presence
//!      of a person in a bucketing interval.
//!   -# IndividualLowCountsByBucketAndPerson: is for detecting unusually
//!      low counts of events for each person in a bucketing interval.
//!   -# IndividualHighCountsByBucketAndPerson: is for detecting unusually
//!      high count of events for each person in a bucketing interval.
//!   -# IndividualLowNonZeroCountByBucketAndPerson: is for detecting
//!      unusually low non-zero counts of events for each person in a
//!      bucketing interval.
//!   -# IndividualHighNonZeroCountByBucketAndPerson: is for detecting
//!      unusually high non-zero counts of events for each person in a
//!      bucketing interval.
//!   -# IndividualUniqueCountByBucketAndPerson: is the number of distinct
//!      values of a categorical field for each person in a bucketing
//!      interval.
//!   -# IndividualInfoContentByBucketAndPerson: is the information content
//!      (or compressibility) of values of a categorical field for each
//!      person in a bucketing interval.
//!   -# IndividualLowInfoContentByBucketAndPerson: is for detecting
//!      unusually low information content (or compressibility) of values
//!      of a categorical field for each person in a bucketing interval.
//!   -# IndividualHighInfoContentByBucketAndPerson: is for detecting
//!      unusually high information content (or compressibility) of values
//!      of a categorical field for each person in a bucketing interval.
//!   -# IndividualArrivalTimesByPerson: is for detecting unusual
//!      event rates based on analyzing the times between messages.
//!      Note, this is automatically selected for modeling event
//!      rates if the time between messages is long w.r.t. the bucketing
//!      interval is large. As such it is not exposed as a function and
//!      is used in conjunction with IndividualCountByBucketAndPerson.
//!   -# IndividualLongArrivalTimesByPerson: is for detecting unusually
//!      low event rates based on analyzing the times between messages.
//!      Note, this is automatically selected for modeling event rates
//!      if the time between messages is long w.r.t. the bucketing interval
//!      is large. As such it is not exposed as a function and is used
//!      in conjunction with IndividualLowCountsByBucketAndPerson.
//!   -# IndividualShortArrivalTimesByPerson: is for detecting unusually
//!      high event rates based on analyzing the times between messages.
//!      Note, this is automatically selected for modeling event rates
//!      if the time between messages is long w.r.t. the bucketing interval
//!      is large. As such it is not exposed as a function and is used
//!      in conjunction with IndividualHighCountsByBucketAndPerson.
//!   -# IndividualMeanByPerson: is the mean of a fixed number of metric
//!      values for each person.
//!   -# IndividualLowMeanByPerson: is for detecting unusually low means
//!      of a fixed number of metric values for each person.
//!   -# IndividualHighMeanByPerson: is for detecting unusually high
//!      means of a fixed number of metric values for each person.
//!   -# IndividualMedianByPerson: is the median of a fixed number of metric
//!      values for each person.
//!   -# IndividualMinByPerson: is the minimum of a fixed number of metric
//!      values for each person.
//!   -# IndividualMaxByPerson: is the maximum of a fixed number of metric
//!      values for each person.
//!   -# IndividualVarianceByPerson: is the variance of a fixed number of
//!      metric values for each person.
//!   -# IndividualSumByBucketAndPerson: is the sum of the metric values
//!      in a bucketing interval for each person.
//!   -# IndividualLowSumByBucketAndPerson: is for detecting unusually
//!      low sums of the metric values in a bucketing interval for each
//!      person.
//!   -# IndividualHighSumByBucketAndPerson: is for detecting unusually
//!      high sums of the metric values in a bucketing interval for each
//!      person.
//!   -# IndividualNonNullSumByBucketAndPerson: is the sum of the metric
//!      values in a bucketing interval ignoring empty buckets.
//!   -# IndividualLowNonNullSumByBucketAndPerson: is for detecting
//!      unusually low sums of the metric values in a bucketing interval
//!      ignoring empty buckets.
//!   -# IndividualHighNonNullSumByBucketAndPerson: is for detecting
//!      unusually high sums of the metric values in a bucketing interval
//!      ignoring empty buckets.
//!   -# PopulationAttributeTotalCountByPerson: is the count of events for
//!      each attribute for each person over all time analyzed as categorical
//!      data.
//!   -# PopulationCountByBucketPersonAndAttribute: is the non-zero count
//!      of events for each (person, attribute) pair in a bucketing interval
//!      analyzed as a population.
//!   -# PopulationIndicatorOfBucketPersonAndAttribute: is the indicator
//!      of the presence of a single (person, attribute) pair in a bucketing
//!      interval analyzed as a population.
//!   -# PopulationUniquePersonCountByAttribute: is the count of unique people
//!      who generate events for each attribute over all time.
//!   -# PopulationUniqueCountByBucketPersonAndAttribute: is the count of
//!      unique values for each person and attribute in a bucketing interval
//!      analyzed as a population.
//!   -# PopulationLowCountsByBucketPersonAndAttribute: is for detecting
//!      unusually low non-zero counts of events for each (person, attribute)
//!      pair in a bucketing interval analyzed as a population.
//!   -# PopulationHighCountsByBucketPersonAndAttribute: is for detecting
//!      unusually high non-zero counts of events for each (person, attribute)
//!      pair in a bucketing interval analyzed as a population.
//!   -# PopulationInfoContentByBucketAndPerson: is the information content
//!      (or compressibility) of values of a categorical field for each person
//!      and attribute in a bucketing interval analyzed as a population.
//!   -# PopulationLowInfoContentByBucketAndPerson: is for detecting
//!      unusually low information content (or compressibility) of values
//!      for each person and attribute in a bucketing interval.
//!   -# PopulationHighInfoContentByBucketAndPerson: is for detecting
//!      unusually high information content (or compressibility) of values
//!      for each person and attribute in a bucketing interval.
//!   -# PopulationMeanByPersonAndAttribute: is the mean of a fixed number
//!      of metric values for each (person, attribute) pair analyzed as a
//!      population.
//!   -# PopulationLowMeanByPersonAndAttribute: is for detecting unusually
//!      low means of a fixed number of metric values for each
//!      (person, attribute) pair analyzed as a population.
//!   -# PopulationHighMeanByPersonAndAttribute: is for detecting unusually
//!      high means of a fixed number of metric values for each
//!      (person, attribute) pair analyzed as a population.
//!   -# PopulationMedianByPersonAndAttribute: is the median of a fixed number
//!      of metric values for each (person, attribute) pair analyzed as a
//!      population.
//!   -# PopulationMinByPersonAndAttribute: is the minimum of a fixed number
//!      of metric values for each (person, attribute) pair analyzed as a
//!      population.
//!   -# PopulationMaxByPersonAndAttribute: is the maximum of a fixed number
//!      of metric values for each (person, attribute) pair analyzed as a
//!      population.
//!   -# PopulationVarianceByPersonAndAttribute: is the variance of a fixed
//!      number of metric values for each (person, attribute) pair analyzed
//!      as a population.
//!   -# PopulationSumByBucketPersonAndAttribute: is the sum of the
//!      metric values in a bucketing interval for each (person, attribute)
//!      pair analyzed as a population.
//!   -# PopulationLowSumByBucketPersonAndAttribute: is for detecting
//!      unusually low sums of the metric values in a bucketing interval
//!      for each (person, attribute) pair analyzed as a population.
//!   -# PopulationHighSumByBucketPersonAndAttribute: is for detecting
//!      unusually high sums of the metric values in a bucketing interval
//!      for each (person, attribute) pair analyzed as a population.
//!   -# PeersAttributeTotalCountByPerson: is the count of events for each
//!      attribute for each person over all time analyzed as categorical
//!      data.
//!   -# PeersCountByBucketPersonAndAttribute: is the non-zero count of
//!      events for each (person, attribute) pair in a bucketing interval
//!      analyzed by peer group.
//!   -# PeersUniqueCountByBucketPersonAndAttribute: is the count of unique
//!      values for each person and attribute in a bucketing interval analyzed
//!      by peer group.
//!   -# PeersLowCountsByBucketPersonAndAttribute: is for detecting unusually
//!      low non-zero counts of events for each (person, attribute) pair in a
//!      bucketing interval analyzed as a population.
//!   -# PeersHighCountsByBucketPersonAndAttribute: is for detecting unusually
//!      high non-zero counts of events for each (person, attribute) pair in a
//!      bucketing interval analyzed as a population.
//!   -# PopulationInfoContentByBucketAndPerson: is for detecting unusual
//!      information content (or compressibility) of values for each person
//!      and attribute in a bucketing interval.
//!   -# PopulationLowInfoContentByBucketAndPerson: is for detecting
//!      unusualy low information content (or compressibility) of values
//!      for each person and attribute in a bucketing interval.
//!   -# PopulationHighInfoContentByBucketAndPerson: is for detecting
//!      unusualy high information content (or compressibility) of values
//!      for each person and attribute in a bucketing interval.
enum EFeature {
    // IMPORTANT: The integer values associated with these enum values are
    // stored in persisted state.  DO NOT CHANGE EXISTING NUMBERS, as this
    // will invalidate persisted state.  Any new enum values that are added
    // should be given new numbers.

    // Individual event rate features
    E_IndividualCountByBucketAndPerson = 0,
    E_IndividualNonZeroCountByBucketAndPerson = 1,
    E_IndividualTotalBucketCountByPerson = 2,
    E_IndividualIndicatorOfBucketPerson = 3,
    E_IndividualLowCountsByBucketAndPerson = 4,
    E_IndividualHighCountsByBucketAndPerson = 5,
    E_IndividualArrivalTimesByPerson = 6,
    E_IndividualLongArrivalTimesByPerson = 7,
    E_IndividualShortArrivalTimesByPerson = 8,
    E_IndividualLowNonZeroCountByBucketAndPerson = 9,
    E_IndividualHighNonZeroCountByBucketAndPerson = 10,
    E_IndividualUniqueCountByBucketAndPerson = 11,
    E_IndividualLowUniqueCountByBucketAndPerson = 12,
    E_IndividualHighUniqueCountByBucketAndPerson = 13,
    E_IndividualInfoContentByBucketAndPerson = 14,
    E_IndividualLowInfoContentByBucketAndPerson = 15,
    E_IndividualHighInfoContentByBucketAndPerson = 16,
    E_IndividualTimeOfDayByBucketAndPerson = 17,
    E_IndividualTimeOfWeekByBucketAndPerson = 18,

    // Individual metric features
    E_IndividualMeanByPerson = 100,
    E_IndividualMinByPerson = 101,
    E_IndividualMaxByPerson = 102,
    E_IndividualSumByBucketAndPerson = 103,
    E_IndividualLowMeanByPerson = 106,
    E_IndividualHighMeanByPerson = 107,
    E_IndividualLowSumByBucketAndPerson = 108,
    E_IndividualHighSumByBucketAndPerson = 109,
    E_IndividualNonNullSumByBucketAndPerson = 110,
    E_IndividualLowNonNullSumByBucketAndPerson = 111,
    E_IndividualHighNonNullSumByBucketAndPerson = 112,
    E_IndividualMeanLatLongByPerson = 113,
    E_IndividualMaxVelocityByPerson = 114,
    E_IndividualMinVelocityByPerson = 115,
    E_IndividualMeanVelocityByPerson = 116,
    E_IndividualSumVelocityByPerson = 117,
    E_IndividualMedianByPerson = 118,
    E_IndividualVarianceByPerson = 119,
    E_IndividualLowVarianceByPerson = 120,
    E_IndividualHighVarianceByPerson = 121,
    E_IndividualLowMedianByPerson = 122,
    E_IndividualHighMedianByPerson = 123,

    // Population event rate features
    E_PopulationAttributeTotalCountByPerson = 200,
    E_PopulationCountByBucketPersonAndAttribute = 201,
    E_PopulationIndicatorOfBucketPersonAndAttribute = 202,
    E_PopulationUniquePersonCountByAttribute = 203,
    E_PopulationUniqueCountByBucketPersonAndAttribute = 204,
    E_PopulationLowCountsByBucketPersonAndAttribute = 205,
    E_PopulationHighCountsByBucketPersonAndAttribute = 206,
    E_PopulationInfoContentByBucketPersonAndAttribute = 207,
    E_PopulationLowInfoContentByBucketPersonAndAttribute = 208,
    E_PopulationHighInfoContentByBucketPersonAndAttribute = 209,
    E_PopulationLowUniqueCountByBucketPersonAndAttribute = 210,
    E_PopulationHighUniqueCountByBucketPersonAndAttribute = 211,
    E_PopulationTimeOfDayByBucketPersonAndAttribute = 212,
    E_PopulationTimeOfWeekByBucketPersonAndAttribute = 213,

    // Population metric features
    E_PopulationMeanByPersonAndAttribute = 300,
    E_PopulationMinByPersonAndAttribute = 301,
    E_PopulationMaxByPersonAndAttribute = 302,
    E_PopulationSumByBucketPersonAndAttribute = 303,
    E_PopulationLowMeanByPersonAndAttribute = 304,
    E_PopulationHighMeanByPersonAndAttribute = 305,
    E_PopulationLowSumByBucketPersonAndAttribute = 306,
    E_PopulationHighSumByBucketPersonAndAttribute = 307,
    E_PopulationMeanLatLongByPersonAndAttribute = 308,
    E_PopulationMaxVelocityByPersonAndAttribute = 309,
    E_PopulationMinVelocityByPersonAndAttribute = 310,
    E_PopulationMeanVelocityByPersonAndAttribute = 311,
    E_PopulationSumVelocityByPersonAndAttribute = 312,
    E_PopulationMedianByPersonAndAttribute = 313,
    E_PopulationVarianceByPersonAndAttribute = 314,
    E_PopulationLowVarianceByPersonAndAttribute = 315,
    E_PopulationHighVarianceByPersonAndAttribute = 316,
    E_PopulationLowMedianByPersonAndAttribute = 317,
    E_PopulationHighMedianByPersonAndAttribute = 318,

    // Peer group event rate feature
    E_PeersAttributeTotalCountByPerson = 400,
    E_PeersCountByBucketPersonAndAttribute = 401,
    //E_PeersIndicatorOfBucketPersonAndAttribute = 402,
    //E_PeersUniquePersonCountByAttribute = 403,
    E_PeersUniqueCountByBucketPersonAndAttribute = 404,
    E_PeersLowCountsByBucketPersonAndAttribute = 405,
    E_PeersHighCountsByBucketPersonAndAttribute = 406,
    E_PeersInfoContentByBucketPersonAndAttribute = 407,
    E_PeersLowInfoContentByBucketPersonAndAttribute = 408,
    E_PeersHighInfoContentByBucketPersonAndAttribute = 409,
    E_PeersLowUniqueCountByBucketPersonAndAttribute = 410,
    E_PeersHighUniqueCountByBucketPersonAndAttribute = 411,
    E_PeersTimeOfDayByBucketPersonAndAttribute = 412,
    E_PeersTimeOfWeekByBucketPersonAndAttribute = 413,

    // Peer group metric features
    E_PeersMeanByPersonAndAttribute = 500,
    E_PeersMinByPersonAndAttribute = 501,
    E_PeersMaxByPersonAndAttribute = 502,
    E_PeersSumByBucketPersonAndAttribute = 503,
    E_PeersLowMeanByPersonAndAttribute = 504,
    E_PeersHighMeanByPersonAndAttribute = 505,
    E_PeersLowSumByBucketPersonAndAttribute = 506,
    E_PeersHighSumByBucketPersonAndAttribute = 507,
    E_PeersMedianByPersonAndAttribute = 508
};

using TFeatureVec = std::vector<EFeature>;

//! Get the dimension of the feature \p feature.
MODEL_EXPORT
std::size_t dimension(EFeature feature);

//! For features which have extra statistics in order to compute
//! influence remove those extra statistics.
MODEL_EXPORT
TDouble2Vec1Vec stripExtraStatistics(EFeature feature, const TDouble2Vec1Vec& values);

//! Check if \p feature is categorical, i.e. we should create
//! a distribution over the distinct values observed for the
//! feature.
MODEL_EXPORT
bool isCategorical(EFeature feature);

//! Check if \p feature is modeling time of day or week.
MODEL_EXPORT
bool isDiurnal(EFeature feature);

//! Check if \p feature is modeling latitude and longitude.
MODEL_EXPORT
bool isLatLong(EFeature feature);

//! Check if \p feature is a constant value. It is inefficient
//! to model the distribution of these features and we use a
//! special very lightweight dummy prior.
MODEL_EXPORT
bool isConstant(EFeature feature);

//! Check if \p feature models the mean of some quantity.
MODEL_EXPORT
bool isMeanFeature(EFeature feature);

//! Check if \p feature models the median of some quantity.
MODEL_EXPORT
bool isMedianFeature(EFeature feature);

//! Check if \p feature models the min of some quantity.
MODEL_EXPORT
bool isMinFeature(EFeature feature);

//! Check if \p feature models the max of some quantity.
MODEL_EXPORT
bool isMaxFeature(EFeature feature);

//! Check if \p feature models the variance of some quantity.
MODEL_EXPORT
bool isVarianceFeature(EFeature feature);

//! Check if \p feature models the sum of some quantity.
MODEL_EXPORT
bool isSumFeature(EFeature feature);

//! Get the variance scale to use when the typical measurement
//! count per sample is \p sampleCount and we are computing
//! the likelihood of \p feature on a sample with measurement
//! count of \p count.
MODEL_EXPORT
double varianceScale(EFeature feature, double sampleCount, double count);

//! Check if the feature is sampled.
MODEL_EXPORT
bool isSampled(EFeature feature);

//! Get the minimum useful sample count for a feature.
MODEL_EXPORT
unsigned minimumSampleCount(EFeature feature);

//! Offset count features so that their range starts at zero.
MODEL_EXPORT
double offsetCountToZero(EFeature feature, double count);

//! Offset count features so that their range starts at zero.
MODEL_EXPORT
void offsetCountToZero(EFeature feature, TDouble1Vec& count);

//! The inverse of offsetCountToZero.
MODEL_EXPORT
double inverseOffsetCountToZero(EFeature feature, double count);

//! The inverse of offsetCountToZero.
MODEL_EXPORT
void inverseOffsetCountToZero(EFeature feature, TDouble1Vec& count);

//! Check if the feature counts empty buckets.
MODEL_EXPORT
bool countsEmptyBuckets(EFeature feature);

//! Get the weight to apply to an empty bucket sample based on the
//! frequency \p feature at which empty buckets are seen for \p feature
//! and the cutoff for empty buckets directly.
MODEL_EXPORT
double emptyBucketCountWeight(EFeature feature, double frequency, double cutoff);

//! Get the rate at which \p feature learns.
MODEL_EXPORT
double learnRate(EFeature feature, const model::SModelParams& params);

//! Get the type of probability calculation to use for \p feature.
MODEL_EXPORT
maths_t::EProbabilityCalculation probabilityCalculation(EFeature feature);

//! Get the time of a sample of \p feature.
//!
//! For most features this is the centre of the bucket however for
//! some metric features the time is deduced in which case \p time
//! is used.
MODEL_EXPORT
core_t::TTime sampleTime(EFeature feature, core_t::TTime bucketStartTime, core_t::TTime bucketLength, core_t::TTime time = 0);

//! Get the support for \p feature.
MODEL_EXPORT
TDouble1VecDouble1VecPr support(EFeature feature);

//! Get the adjusted probability for \p feature.
MODEL_EXPORT
double adjustProbability(model_t::EFeature feature, core_t::TTime elapsedTime, double probability);

//! Get the influence calculator for \p feature.
MODEL_EXPORT
TInfluenceCalculatorCPtr influenceCalculator(EFeature feature);

//! Returns true if the probability calculation for \p feature
//! requires adjustment while interim results are produced.
MODEL_EXPORT
bool requiresInterimResultAdjustment(EFeature feature);

//! Maps internal feature names to human readable output function
//! names that can be included in the results.
MODEL_EXPORT
const std::string& outputFunctionName(EFeature feature);

//! Get a string description of \p feature.
MODEL_EXPORT
std::string print(EFeature feature);

//! Individual count feature case statement block.
//!
//! \note I've created some macro definitions of blocks of case
//! statements corresponding to categories of features that crop
//! up repeatedly in switch statements. I think it is less error
//! prone to minimize the number of different places they are
//! duplicated. Using macros particularly at header scope is
//! generally a bad idea so don't take this as a precedent for
//! crazy macro magic (see item 1.14 of our coding standards for
//! guidelines).
#define CASE_INDIVIDUAL_COUNT                                                                                                              \
    case model_t::E_IndividualCountByBucketAndPerson:                                                                                      \
    case model_t::E_IndividualNonZeroCountByBucketAndPerson:                                                                               \
    case model_t::E_IndividualTotalBucketCountByPerson:                                                                                    \
    case model_t::E_IndividualIndicatorOfBucketPerson:                                                                                     \
    case model_t::E_IndividualLowCountsByBucketAndPerson:                                                                                  \
    case model_t::E_IndividualHighCountsByBucketAndPerson:                                                                                 \
    case model_t::E_IndividualArrivalTimesByPerson:                                                                                        \
    case model_t::E_IndividualLongArrivalTimesByPerson:                                                                                    \
    case model_t::E_IndividualShortArrivalTimesByPerson:                                                                                   \
    case model_t::E_IndividualLowNonZeroCountByBucketAndPerson:                                                                            \
    case model_t::E_IndividualHighNonZeroCountByBucketAndPerson:                                                                           \
    case model_t::E_IndividualUniqueCountByBucketAndPerson:                                                                                \
    case model_t::E_IndividualLowUniqueCountByBucketAndPerson:                                                                             \
    case model_t::E_IndividualHighUniqueCountByBucketAndPerson:                                                                            \
    case model_t::E_IndividualInfoContentByBucketAndPerson:                                                                                \
    case model_t::E_IndividualHighInfoContentByBucketAndPerson:                                                                            \
    case model_t::E_IndividualLowInfoContentByBucketAndPerson:                                                                             \
    case model_t::E_IndividualTimeOfDayByBucketAndPerson:                                                                                  \
    case model_t::E_IndividualTimeOfWeekByBucketAndPerson

//! Individual metric feature case statement block.
#define CASE_INDIVIDUAL_METRIC                                                                                                             \
    case model_t::E_IndividualMeanByPerson:                                                                                                \
    case model_t::E_IndividualMedianByPerson:                                                                                              \
    case model_t::E_IndividualMinByPerson:                                                                                                 \
    case model_t::E_IndividualMaxByPerson:                                                                                                 \
    case model_t::E_IndividualSumByBucketAndPerson:                                                                                        \
    case model_t::E_IndividualLowMeanByPerson:                                                                                             \
    case model_t::E_IndividualHighMeanByPerson:                                                                                            \
    case model_t::E_IndividualLowSumByBucketAndPerson:                                                                                     \
    case model_t::E_IndividualHighSumByBucketAndPerson:                                                                                    \
    case model_t::E_IndividualNonNullSumByBucketAndPerson:                                                                                 \
    case model_t::E_IndividualLowNonNullSumByBucketAndPerson:                                                                              \
    case model_t::E_IndividualHighNonNullSumByBucketAndPerson:                                                                             \
    case model_t::E_IndividualMeanLatLongByPerson:                                                                                         \
    case model_t::E_IndividualMaxVelocityByPerson:                                                                                         \
    case model_t::E_IndividualMinVelocityByPerson:                                                                                         \
    case model_t::E_IndividualMeanVelocityByPerson:                                                                                        \
    case model_t::E_IndividualSumVelocityByPerson:                                                                                         \
    case model_t::E_IndividualVarianceByPerson:                                                                                            \
    case model_t::E_IndividualLowVarianceByPerson:                                                                                         \
    case model_t::E_IndividualHighVarianceByPerson:                                                                                        \
    case model_t::E_IndividualLowMedianByPerson:                                                                                           \
    case model_t::E_IndividualHighMedianByPerson

//! Population count feature case statement block.
#define CASE_POPULATION_COUNT                                                                                                              \
    case model_t::E_PopulationAttributeTotalCountByPerson:                                                                                 \
    case model_t::E_PopulationCountByBucketPersonAndAttribute:                                                                             \
    case model_t::E_PopulationIndicatorOfBucketPersonAndAttribute:                                                                         \
    case model_t::E_PopulationUniqueCountByBucketPersonAndAttribute:                                                                       \
    case model_t::E_PopulationLowUniqueCountByBucketPersonAndAttribute:                                                                    \
    case model_t::E_PopulationHighUniqueCountByBucketPersonAndAttribute:                                                                   \
    case model_t::E_PopulationUniquePersonCountByAttribute:                                                                                \
    case model_t::E_PopulationLowCountsByBucketPersonAndAttribute:                                                                         \
    case model_t::E_PopulationHighCountsByBucketPersonAndAttribute:                                                                        \
    case model_t::E_PopulationInfoContentByBucketPersonAndAttribute:                                                                       \
    case model_t::E_PopulationLowInfoContentByBucketPersonAndAttribute:                                                                    \
    case model_t::E_PopulationHighInfoContentByBucketPersonAndAttribute:                                                                   \
    case model_t::E_PopulationTimeOfDayByBucketPersonAndAttribute:                                                                         \
    case model_t::E_PopulationTimeOfWeekByBucketPersonAndAttribute

//! Population metric feature case statement block.
#define CASE_POPULATION_METRIC                                                                                                             \
    case model_t::E_PopulationMeanByPersonAndAttribute:                                                                                    \
    case model_t::E_PopulationMedianByPersonAndAttribute:                                                                                  \
    case model_t::E_PopulationMinByPersonAndAttribute:                                                                                     \
    case model_t::E_PopulationMaxByPersonAndAttribute:                                                                                     \
    case model_t::E_PopulationSumByBucketPersonAndAttribute:                                                                               \
    case model_t::E_PopulationLowMeanByPersonAndAttribute:                                                                                 \
    case model_t::E_PopulationHighMeanByPersonAndAttribute:                                                                                \
    case model_t::E_PopulationLowSumByBucketPersonAndAttribute:                                                                            \
    case model_t::E_PopulationHighSumByBucketPersonAndAttribute:                                                                           \
    case model_t::E_PopulationMeanLatLongByPersonAndAttribute:                                                                             \
    case model_t::E_PopulationMaxVelocityByPersonAndAttribute:                                                                             \
    case model_t::E_PopulationMinVelocityByPersonAndAttribute:                                                                             \
    case model_t::E_PopulationMeanVelocityByPersonAndAttribute:                                                                            \
    case model_t::E_PopulationSumVelocityByPersonAndAttribute:                                                                             \
    case model_t::E_PopulationVarianceByPersonAndAttribute:                                                                                \
    case model_t::E_PopulationLowVarianceByPersonAndAttribute:                                                                             \
    case model_t::E_PopulationHighVarianceByPersonAndAttribute:                                                                            \
    case model_t::E_PopulationLowMedianByPersonAndAttribute:                                                                               \
    case model_t::E_PopulationHighMedianByPersonAndAttribute

//! Peers count feature case statement block.
#define CASE_PEERS_COUNT                                                                                                                   \
    case model_t::E_PeersAttributeTotalCountByPerson:                                                                                      \
    case model_t::E_PeersCountByBucketPersonAndAttribute:                                                                                  \
    case model_t::E_PeersUniqueCountByBucketPersonAndAttribute:                                                                            \
    case model_t::E_PeersLowCountsByBucketPersonAndAttribute:                                                                              \
    case model_t::E_PeersHighCountsByBucketPersonAndAttribute:                                                                             \
    case model_t::E_PeersInfoContentByBucketPersonAndAttribute:                                                                            \
    case model_t::E_PeersLowInfoContentByBucketPersonAndAttribute:                                                                         \
    case model_t::E_PeersHighInfoContentByBucketPersonAndAttribute:                                                                        \
    case model_t::E_PeersLowUniqueCountByBucketPersonAndAttribute:                                                                         \
    case model_t::E_PeersHighUniqueCountByBucketPersonAndAttribute:                                                                        \
    case model_t::E_PeersTimeOfDayByBucketPersonAndAttribute:                                                                              \
    case model_t::E_PeersTimeOfWeekByBucketPersonAndAttribute

// Peers metric features case statement block.
#define CASE_PEERS_METRIC                                                                                                                  \
    case model_t::E_PeersMeanByPersonAndAttribute:                                                                                         \
    case model_t::E_PeersMedianByPersonAndAttribute:                                                                                       \
    case model_t::E_PeersMinByPersonAndAttribute:                                                                                          \
    case model_t::E_PeersMaxByPersonAndAttribute:                                                                                          \
    case model_t::E_PeersSumByBucketPersonAndAttribute:                                                                                    \
    case model_t::E_PeersLowMeanByPersonAndAttribute:                                                                                      \
    case model_t::E_PeersHighMeanByPersonAndAttribute:                                                                                     \
    case model_t::E_PeersLowSumByBucketPersonAndAttribute:                                                                                 \
    case model_t::E_PeersHighSumByBucketPersonAndAttribute

//! The categories of metric feature.
//!
//! These enumerate the distinct types of metric statistic
//! which we gather.
enum EMetricCategory {
    E_Mean,
    E_Min,
    E_Max,
    E_Sum,
    E_MultivariateMean,
    E_MultivariateMin,
    E_MultivariateMax,
    E_Median,
    E_Variance
    // If you add any more enumeration values here then be sure to update the
    // constant beneath too!
};

//! Must correspond to the number of enumeration values of EMetricCategory
static const size_t NUM_METRIC_CATEGORIES = 9;

//! Get the metric feature data corresponding to \p feature
//! if there is one.
MODEL_EXPORT
bool metricCategory(EFeature feature, EMetricCategory& result);

//! Get a string description of \p category.
MODEL_EXPORT
std::string print(EMetricCategory category);

//! The categories of event rate features.
//!
//! The enumerate the distinct type of event rate statistics
//! which we gather.
enum EEventRateCategory { E_MeanArrivalTimes, E_AttributePeople, E_UniqueValues, E_DiurnalTimes };

//! Get a string description of \p category.
MODEL_EXPORT
std::string print(EEventRateCategory category);

//! An enumeration of the categories of analysis we perform.
//!   -# Event rate: analysis of message rates.
//!   -# Metric: analysis of message values.
//!   -# Population event rate: analysis of message rates as
//!      a population.
//!   -# Population metric: analysis of message values as
//!      a population.
//!   -# Population event rate: analysis of message rates in
//!      peer groups.
//!   -# Population metric: analysis of message values in
//!      peer groups.
enum EAnalysisCategory { E_EventRate, E_Metric, E_PopulationEventRate, E_PopulationMetric, E_PeersEventRate, E_PeersMetric };

//! Get the category of analysis to which \p feature belongs.
MODEL_EXPORT
EAnalysisCategory analysisCategory(EFeature feature);

//! Get a string description of \p category.
MODEL_EXPORT
std::string print(EAnalysisCategory category);

//! The different ways we might be told the fields for receiving pre-summarised
//! data.
enum ESummaryMode {
    E_None,  //!< No pre-summarisation of input
    E_Manual //!< Config defines the field names for pre-summarised input
};

//! An enumeration of the options for ExcludeFrequent, which removes
//! samples from populations which occur in more than 10% of buckets
//!   -# E_XF_None: default behaviour
//!   -# E_XF_By: remove popular "attributes" from populations
//!   -# E_XF_Over: remove popular "people" from populations
//!   -# E_XF_Both: remove popular "people" and "attributes" from populations
enum EExcludeFrequent { E_XF_None = 0, E_XF_By = 1, E_XF_Over = 2, E_XF_Both = 3 };

//! An enumeration of the ResourceMonitor memory status -
//! Start in the OK state. Moves into soft limit if aggressive pruning
//! has taken place to avoid hitting the memory limit,
//! and goes to hard limit if samples have been dropped
enum EMemoryStatus {
    E_MemoryStatusOk = 0,        //!< Memory usage normal
    E_MemoryStatusSoftLimit = 1, //!< Pruning has taken place to reduce usage
    E_MemoryStatusHardLimit = 2  //!< Samples have been dropped
};

//! Get a string description of \p memoryStatus.
MODEL_EXPORT
std::string print(EMemoryStatus memoryStatus);

//! Styles of probability aggregation available:
//!   -# AggregatePeople: the style used to aggregate results for distinct
//!      values of the over and partition field.
//!   -# AggregateAttributes: the style used to aggregate results for distinct
//!      values of the by field.
//!   -# AggregateDetectors: the style used to aggregate distinct detector
//!      results.
enum EAggregationStyle { E_AggregatePeople = 0, E_AggregateAttributes = 1, E_AggregateDetectors = 2 };
const std::size_t NUMBER_AGGREGATION_STYLES = E_AggregateDetectors + 1;

//! Controllable aggregation parameters:
//!   -# JointProbabilityWeight: the weight assigned to the joint probability
//!      calculation, which should be in the range [0,1].
//!   -# ExtremeProbabilityWeight: the weight assigned to the extreme probability
//!      calculation, which should be in the range [0,1].
//!   -# MinExtremeSamples: the minimum number m of samples to consider in the
//!      m from n probability calculation.
//!   -# MaxExtremeSamples: the maximum number m of samples to consider in the
//!      m from n probability calculation.
enum EAggregationParam { E_JointProbabilityWeight = 0, E_ExtremeProbabilityWeight = 1, E_MinExtremeSamples = 2, E_MaxExtremeSamples = 3 };
const std::size_t NUMBER_AGGREGATION_PARAMS = E_MaxExtremeSamples + 1;

//! The dummy attribute identifier used for modeling individual features.
const std::size_t INDIVIDUAL_ANALYSIS_ATTRIBUTE_ID = 0u;
}
}

#endif // INCLUDED_ml_model_t_ModelTypes_h
