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

#ifndef INCLUDED_ml_maths_t_MathsTypes_h
#define INCLUDED_ml_maths_t_MathsTypes_h

#include <core/CFloatStorage.h>
#include <core/CSmallVector.h>

#include <maths/ImportExport.h>

#include <utility>
#include <vector>

namespace ml {
namespace maths {
using core::CFloatStorage;
class CCalendarComponent;
class CSeasonalComponent;
}
namespace maths_t {

using TDoubleDoublePr = std::pair<double, double>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TDouble10Vec4Vec = core::CSmallVector<TDouble10Vec, 4>;
using TDouble10Vec4Vec1Vec = core::CSmallVector<TDouble10Vec4Vec, 1>;
using TSeasonalComponentVec = std::vector<maths::CSeasonalComponent>;
using TCalendarComponentVec = std::vector<maths::CCalendarComponent>;

//! An enumeration of the types of data which can be modeled.
//!
//! The possible values are:
//!   -# DiscreteData: which indicates the data take a finite number
//!      of distinct values.
//!   -# IntegerData: which indicates the data takes only integer
//!      values.
//!   -# ContinuousData: which indicates the takes real values.
//!   -# MixedData: which indicates the data can be decomposed into
//!      some combination of the other three data types.
enum EDataType { E_DiscreteData, E_IntegerData, E_ContinuousData, E_MixedData };

//! An enumeration of the types of weight which can be applied
//! when adding samples, calculating marginal likelihood or
//! computing the probability of less likely samples for the prior
//! and clusterer class hierarchies. The possible values are:
//!   -# CountWeight: which we interpret as equivalent to the sample
//!      vector containing "weight" samples of the corresponding
//!      value.
//!   -# SeasonalVarianceScaleWeight: which we interpret as the
//!      transformation \f$Y = m + \sqrt{\lambda}(X - m)\f$ where
//!      \f$Y\f$ is distributed as the predictive distribution.
//!   -# CountVarianceScaleWeight: which we interpret as equivalent
//!      to the sample likelihood function having its variance scaled
//!      by "weight" w.r.t. the likelihood function corresponding
//!      to the prior parameters.
//!   -# WinsorisationWeight: only affects update where it basically
//!      behaves like CountWeight except for the way it interacts
//!      with clustering.
enum ESampleWeightStyle {
    E_SampleCountWeight,
    E_SampleSeasonalVarianceScaleWeight,
    E_SampleCountVarianceScaleWeight,
    E_SampleWinsorisationWeight
};

//! IMPORTANT: this must be kept this up-to-date with ESampleWeightStyle.
const std::size_t NUMBER_WEIGHT_STYLES = 4;

using TWeightStyleVec = core::CSmallVector<ESampleWeightStyle, 4>;

//! Extract the effective sample count from a collection of weights.
MATHS_EXPORT
double count(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights);

//! Extract the effective sample count from a collection of weights.
MATHS_EXPORT
TDouble10Vec count(std::size_t dimension, const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec& weights);

//! Extract the effective sample count with which to update a model
//! from a collection of weights.
MATHS_EXPORT
double countForUpdate(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights);

//! Extract the effective sample count with which to update a model
//! from a collection of weights.
MATHS_EXPORT
TDouble10Vec countForUpdate(std::size_t dimension, const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec& weights);

//! Extract the variance scale from a collection of weights.
MATHS_EXPORT
double seasonalVarianceScale(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights);

//! Extract the variance scale from a collection of weights.
MATHS_EXPORT
TDouble10Vec seasonalVarianceScale(std::size_t dimension, const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec& weights);

//! Extract the variance scale from a collection of weights.
MATHS_EXPORT
double countVarianceScale(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights);

//! Extract the variance scale from a collection of weights.
MATHS_EXPORT
TDouble10Vec countVarianceScale(std::size_t dimension, const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasSeasonalVarianceScale(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasSeasonalVarianceScale(const TWeightStyleVec& weightStyles, const TDouble4Vec1Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasSeasonalVarianceScale(const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasSeasonalVarianceScale(const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec1Vec& weights);

//! Check if a non-unit count variance scale applies.
MATHS_EXPORT
bool hasCountVarianceScale(const TWeightStyleVec& weightStyles, const TDouble4Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasCountVarianceScale(const TWeightStyleVec& weightStyles, const TDouble4Vec1Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasCountVarianceScale(const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasCountVarianceScale(const TWeightStyleVec& weightStyles, const TDouble10Vec4Vec1Vec& weights);

//! Enumerates the possible probability of less likely sample calculations.
//!
//! The possible calculations are:
//!   -# OneSidedBelow - for which we calculate the probability of
//!      seeing a smaller value.
//!   -# TwoSided - for which we calculate the probability of a lower
//!      likelihood value.
//!   -# OneSidedAbove - for which we calculate the probability of
//!      seeing a larger value.
//!
//! The idea of the "one sided" calculations is to support order
//! statistics for which we aren't interested in smaller values
//! for the sample minimum or larger values for the sample maximum.
//! Note that we normalize the one sided probabilities so they equal
//! 1 at the distribution median.
enum EProbabilityCalculation { E_OneSidedBelow, E_TwoSided, E_OneSidedAbove };

//! This controls the calculation of the cluster probabilities.
//! There are two styles available:
//!   -# Equal: all clusters have equal weight.
//!   -# Fraction: the weight of a cluster is proportional to the
//!      number of points which have been assigned to the cluster.
enum EClusterWeightCalc { E_ClustersEqualWeight, E_ClustersFractionWeight };

//! A set of statuses which track the result of a floating point
//! calculations. These provide finer grained information than
//! a pass/fail boolean which can be used to take appropriate
//! action in the calling context.
enum EFloatingPointErrorStatus { E_FpNoErrors = 0x0, E_FpOverflowed = 0x1, E_FpFailed = 0x2, E_FpAllErrors = 0x3 };

//! Enumerates the cases that a collection of samples is either in
//! the left tail, right tail or a mixture or neither of the tails
//! of a distribution. The possible values are:
//!   -# Undetermined is a special value used internally to indicate
//!      the tail is not calculated,
//!   -# Left denotes the case all samples are to the left of every
//!      mode in the distribution,
//!   -# Right denotes the case all samples are to the right of every
//!      mode in the distribution,
//!   -# Mixed or neither is used to denote the case that some are
//!      to left, some to the right and/or some are between the left
//!      and rightmost modes.
enum ETail { E_UndeterminedTail = 0x0, E_LeftTail = 0x1, E_RightTail = 0x2, E_MixedOrNeitherTail = 0x3 };
}
}

#endif // INCLUDED_ml_maths_t_MathsTypes_h
