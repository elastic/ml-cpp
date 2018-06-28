/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_t_MathsTypes_h
#define INCLUDED_ml_maths_t_MathsTypes_h

#include <core/CFloatStorage.h>
#include <core/CSmallVector.h>

#include <maths/ImportExport.h>

#include <boost/array.hpp>

#include <cstddef>
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
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble10Vec = core::CSmallVector<double, 10>;
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

template<typename VECTOR>
using TWeightsAry = boost::array<VECTOR, NUMBER_WEIGHT_STYLES>;
using TDoubleWeightsAry = TWeightsAry<double>;
using TDoubleWeightsAry1Vec = core::CSmallVector<TDoubleWeightsAry, 1>;
using TDouble2VecWeightsAry = TWeightsAry<TDouble2Vec>;
using TDouble2VecWeightsAry1Vec = core::CSmallVector<TDouble2VecWeightsAry, 1>;
using TDouble10VecWeightsAry = TWeightsAry<TDouble10Vec>;
using TDouble10VecWeightsAry1Vec = core::CSmallVector<TDouble10VecWeightsAry, 1>;

namespace maths_types_detail {

//! \brief Constructs a unit weight.
template<typename VECTOR>
struct SUnitWeightFactory {
    static std::size_t dimension(const VECTOR& weight) { return weight.size(); }
    static VECTOR weight(std::size_t dimension) {
        return VECTOR(dimension, 1.0);
    }
};
//! \brief Constructs a unit weight.
template<>
struct SUnitWeightFactory<double> {
    static std::size_t dimension(double) { return 1; }
    static double weight(std::size_t) { return 1.0; }
};

//! \brief Add two weights.
template<typename VECTOR>
struct SWeightArithmetic {
    static void add(const VECTOR& lhs, VECTOR& rhs) {
        for (std::size_t i = 0u; i < lhs.size(); ++i) {
            rhs[i] += lhs[i];
        }
    }
    static void multiply(const VECTOR& lhs, VECTOR& rhs) {
        for (std::size_t i = 0u; i < lhs.size(); ++i) {
            rhs[i] *= lhs[i];
        }
    }
};
//! \brief Add two weights.
template<>
struct SWeightArithmetic<double> {
    static void add(double lhs, double& rhs) { rhs += lhs; }
    static void multiply(double lhs, double& rhs) { rhs *= lhs; }
};
}

//! \brief A collection of weight styles and weights.
class MATHS_EXPORT CUnitWeights {
public:
    //! A unit weight.
    static const TDoubleWeightsAry UNIT;
    //! A single unit weight.
    static const TDoubleWeightsAry1Vec SINGLE_UNIT;
    //! Get a conformable unit weight for \p weight.
    template<typename VECTOR>
    static TWeightsAry<VECTOR> unit(const VECTOR& weight) {
        return unit<VECTOR>(maths_types_detail::SUnitWeightFactory<VECTOR>::dimension(weight));
    }
    //! Get a unit weight for data with \p dimension.
    template<typename VECTOR>
    static TWeightsAry<VECTOR> unit(std::size_t dimension) {
        TWeightsAry<VECTOR> result;
        result.fill(maths_types_detail::SUnitWeightFactory<VECTOR>::weight(dimension));
        return result;
    }
    //! Get a single conformable unit weight for \p weight.
    template<typename VECTOR>
    static core::CSmallVector<TWeightsAry<VECTOR>, 1> singleUnit(const VECTOR& weight) {
        return {unit<VECTOR>(weight)};
    }
    //! Get a single unit weight for data with \p dimension.
    template<typename VECTOR>
    static core::CSmallVector<TWeightsAry<VECTOR>, 1> singleUnit(std::size_t dimension) {
        return {unit<VECTOR>(dimension)};
    }
};

//! Get a weights array with count weight \p weight.
template<typename VECTOR>
TWeightsAry<VECTOR> countWeight(const VECTOR& weight) {
    TWeightsAry<VECTOR> result(CUnitWeights::unit<VECTOR>(weight));
    result[E_SampleCountWeight] = weight;
    return result;
}

//! Get a weights array with count weight \p weight.
MATHS_EXPORT
TDoubleWeightsAry countWeight(double weight);

//! Get a weights array with count weight \p weight.
MATHS_EXPORT
TDouble10VecWeightsAry countWeight(double weight, std::size_t dimension);

//! Set the count weight in \p weights to \p weight.
template<typename VECTOR>
void setCount(const VECTOR& weight, TWeightsAry<VECTOR>& weights) {
    weights[E_SampleCountWeight] = weight;
}

//! Set the count weight in \p weights to \p weight.
MATHS_EXPORT
void setCount(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights);

//! Add \p weight to the count weight of \p weights.
template<typename VECTOR>
void addCount(const VECTOR& weight, TWeightsAry<VECTOR>& weights) {
    maths_types_detail::SWeightArithmetic<VECTOR>::add(weight, weights[E_SampleCountWeight]);
}

//! Extract the effective sample count from a collection of weights.
template<typename VECTOR>
const VECTOR& count(const TWeightsAry<VECTOR>& weights) {
    return weights[E_SampleCountWeight];
}

//! Extract the effective sample count with which to update a model
//! from a collection of weights.
MATHS_EXPORT
double countForUpdate(const TDoubleWeightsAry& weights);

//! Extract the effective sample count with which to update a model
//! from a collection of weights.
MATHS_EXPORT
TDouble10Vec countForUpdate(const TDouble10VecWeightsAry& weights);

//! Get a weights array with Winsorisation weight \p weight.
template<typename VECTOR>
TWeightsAry<VECTOR> winsorisationWeight(const VECTOR& weight) {
    TWeightsAry<VECTOR> result(CUnitWeights::unit<VECTOR>(weight));
    result[E_SampleWinsorisationWeight] = weight;
    return result;
}

//! Get a weights array with Winsorisation weight \p weight.
MATHS_EXPORT
TDoubleWeightsAry winsorisationWeight(double weight);

//! Get a weights array with Winsorisation weight \p weight.
MATHS_EXPORT
TDouble10VecWeightsAry winsorisationWeight(double weight, std::size_t dimension);

//! Set the Winsorisation weight in \p weights to \p weight.
template<typename VECTOR>
void setWinsorisationWeight(const VECTOR& weight, TWeightsAry<VECTOR>& weights) {
    weights[E_SampleWinsorisationWeight] = weight;
}

//! Set the Winsorisation weight in \p weights to \p weight.
MATHS_EXPORT
void setWinsorisationWeight(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights);

//! Extract the Winsorisation weight from a collection of weights.
template<typename VECTOR>
const VECTOR& winsorisationWeight(const TWeightsAry<VECTOR>& weights) {
    return weights[E_SampleWinsorisationWeight];
}

//! Check if a non-unit Winsorisation weight applies.
MATHS_EXPORT
bool isWinsorised(const TDoubleWeightsAry& weights);

//! Check if a non-unit Winsorisation weight applies.
MATHS_EXPORT
bool isWinsorised(const TDoubleWeightsAry1Vec& weights);

//! Check if a non-unit Winsorisation weight applies.
template<typename VECTOR>
bool isWinsorised(const TWeightsAry<VECTOR>& weights) {
    return std::any_of(weights[E_SampleWinsorisationWeight].begin(),
                       weights[E_SampleWinsorisationWeight].end(),
                       [](double weight) { return weight != 1.0; });
}

//! Check if a non-unit Winsorisation weight applies.
template<typename VECTOR>
bool isWinsorised(const core::CSmallVector<TWeightsAry<VECTOR>, 1>& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TWeightsAry<VECTOR>& weight) {
        return isWinsorised(weight);
    });
}

//! Get a weights array with seasonal variance scale \p weight.
template<typename VECTOR>
TWeightsAry<VECTOR> seasonalVarianceScaleWeight(const VECTOR& weight) {
    TWeightsAry<VECTOR> result(CUnitWeights::unit<VECTOR>(weight));
    result[E_SampleSeasonalVarianceScaleWeight] = weight;
    return result;
}

//! Get a weights vector with seasonal variance scale \p weight.
MATHS_EXPORT
TDoubleWeightsAry seasonalVarianceScaleWeight(double weight);

//! Get a weights vector with seasonal variance scale \p weight.
MATHS_EXPORT
TDouble10VecWeightsAry seasonalVarianceScaleWeight(double weight, std::size_t dimension);

//! Set the seasonal variance scale weight in \p weights to \p weight.
template<typename VECTOR>
void setSeasonalVarianceScale(const VECTOR& weight, TWeightsAry<VECTOR>& weights) {
    weights[E_SampleSeasonalVarianceScaleWeight] = weight;
}

//! Set the seasonal variance scale weight in \p weights to \p weight.
MATHS_EXPORT
void setSeasonalVarianceScale(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights);

//! Extract the variance scale from a collection of weights.
template<typename VECTOR>
const VECTOR& seasonalVarianceScale(const TWeightsAry<VECTOR>& weights) {
    return weights[E_SampleSeasonalVarianceScaleWeight];
}

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasSeasonalVarianceScale(const TDoubleWeightsAry& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasSeasonalVarianceScale(const TDoubleWeightsAry1Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
template<typename VECTOR>
bool hasSeasonalVarianceScale(const TWeightsAry<VECTOR>& weights) {
    return std::any_of(weights[E_SampleSeasonalVarianceScaleWeight].begin(),
                       weights[E_SampleSeasonalVarianceScaleWeight].end(),
                       [](double weight) { return weight != 1.0; });
}

//! Check if a non-unit seasonal variance scale applies.
template<typename VECTOR>
bool hasSeasonalVarianceScale(const core::CSmallVector<TWeightsAry<VECTOR>, 1>& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TWeightsAry<VECTOR>& weight) {
        return hasSeasonalVarianceScale(weight);
    });
}

//! Get a weights array with count variance scale \p weight.
template<typename VECTOR>
TWeightsAry<VECTOR> countVarianceScaleWeight(const VECTOR& weight) {
    TWeightsAry<VECTOR> result(CUnitWeights::unit<VECTOR>(weight));
    result[E_SampleCountVarianceScaleWeight] = weight;
    return result;
}

//! Get a weights vector with count variance scale \p weight.
MATHS_EXPORT
TDoubleWeightsAry countVarianceScaleWeight(double weight);

//! Get a weights vector with count variance scale \p weight.
MATHS_EXPORT
TDouble10VecWeightsAry countVarianceScaleWeight(double weight, std::size_t dimension);

//! Set the count variance scale weight in \p weights to \p weight.
template<typename VECTOR>
void setCountVarianceScale(const VECTOR& weight, TWeightsAry<VECTOR>& weights) {
    weights[E_SampleCountVarianceScaleWeight] = weight;
}

//! Set the count variance scale weight in \p weights to \p weight.
MATHS_EXPORT
void setCountVarianceScale(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights);

//! Multiply the count variance scale of \p weights by \p weight.
template<typename VECTOR>
void multiplyCountVarianceScale(const VECTOR& weight, TWeightsAry<VECTOR>& weights) {
    maths_types_detail::SWeightArithmetic<VECTOR>::multiply(
        weight, weights[E_SampleCountVarianceScaleWeight]);
}

//! Extract the variance scale from a collection of weights.
template<typename VECTOR>
const VECTOR& countVarianceScale(const TWeightsAry<VECTOR>& weights) {
    return weights[E_SampleCountVarianceScaleWeight];
}

//! Check if a non-unit count variance scale applies.
MATHS_EXPORT
bool hasCountVarianceScale(const TDoubleWeightsAry& weights);

//! Check if a non-unit seasonal variance scale applies.
MATHS_EXPORT
bool hasCountVarianceScale(const TDoubleWeightsAry1Vec& weights);

//! Check if a non-unit seasonal variance scale applies.
template<typename VECTOR>
bool hasCountVarianceScale(const TWeightsAry<VECTOR>& weights) {
    return std::any_of(weights[E_SampleCountVarianceScaleWeight].begin(),
                       weights[E_SampleCountVarianceScaleWeight].end(),
                       [](double weight) { return weight != 1.0; });
}

//! Check if a non-unit seasonal variance scale applies.
template<typename VECTOR>
bool hasCountVarianceScale(const core::CSmallVector<TWeightsAry<VECTOR>, 1>& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TWeightsAry<VECTOR>& weight) {
        return hasCountVarianceScale(weight);
    });
}

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
enum EFloatingPointErrorStatus {
    E_FpNoErrors = 0x0,
    E_FpOverflowed = 0x1,
    E_FpFailed = 0x2,
    E_FpAllErrors = 0x3
};

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
enum ETail {
    E_UndeterminedTail = 0x0,
    E_LeftTail = 0x1,
    E_RightTail = 0x2,
    E_MixedOrNeitherTail = 0x3
};
}
}

#endif // INCLUDED_ml_maths_t_MathsTypes_h
