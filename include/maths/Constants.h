/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_Constants_h
#define INCLUDED_ml_maths_Constants_h

#include <core/CSmallVector.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

namespace ml {
namespace maths {

//! The minimum coefficient of variation supported by the models.
//! In general, if the coefficient of variation for the data becomes
//! too small we run into numerical problems in the analytics. So,
//! in addSamples we effectively add on variation in the data on the
//! order of this value. This is scale invariant since it includes
//! the sample mean. However, it means we are insensitive anomalous
//! deviations in data whose variation is significantly smaller than
//! this minimum value.
const double MINIMUM_COEFFICIENT_OF_VARIATION{1e-4};

//! The largest probability for which an event is considered anomalous
//! enough to be worthwhile showing a user.
const double LARGEST_SIGNIFICANT_PROBABILITY{0.05};

//! The largest probability that it is deemed significantly anomalous.
const double SMALL_PROBABILITY{1e-4};

//! The largest probability that it is deemed extremely anomalous.
//! Probabilities smaller than this are only weakly discriminated
//! in the sense that they are given the correct order, but fairly
//! similar score.
const double MINUSCULE_PROBABILITY{1e-50};

//! The margin between the smallest value and the support left end
//! to use for the gamma distribution.
const double GAMMA_OFFSET_MARGIN{0.1};

//! The margin between the smallest value and the support left end
//! to use for the log-normal distribution.
const double LOG_NORMAL_OFFSET_MARGIN{1.0};

//! The minimum amount by which a trend decomposition component can
//! reduce the prediction error variance and still be worthwhile
//! modeling. We have different thresholds because we have inductive
//! bias for particular types of components.
const double SIGNIFICANT_VARIANCE_REDUCTION[]{0.7, 0.5};

//! The minimum repeated amplitude of a seasonal component, as a
//! multiple of error standard deviation, to be worthwhile modeling.
//! We have different thresholds because we have inductive bias for
//! particular types of components.
const double SIGNIFICANT_AMPLITUDE[]{1.0, 2.0};

//! The minimum autocorrelation of a seasonal component to be
//! worthwhile modeling. We have different thresholds because we
//! have inductive bias for particular types of components.
const double SIGNIFICANT_AUTOCORRELATION[]{0.5, 0.7};

//! The maximum significance of a test statistic to choose to model
//! a trend decomposition component.
const double MAXIMUM_SIGNIFICANCE{0.001};

//! The minimum variance scale for which the likelihood function
//! can be accurately adjusted. For smaller scales errors are
//! introduced for some priors.
const double MINIMUM_ACCURATE_VARIANCE_SCALE{0.5};

//! The maximum variance scale for which the likelihood function
//! can be accurately adjusted. For larger scales errors are
//! introduced for some priors.
const double MAXIMUM_ACCURATE_VARIANCE_SCALE{2.0};

//! The confidence interval to use for the seasonal trend and
//! variation. We detrend to the nearest point in the confidence
//! interval and use the upper confidence interval variance when
//! scaling the likelihood function so that we don't get transient
//! anomalies after detecting a periodic trend (when the trend
//! can be in significant error).
const double DEFAULT_SEASONAL_CONFIDENCE_INTERVAL{50.0};

//! \brief A collection of weight styles and weights.
class MATHS_EXPORT CConstantWeights {
public:
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble4Vec = core::CSmallVector<double, 4>;
    using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
    using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
    using TDouble2Vec4Vec1Vec = core::CSmallVector<TDouble2Vec4Vec, 1>;

public:
    //! A single count weight style.
    static const maths_t::TWeightStyleVec COUNT;
    //! A single count variance weight style.
    static const maths_t::TWeightStyleVec COUNT_VARIANCE;
    //! A single seasonal variance weight style.
    static const maths_t::TWeightStyleVec SEASONAL_VARIANCE;
    //! A unit weight.
    static const TDouble4Vec UNIT;
    //! A single unit weight.
    static const TDouble4Vec1Vec SINGLE_UNIT;
    //! Get a unit weight for data with \p dimension.
    template<typename VECTOR>
    static core::CSmallVector<VECTOR, 4> unit(std::size_t dimension) {
        return TDouble2Vec4Vec{VECTOR(dimension, 1.0)};
    }
    //! Get a single unit weight for data with \p dimension.
    template<typename VECTOR>
    static core::CSmallVector<core::CSmallVector<VECTOR, 4>, 1> singleUnit(std::size_t dimension) {
        return core::CSmallVector<core::CSmallVector<VECTOR, 4>, 1>{core::CSmallVector<VECTOR, 4>{VECTOR(dimension, 1.0)}};
    }
};

//! The minimum fractional count of points in a cluster.
const double MINIMUM_CLUSTER_SPLIT_FRACTION{0.0};

//! The default minimum count of points in a cluster.
const double MINIMUM_CLUSTER_SPLIT_COUNT{24.0};

//! The minimum count of a category in the sketch to cluster.
const double MINIMUM_CATEGORY_COUNT{0.5};

//! Get the maximum amount we'll penalize a model in addSamples.
MATHS_EXPORT double maxModelPenalty(double numberSamples);
}
}

#endif // INCLUDED_ml_maths_Constants_h
