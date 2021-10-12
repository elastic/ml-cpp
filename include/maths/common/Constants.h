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

#ifndef INCLUDED_ml_maths_common_Constants_h
#define INCLUDED_ml_maths_common_Constants_h

#include <core/CSmallVector.h>

#include <maths/common/ImportExport.h>
#include <maths/common/MathsTypes.h>

#include <cmath>

namespace ml {
namespace maths {
namespace common {
//! The minimum coefficient of variation supported by the models.
//! In general, if the coefficient of variation for the data becomes
//! too small we run into numerical problems in the analytics. So,
//! in addSamples we effectively add on variation in the data on the
//! order of this value. This is scale invariant since it includes
//! the sample mean. However, it means we are insensitive anomalous
//! deviations in data whose variation is significantly smaller than
//! this minimum value.
constexpr double MINIMUM_COEFFICIENT_OF_VARIATION{1e-4};

//! The largest probability for which an event is considered anomalous
//! enough to be worthwhile showing a user.
constexpr double LARGEST_SIGNIFICANT_PROBABILITY{0.05};

//! The largest probability that it is deemed significantly anomalous.
constexpr double SMALL_PROBABILITY{1e-4};

//! The largest probability that it is deemed extremely anomalous.
//! Probabilities smaller than this are only weakly discriminated
//! in the sense that they are given the correct order, but fairly
//! similar score.
constexpr double MINUSCULE_PROBABILITY{1e-50};

//! The margin between the smallest value and the support left end
//! to use for the gamma distribution.
constexpr double GAMMA_OFFSET_MARGIN{0.1};

//! The margin between the smallest value and the support left end
//! to use for the log-normal distribution.
constexpr double LOG_NORMAL_OFFSET_MARGIN{1.0};

//! The minimum amount by which a trend decomposition component can
//! reduce the prediction error variance and still be worthwhile
//! modeling. We have different thresholds because we have inductive
//! bias for particular types of components.
constexpr double COMPONENT_SIGNIFICANT_VARIANCE_REDUCTION[]{0.6, 0.4};

//! The default number of regression models used in periodic and
//! calendar cyclic components of the trend decomposition.
constexpr std::size_t COMPONENT_SIZE{36};

//! The minimum variance scale for which the likelihood function
//! can be accurately adjusted. For smaller scales errors are
//! introduced for some priors.
constexpr double MINIMUM_ACCURATE_VARIANCE_SCALE{0.5};

//! The maximum variance scale for which the likelihood function
//! can be accurately adjusted. For larger scales errors are
//! introduced for some priors.
constexpr double MAXIMUM_ACCURATE_VARIANCE_SCALE{2.0};

//! The confidence interval to use for the seasonal trend and
//! variation. We detrend to the nearest point in the confidence
//! interval and use the upper confidence interval variance when
//! scaling the likelihood function so that we don't get transient
//! anomalies after detecting a periodic trend (when the trend
//! can be in significant error).
constexpr double DEFAULT_SEASONAL_CONFIDENCE_INTERVAL{50.0};

//! The minimum fractional count of points in a cluster.
constexpr double MINIMUM_CLUSTER_SPLIT_FRACTION{0.0};

//! The default minimum count of points in a cluster.
constexpr double MINIMUM_CLUSTER_SPLIT_COUNT{24.0};

//! The minimum count of a category in the sketch to cluster.
constexpr double MINIMUM_CATEGORY_COUNT{0.5};
}
}
}

#endif // INCLUDED_ml_maths_common_Constants_h
