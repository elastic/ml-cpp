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

#ifndef INCLUDED_ml_maths_common_t_MathsTypesDetail_h
#define INCLUDED_ml_maths_common_t_MathsTypesDetail_h

#include <maths/common/MathsTypes.h>

#include <algorithm>

namespace ml {
namespace maths_t {
template<typename VECTOR>
bool isWinsorised(const TWeightsAry<VECTOR>& weights) {
    return std::any_of(weights[E_SampleOutlierWeight].begin(),
                       weights[E_SampleOutlierWeight].end(),
                       [](double weight) { return weight != 1.0; });
}

template<typename VECTOR>
bool isWinsorised(const core::CSmallVector<TWeightsAry<VECTOR>, 1>& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TWeightsAry<VECTOR>& weight) {
        return isWinsorised(weight);
    });
}

template<typename VECTOR>
bool hasSeasonalVarianceScale(const TWeightsAry<VECTOR>& weights) {
    return std::any_of(weights[E_SampleSeasonalVarianceScaleWeight].begin(),
                       weights[E_SampleSeasonalVarianceScaleWeight].end(),
                       [](double weight) { return weight != 1.0; });
}

template<typename VECTOR>
bool hasSeasonalVarianceScale(const core::CSmallVector<TWeightsAry<VECTOR>, 1>& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TWeightsAry<VECTOR>& weight) {
        return hasSeasonalVarianceScale(weight);
    });
}

template<typename VECTOR>
bool hasCountVarianceScale(const TWeightsAry<VECTOR>& weights) {
    return std::any_of(weights[E_SampleCountVarianceScaleWeight].begin(),
                       weights[E_SampleCountVarianceScaleWeight].end(),
                       [](double weight) { return weight != 1.0; });
}

template<typename VECTOR>
bool hasCountVarianceScale(const core::CSmallVector<TWeightsAry<VECTOR>, 1>& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TWeightsAry<VECTOR>& weight) {
        return hasCountVarianceScale(weight);
    });
}
}
}

#endif // INCLUDED_ml_maths_common_t_MathsTypesDetail_h
