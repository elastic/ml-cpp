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

#ifndef INCLUDED_ml_maths_common_CModelDetail_h
#define INCLUDED_ml_maths_common_CModelDetail_h

#include <maths/common/CModel.h>
#include <maths/common/CMultivariatePrior.h>
#include <maths/common/CPrior.h>

#include <optional>

namespace ml {
namespace maths {
namespace common {
template<typename VECTOR>
VECTOR CModel::marginalLikelihoodMean(const maths::common::CPrior& prior) {
    return VECTOR{prior.marginalLikelihoodMean()};
}

template<typename VECTOR>
VECTOR CModel::marginalLikelihoodMean(const maths::common::CMultivariatePrior& prior) {
    return prior.marginalLikelihoodMean();
}

template<typename TREND, typename VECTOR>
std::optional<VECTOR> CModel::predictionError(const TREND& trend, const VECTOR& sample) {
    std::optional<VECTOR> result;
    std::size_t dimension = sample.size();
    for (std::size_t i = 0; i < dimension; ++i) {
        if (trend[i]->initialized()) {
            result.emplace(dimension, 0.0);
            for (/**/; i < dimension; ++i) {
                if (trend[i]->initialized()) {
                    (*result)[i] = sample[i];
                }
            }
        }
    }
    return result;
}

template<typename PRIOR, typename VECTOR>
std::optional<VECTOR> CModel::predictionError(double propagationInterval,
                                              const PRIOR& prior,
                                              const VECTOR& sample) {
    std::optional<VECTOR> result;
    if (prior->numberSamples() > 20.0 / propagationInterval) {
        std::size_t dimension{sample.size()};
        result.emplace(sample);
        VECTOR mean(marginalLikelihoodMean<VECTOR>(*prior));
        for (std::size_t d = 0; d < dimension; ++d) {
            (*result)[d] -= mean[d];
        }
    }
    return result;
}
}
}
}

#endif // INCLUDED_ml_maths_common_CModelDetail_h
