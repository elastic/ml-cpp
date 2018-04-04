/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CModelDetail_h
#define INCLUDED_ml_maths_CModelDetail_h

#include <maths/CModel.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CPrior.h>

#include <boost/optional.hpp>

namespace ml {
namespace maths {

template<typename VECTOR>
VECTOR CModel::marginalLikelihoodMean(const maths::CPrior& prior) {
    return VECTOR{prior.marginalLikelihoodMean()};
}

template<typename VECTOR>
VECTOR CModel::marginalLikelihoodMean(const maths::CMultivariatePrior& prior) {
    return prior.marginalLikelihoodMean();
}

template<typename TREND, typename VECTOR>
boost::optional<VECTOR> CModel::predictionError(const TREND& trend, const VECTOR& sample) {
    boost::optional<VECTOR> result;
    std::size_t dimension = sample.size();
    for (std::size_t i = 0u; i < dimension; ++i) {
        if (trend[i]->initialized()) {
            result.reset(VECTOR(dimension, 0.0));
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
boost::optional<VECTOR> CModel::predictionError(double propagationInterval, const PRIOR& prior, const VECTOR& sample) {
    boost::optional<VECTOR> result;
    if (prior->numberSamples() > 20.0 / propagationInterval) {
        std::size_t dimension{sample.size()};
        result.reset(sample);
        VECTOR mean(marginalLikelihoodMean<VECTOR>(*prior));
        for (std::size_t d = 0u; d < dimension; ++d) {
            (*result)[d] -= mean[d];
        }
    }
    return result;
}
}
}

#endif // INCLUDED_ml_maths_CModelDetail_h
