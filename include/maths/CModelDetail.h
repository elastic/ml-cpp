/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
boost::optional<VECTOR>
CModel::predictionError(double propagationInterval, const PRIOR& prior, const VECTOR& sample) {
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
