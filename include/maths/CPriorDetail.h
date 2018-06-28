/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CCompositeFunctions.h>
#include <maths/CIntegration.h>
#include <maths/CPrior.h>

namespace ml {
namespace maths {

template<typename F, typename T>
bool CPrior::expectation(const F& f,
                         std::size_t numberIntervals,
                         T& result,
                         const TDoubleWeightsAry& weight) const {

    if (numberIntervals == 0) {
        LOG_ERROR(<< "Must specify non-zero number of intervals");
        return false;
    }

    result = T();

    double n{static_cast<double>(numberIntervals)};
    TDoubleDoublePr interval{this->marginalLikelihoodConfidenceInterval(
        100.0 - 1.0 / (100.0 * n), weight)};
    double x{interval.first};
    double dx{(interval.second - interval.first) / n};

    double Z{0.0};
    TDoubleWeightsAry1Vec weights{weight};
    CPrior::CLogMarginalLikelihood logLikelihood(*this, weights);
    CCompositeFunctions::CExp<const CPrior::CLogMarginalLikelihood&> likelihood(logLikelihood);
    for (std::size_t i = 0u; i < numberIntervals; ++i, x += dx) {
        T productIntegral;
        T fIntegral;
        double likelihoodIntegral;
        if (!CIntegration::productGaussLegendre<CIntegration::OrderThree>(
                f, likelihood, x, x + dx, productIntegral, fIntegral, likelihoodIntegral)) {
            result = T();
            return false;
        }
        result += productIntegral;
        Z += likelihoodIntegral;
    }
    result /= Z;
    return true;
}

} // maths
} // ml
