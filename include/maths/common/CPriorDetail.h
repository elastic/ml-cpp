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

#include <maths/common/CCompositeFunctions.h>
#include <maths/common/CIntegration.h>
#include <maths/common/CPrior.h>

namespace ml {
namespace maths {
namespace common {
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
    for (std::size_t i = 0; i < numberIntervals; ++i, x += dx) {
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
}
} // ml
