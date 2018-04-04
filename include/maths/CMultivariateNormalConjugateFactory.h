/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMultivariateNormalConjugateFactory_h
#define INCLUDED_ml_maths_CMultivariateNormalConjugateFactory_h

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/shared_ptr.hpp>

#include <cstddef>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}

namespace maths {
class CMultivariatePrior;
struct SDistributionRestoreParams;

//! \brief Factory for multivariate normal conjugate priors.
class MATHS_EXPORT CMultivariateNormalConjugateFactory {
public:
    using TPriorPtr = boost::shared_ptr<CMultivariatePrior>;

public:
    //! Create a new non-informative multivariate normal prior.
    static TPriorPtr nonInformative(std::size_t dimension, maths_t::EDataType dataType, double decayRate);

    //! Create reading state from its state document representation.
    static bool
    restore(std::size_t dimenion, const SDistributionRestoreParams& params, TPriorPtr& ptr, core::CStateRestoreTraverser& traverser);
};
}
}

#endif // INCLUDED_ml_maths_CPriorStateSerialiserMultivariateNormal_h
