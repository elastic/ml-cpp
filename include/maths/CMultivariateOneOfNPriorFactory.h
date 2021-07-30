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

#ifndef INCLUDED_ml_maths_CMultivariateOneOfNPriorFactory_h
#define INCLUDED_ml_maths_CMultivariateOneOfNPriorFactory_h

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <cstddef>
#include <memory>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}

namespace maths {
class CMultivariatePrior;
struct SDistributionRestoreParams;

//! \brief Factory for multivariate 1-of-n priors.
class MATHS_EXPORT CMultivariateOneOfNPriorFactory {
public:
    using TPriorPtr = std::unique_ptr<CMultivariatePrior>;
    using TPriorPtrVec = std::vector<TPriorPtr>;

public:
    //! Create a new non-informative multivariate normal prior.
    static TPriorPtr nonInformative(std::size_t dimension,
                                    maths_t::EDataType dataType,
                                    double decayRate,
                                    const TPriorPtrVec& models);

    //! Create reading state from its state document representation.
    static bool restore(std::size_t dimension,
                        const SDistributionRestoreParams& params,
                        TPriorPtr& ptr,
                        core::CStateRestoreTraverser& traverser);
};
}
}

#endif // INCLUDED_ml_maths_CMultivariateOneOfNPriorFactory_h
