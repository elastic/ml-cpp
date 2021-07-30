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

#include <maths/CMultivariateOneOfNPriorFactory.h>

#include <maths/CMultivariateOneOfNPrior.h>

namespace ml {
namespace maths {

CMultivariateOneOfNPriorFactory::TPriorPtr
CMultivariateOneOfNPriorFactory::nonInformative(std::size_t dimension,
                                                maths_t::EDataType dataType,
                                                double decayRate,
                                                const TPriorPtrVec& models) {
    return std::make_unique<CMultivariateOneOfNPrior>(dimension, models, dataType, decayRate);
}

bool CMultivariateOneOfNPriorFactory::restore(std::size_t dimension,
                                              const SDistributionRestoreParams& params,
                                              TPriorPtr& ptr,
                                              core::CStateRestoreTraverser& traverser) {
    ptr = std::make_unique<CMultivariateOneOfNPrior>(dimension, params, traverser);
    return true;
}
}
}
