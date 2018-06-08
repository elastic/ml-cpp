/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultivariateOneOfNPriorFactory.h>

#include <maths/CMultivariateOneOfNPrior.h>

#include <boost/make_unique.hpp>

namespace ml {
namespace maths {

CMultivariateOneOfNPriorFactory::TPriorPtr
CMultivariateOneOfNPriorFactory::nonInformative(std::size_t dimension,
                                                maths_t::EDataType dataType,
                                                double decayRate,
                                                const TPriorPtrVec& models) {
    return boost::make_unique<CMultivariateOneOfNPrior>(dimension, models, dataType, decayRate);
}

bool CMultivariateOneOfNPriorFactory::restore(std::size_t dimension,
                                              const SDistributionRestoreParams& params,
                                              TPriorPtr& ptr,
                                              core::CStateRestoreTraverser& traverser) {
    ptr = boost::make_unique<CMultivariateOneOfNPrior>(dimension, params, traverser);
    return true;
}
}
}
