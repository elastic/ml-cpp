/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CMultivariateOneOfNPriorFactory.h>

#include <maths/CMultivariateOneOfNPrior.h>

namespace ml {
namespace maths {

CMultivariateOneOfNPriorFactory::TPriorPtr
CMultivariateOneOfNPriorFactory::nonInformative(std::size_t dimension,
                                                maths_t::EDataType dataType,
                                                double decayRate,
                                                const TPriorPtrVec& models) {
    return TPriorPtr(new CMultivariateOneOfNPrior(dimension, models, dataType, decayRate));
}

bool CMultivariateOneOfNPriorFactory::restore(std::size_t dimension,
                                              const SDistributionRestoreParams& params,
                                              TPriorPtr& ptr,
                                              core::CStateRestoreTraverser& traverser) {
    ptr.reset(new CMultivariateOneOfNPrior(dimension, params, traverser));
    return true;
}
}
}
