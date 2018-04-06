/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultivariateNormalConjugateFactory.h>

#include <maths/CMultivariateNormalConjugate.h>

namespace ml {
namespace maths {

namespace {

template<std::size_t N>
class CFactory {
public:
    static CMultivariateNormalConjugate<N>* make(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser) {
        return new CMultivariateNormalConjugate<N>(params, traverser);
    }

    static CMultivariateNormalConjugate<N>* make(maths_t::EDataType dataType, double decayRate) {
        return CMultivariateNormalConjugate<N>::nonInformativePrior(dataType, decayRate).clone();
    }
};
}

#define CREATE_PRIOR(N)                                                                                                                    \
    switch (N) {                                                                                                                           \
    case 2:                                                                                                                                \
        ptr.reset(CFactory<2>::make(FACTORY_ARGS));                                                                                        \
        break;                                                                                                                             \
    case 3:                                                                                                                                \
        ptr.reset(CFactory<3>::make(FACTORY_ARGS));                                                                                        \
        break;                                                                                                                             \
    case 4:                                                                                                                                \
        ptr.reset(CFactory<4>::make(FACTORY_ARGS));                                                                                        \
        break;                                                                                                                             \
    case 5:                                                                                                                                \
        ptr.reset(CFactory<5>::make(FACTORY_ARGS));                                                                                        \
        break;                                                                                                                             \
    default:                                                                                                                               \
        LOG_ERROR("Unsupported dimension " << N);                                                                                          \
        break;                                                                                                                             \
    }

CMultivariateNormalConjugateFactory::TPriorPtr
CMultivariateNormalConjugateFactory::nonInformative(std::size_t dimension, maths_t::EDataType dataType, double decayRate) {
    TPriorPtr ptr;
#define FACTORY_ARGS dataType, decayRate
    CREATE_PRIOR(dimension);
#undef FACTORY_ARGS
    return ptr;
}

bool CMultivariateNormalConjugateFactory::restore(std::size_t dimension,
                                                  const SDistributionRestoreParams& params,
                                                  TPriorPtr& ptr,
                                                  core::CStateRestoreTraverser& traverser) {
    ptr.reset();
#define FACTORY_ARGS params, traverser
    CREATE_PRIOR(dimension);
#undef FACTORY_ARGS
    return ptr != nullptr;
}

#undef CREATE_PRIOR
}
}
