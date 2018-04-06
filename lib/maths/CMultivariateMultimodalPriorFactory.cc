/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultivariateMultimodalPriorFactory.h>

#include <maths/CMultivariateMultimodalPrior.h>
#include <maths/CXMeansOnlineFactory.h>

#include <boost/scoped_ptr.hpp>

namespace ml {
namespace maths {

namespace {

template<std::size_t N>
class CFactory {
public:
    static CMultivariateMultimodalPrior<N>* make(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser) {
        return new CMultivariateMultimodalPrior<N>(params, traverser);
    }

    static CMultivariateMultimodalPrior<N>* make(maths_t::EDataType dataType,
                                                 double decayRate,
                                                 maths_t::EClusterWeightCalc weightCalc,
                                                 double minimumClusterFraction,
                                                 double minimumClusterCount,
                                                 double minimumCategoryCount,
                                                 const CMultivariatePrior& seedPrior) {
        boost::scoped_ptr<CClusterer<CVectorNx1<CFloatStorage, N>>> clusterer(CXMeansOnlineFactory::make<CFloatStorage, N>(
            dataType, weightCalc, decayRate, minimumClusterFraction, minimumClusterCount, minimumCategoryCount));
        return new CMultivariateMultimodalPrior<N>(dataType, *clusterer, seedPrior, decayRate);
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

CMultivariateMultimodalPriorFactory::TPriorPtr CMultivariateMultimodalPriorFactory::nonInformative(std::size_t dimension,
                                                                                                   maths_t::EDataType dataType,
                                                                                                   double decayRate,
                                                                                                   maths_t::EClusterWeightCalc weightCalc,
                                                                                                   double minimumClusterFraction,
                                                                                                   double minimumClusterCount,
                                                                                                   double minimumCategoryCount,
                                                                                                   const CMultivariatePrior& seedPrior) {
    TPriorPtr ptr;
#define FACTORY_ARGS dataType, decayRate, weightCalc, minimumClusterFraction, minimumClusterCount, minimumCategoryCount, seedPrior
    CREATE_PRIOR(dimension)
#undef FACTORY_ARGS
    return ptr;
}

bool CMultivariateMultimodalPriorFactory::restore(std::size_t dimension,
                                                  const SDistributionRestoreParams& params,
                                                  TPriorPtr& ptr,
                                                  core::CStateRestoreTraverser& traverser) {
    ptr.reset();
#define FACTORY_ARGS params, traverser
    CREATE_PRIOR(dimension)
#undef FACTORY_ARGS
    return ptr != nullptr;
}

#undef CREATE_PRIOR
}
}
