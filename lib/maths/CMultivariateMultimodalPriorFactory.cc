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
