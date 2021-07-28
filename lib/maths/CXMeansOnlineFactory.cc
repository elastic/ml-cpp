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

#include <maths/CXMeansOnlineFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/CXMeansOnline.h>

namespace ml {
namespace maths {
namespace xmeans_online_factory_detail {

#define XMEANS_FACTORY(T, N)                                                                    \
    CClusterer<CVectorNx1<T, N>>* CFactory<T, N>::make(                                         \
        maths_t::EDataType dataType, maths_t::EClusterWeightCalc weightCalc,                    \
        double decayRate, double minimumClusterFraction,                                        \
        double minimumClusterCount, double minimumCategoryCount) {                              \
        return new CXMeansOnline<T, N>(dataType, weightCalc, decayRate, minimumClusterFraction, \
                                       minimumClusterCount, minimumCategoryCount);              \
    }                                                                                           \
    CClusterer<CVectorNx1<T, N>>* CFactory<T, N>::restore(                                      \
        const SDistributionRestoreParams& params,                                               \
        const CClustererTypes::TSplitFunc& splitFunc,                                           \
        const CClustererTypes::TMergeFunc& mergeFunc,                                           \
        core::CStateRestoreTraverser& traverser) {                                              \
        return new CXMeansOnline<T, N>(params, splitFunc, mergeFunc, traverser);                \
    }
XMEANS_FACTORY(CFloatStorage, 2)
XMEANS_FACTORY(CFloatStorage, 3)
XMEANS_FACTORY(CFloatStorage, 4)
XMEANS_FACTORY(CFloatStorage, 5)
#undef XMEANS_FACTORY
}
}
}
