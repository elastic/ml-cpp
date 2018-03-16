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

#include <maths/CXMeansOnlineFactory.h>

#include <core/CStateRestoreTraverser.h>

#include <maths/CXMeansOnline.h>

namespace ml {
namespace maths {
namespace xmeans_online_factory_detail {

#define XMEANS_FACTORY(T, N)                                                                                           \
    CClusterer<CVectorNx1<T, N>>* CFactory<T, N>::make(maths_t::EDataType dataType,                                    \
                                                       maths_t::EClusterWeightCalc weightCalc,                         \
                                                       double decayRate,                                               \
                                                       double minimumClusterFraction,                                  \
                                                       double minimumClusterCount,                                     \
                                                       double minimumCategoryCount) {                                  \
        return new CXMeansOnline<T, N>(                                                                                \
            dataType, weightCalc, decayRate, minimumClusterFraction, minimumClusterCount, minimumCategoryCount);       \
    }                                                                                                                  \
    CClusterer<CVectorNx1<T, N>>* CFactory<T, N>::restore(const SDistributionRestoreParams& params,                    \
                                                          const CClustererTypes::TSplitFunc& splitFunc,                \
                                                          const CClustererTypes::TMergeFunc& mergeFunc,                \
                                                          core::CStateRestoreTraverser& traverser) {                   \
        return new CXMeansOnline<T, N>(params, splitFunc, mergeFunc, traverser);                                       \
    }
XMEANS_FACTORY(CFloatStorage, 2)
XMEANS_FACTORY(CFloatStorage, 3)
XMEANS_FACTORY(CFloatStorage, 4)
XMEANS_FACTORY(CFloatStorage, 5)
#undef XMEANS_FACTORY
}
}
}
