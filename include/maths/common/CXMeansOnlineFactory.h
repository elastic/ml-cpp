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

#ifndef INCLUDED_ml_maths_common_CXMeansOnlineFactory_h
#define INCLUDED_ml_maths_common_CXMeansOnlineFactory_h

#include <maths/common/CClusterer.h>
#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/ImportExport.h>
#include <maths/common/MathsTypes.h>

#include <boost/static_assert.hpp>

#include <cstddef>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}
namespace maths {
namespace common {
template<typename POINT>
class CClusterer;
struct SDistributionRestoreParams;

namespace xmeans_online_factory_detail {
template<typename T, std::size_t N>
class CFactory {};

#define XMEANS_FACTORY(T, N)                                                              \
    template<>                                                                            \
    class MATHS_COMMON_EXPORT CFactory<T, N> {                                            \
    public:                                                                               \
        static CClusterer<CVectorNx1<T, N>>* make(maths_t::EDataType dataType,            \
                                                  maths_t::EClusterWeightCalc weightCalc, \
                                                  double decayRate,                       \
                                                  double minimumClusterFraction,          \
                                                  double minimumClusterCount,             \
                                                  double minimumCategoryCount);           \
        static CClusterer<CVectorNx1<T, N>>*                                              \
        restore(const SDistributionRestoreParams& params,                                 \
                const CClustererTypes::TSplitFunc& splitFunc,                             \
                const CClustererTypes::TMergeFunc& mergeFunc,                             \
                core::CStateRestoreTraverser& traverser);                                 \
    }
XMEANS_FACTORY(CFloatStorage, 2);
XMEANS_FACTORY(CFloatStorage, 3);
XMEANS_FACTORY(CFloatStorage, 4);
XMEANS_FACTORY(CFloatStorage, 5);
#undef XMEANS_FACTORY
}

//! \brief Factory for multivariate x-means online clusterers.
class MATHS_COMMON_EXPORT CXMeansOnlineFactory {
public:
    //! Create a new x-means clusterer.
    //!
    //! \param[in] dataType The type of data which will be clustered.
    //! \param[in] weightCalc The style of the cluster weight calculation
    //! (see maths_t::EClusterWeightCalc for details).
    //! \param[in] decayRate Controls the rate at which information is
    //! lost from the clusters.
    //! \param[in] minimumClusterFraction The minimum fractional count
    //! of points in a cluster.
    //! \param[in] minimumClusterCount The minimum count of points in a
    //! cluster.
    template<typename T, std::size_t N>
    static inline CClusterer<CVectorNx1<T, N>>* make(maths_t::EDataType dataType,
                                                     maths_t::EClusterWeightCalc weightCalc,
                                                     double decayRate,
                                                     double minimumClusterFraction,
                                                     double minimumClusterCount,
                                                     double minimumCategoryCount) {
        return xmeans_online_factory_detail::CFactory<T, N>::make(
            dataType, weightCalc, decayRate, minimumClusterFraction,
            minimumClusterCount, minimumCategoryCount);
    }

    //! Construct by traversing a state document.
    template<typename T, std::size_t N>
    static inline CClusterer<CVectorNx1<T, N>>*
    restore(const SDistributionRestoreParams& params,
            const CClustererTypes::TSplitFunc& splitFunc,
            const CClustererTypes::TMergeFunc& mergeFunc,
            core::CStateRestoreTraverser& traverser) {
        return xmeans_online_factory_detail::CFactory<T, N>::restore(
            params, splitFunc, mergeFunc, traverser);
    }
};
}
}
}

#endif // INCLUDED_ml_maths_common_CXMeansOnlineFactory_h
