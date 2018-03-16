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

#ifndef INCLUDED_ml_maths_CXMeansOnlineFactory_h
#define INCLUDED_ml_maths_CXMeansOnlineFactory_h

#include <maths/CClusterer.h>
#include <maths/CLinearAlgebraFwd.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/static_assert.hpp>

#include <cstddef>

namespace ml {
namespace core {
class CStateRestoreTraverser;
}
namespace maths {
template<typename POINT>
class CClusterer;
struct SDistributionRestoreParams;

namespace xmeans_online_factory_detail {
template<typename T, std::size_t N>
class CFactory {};

#define XMEANS_FACTORY(T, N)                                                                                           \
    template<>                                                                                                         \
    class MATHS_EXPORT CFactory<T, N> {                                                                                \
    public:                                                                                                            \
        static CClusterer<CVectorNx1<T, N>>* make(maths_t::EDataType dataType,                                         \
                                                  maths_t::EClusterWeightCalc weightCalc,                              \
                                                  double decayRate,                                                    \
                                                  double minimumClusterFraction,                                       \
                                                  double minimumClusterCount,                                          \
                                                  double minimumCategoryCount);                                        \
        static CClusterer<CVectorNx1<T, N>>* restore(const SDistributionRestoreParams& params,                         \
                                                     const CClustererTypes::TSplitFunc& splitFunc,                     \
                                                     const CClustererTypes::TMergeFunc& mergeFunc,                     \
                                                     core::CStateRestoreTraverser& traverser);                         \
    }
XMEANS_FACTORY(CFloatStorage, 2);
XMEANS_FACTORY(CFloatStorage, 3);
XMEANS_FACTORY(CFloatStorage, 4);
XMEANS_FACTORY(CFloatStorage, 5);
#undef XMEANS_FACTORY
}

//! \brief Factory for multivariate x-means online clusterers.
class MATHS_EXPORT CXMeansOnlineFactory {
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
            dataType, weightCalc, decayRate, minimumClusterFraction, minimumClusterCount, minimumCategoryCount);
    }

    //! Construct by traversing a state document.
    template<typename T, std::size_t N>
    static inline CClusterer<CVectorNx1<T, N>>* restore(const SDistributionRestoreParams& params,
                                                        const CClustererTypes::TSplitFunc& splitFunc,
                                                        const CClustererTypes::TMergeFunc& mergeFunc,
                                                        core::CStateRestoreTraverser& traverser) {
        return xmeans_online_factory_detail::CFactory<T, N>::restore(params, splitFunc, mergeFunc, traverser);
    }
};
}
}

#endif // INCLUDED_ml_maths_CXMeansOnlineFactory_h
