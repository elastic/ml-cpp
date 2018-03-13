/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_maths_CRestoreParams_h
#define INCLUDED_ml_maths_CRestoreParams_h

#include <core/CoreTypes.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/ref.hpp>

namespace ml {
namespace maths {
class CModelParams;

//! \brief Gatherers up extra parameters supplied when restoring
//! time series decompositions.
struct MATHS_EXPORT STimeSeriesDecompositionRestoreParams {
    STimeSeriesDecompositionRestoreParams(double decayRate,
                                          core_t::TTime minimumBucketLength,
                                          std::size_t componentSize);

    //! The rate at which decomposition loses information.
    double s_DecayRate;

    //! The data bucket length.
    core_t::TTime s_MinimumBucketLength;

    //! The decomposition seasonal component size.
    std::size_t s_ComponentSize;
};

//! \brief Gatherers up extra parameters supplied when restoring
//! distribution models.
struct MATHS_EXPORT SDistributionRestoreParams {
    SDistributionRestoreParams(maths_t::EDataType dataType,
                               double decayRate,
                               double minimumClusterFraction,
                               double minimumClusterCount,
                               double minimumCategoryCount);

    //! The type of data being clustered.
    maths_t::EDataType s_DataType;

    //! The rate at which cluster priors decay to non-informative.
    double s_DecayRate;

    //! The minimum cluster fractional count.
    double s_MinimumClusterFraction;

    //! The minimum cluster count.
    double s_MinimumClusterCount;

    //! The minimum count for a category in the sketch to cluster.
    double s_MinimumCategoryCount;
};

//! \brief Gatherers up extra parameters supplied when restoring
//! time series decompositions.
struct MATHS_EXPORT SModelRestoreParams {
    using TModelParamsCRef = boost::reference_wrapper<const CModelParams>;

    SModelRestoreParams(const CModelParams &params,
                        const STimeSeriesDecompositionRestoreParams &decompositionParams,
                        const SDistributionRestoreParams &distributionParams);

    //! The model parameters.
    TModelParamsCRef s_Params;

    //! The time series decomposition restore parameters.
    STimeSeriesDecompositionRestoreParams s_DecompositionParams;

    //! The time series decomposition restore parameters.
    SDistributionRestoreParams s_DistributionParams;
};
}
}

#endif// INCLUDED_ml_maths_CRestoreParams_h
